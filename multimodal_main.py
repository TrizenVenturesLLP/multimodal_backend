from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import numpy as np
import sys
import uuid
import time
import logging
import subprocess
import json
import asyncio
import shutil
from datetime import datetime
from dotenv import load_dotenv
import uvicorn

# Load environment variables from consolidated .env
load_dotenv()

# Add subdirectories to sys.path to allow imports from them
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "video"))

# Import specialized services
from audio.services.audio_emotion_detector import AudioEmotionDetector
from video.services.optimized_emotion_detector import OptimizedEmotionDetector
from text.services.text_processor import TextProcessor
from services.fusion_logic import FusionLogicService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger("MultimodalAPI")

# Redirect system temp storage
import tempfile
tempfile.tempdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_cache")
os.makedirs(tempfile.tempdir, exist_ok=True)
logger.info(f"Redirected system temp storage to: {tempfile.tempdir}")

app = FastAPI(
    title="Multimodal Emotion Analysis API",
    description="Unified Audio + Video + text emotion detection and grading (Async Support)",
    version="1.2.1"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job tracking (In-memory for status, file-based for results)
jobs_status = {}

# Fix for Swagger UI when behind a reverse proxy (CapRover)
@app.middleware("http")
async def fix_proxy_headers(request: Request, call_next):
    proto = request.headers.get("X-Forwarded-Proto")
    if proto:
        request.scope["scheme"] = proto
    return await call_next(request)

# Directory configuration
TEMP_VIDEO_DIR = "temp_videos"
TEMP_AUDIO_DIR = "temp_audio"
TEMP_RESULTS_DIR = "temp_results"
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
os.makedirs(TEMP_RESULTS_DIR, exist_ok=True)

# Initialize global services
audio_detector = AudioEmotionDetector()
text_processor = TextProcessor()
video_detector = OptimizedEmotionDetector()

def get_ffmpeg_path():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return 'ffmpeg'
    except:
        pass
    if os.name == 'nt':
        username = os.getenv('USERNAME') or os.getenv('USER')
        winget_path = f"C:\\Users\\{username}\\AppData\\Local\\Microsoft\\WinGet\\Packages"
        if os.path.exists(winget_path):
            for root, dirs, files in os.walk(winget_path):
                if 'ffmpeg.exe' in files:
                    return os.path.join(root, 'ffmpeg.exe')
    return 'ffmpeg'

FFMPEG_EXE = get_ffmpeg_path()

async def process_analysis_job(job_id: str, video_path: str, filename: str):
    """Background task to run the full pipeline"""
    jobs_status[job_id] = {"status": "processing", "progress": 10, "step": "Initializing", "started_at": datetime.now().isoformat()}
    audio_path = os.path.join(TEMP_AUDIO_DIR, f"{job_id}.wav")
    
    try:
        # Step 1: Extract Audio
        jobs_status[job_id].update({"progress": 20, "step": "Extracting audio"})
        logger.info(f"[{job_id}] Extracting audio...")
        command = [FFMPEG_EXE, '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1', audio_path]
        subprocess.run(command, capture_output=True, check=True)

        # Step 2: Sequential Model Inference (Memory Optimized)
        loop = asyncio.get_event_loop()
        
        # 2a. Video Analysis
        jobs_status[job_id].update({"progress": 30, "step": "Analyzing Video"})
        logger.info(f"[{job_id}] Starting Video Inference...")
        video_results = await loop.run_in_executor(None, video_detector.process_video, video_path)
        
        # 2b. Audio Analysis
        jobs_status[job_id].update({"progress": 50, "step": "Analyzing Audio"})
        logger.info(f"[{job_id}] Starting Audio Inference...")
        audio_results = await loop.run_in_executor(None, audio_detector.process_audio, audio_path)
        
        # 2c. Text & Transcription
        jobs_status[job_id].update({"progress": 70, "step": "Analyzing Transcription & Content"})
        logger.info(f"[{job_id}] Starting Text Inference (WhisperX)...")
        text_results = await loop.run_in_executor(None, text_processor.process_audio, audio_path)
        
        # Step 3: Fusion & Analysis
        jobs_status[job_id].update({"progress": 80, "step": "Fusing metrics"})
        audio_probs = audio_results["probabilities"]
        video_probs_list = [f['probabilities'] for f in video_results.get('emotion_data', []) if 'probabilities' in f]
        video_labels_list = [f['emotion'] for f in video_results.get('emotion_data', [])]
        
        if not video_probs_list: 
            dist = video_results['emotion_distribution']
            video_probs_list = [[(dist.get(emo, 0) / 100.0) for emo in FusionLogicService.STANDARD_EMOTIONS]]
        
        audio_metrics = FusionLogicService.calculate_audio_rubrics(audio_probs, labels=list(audio_results['emotion_distribution'].keys()))
        video_metrics, video_avg_probs = FusionLogicService.calculate_video_rubrics(video_probs_list, video_labels_list)
        
        text_analysis_results = text_processor.analyze_text_metrics(text_results.get("qa_pairs", []))
        duration = video_results.get("processing_stats", {}).get("duration", 0)
        text_metrics, text_overall_score, text_stats = FusionLogicService.calculate_text_rubrics(text_analysis_results, duration)

        fusion_results = FusionLogicService.fuse_metrics(audio_metrics, video_metrics, text_metrics, audio_probs, video_avg_probs, text_overall_score)
        
        # Step 4: Evidence & Timeline
        jobs_status[job_id].update({"progress": 85, "step": "Extracting evidence"})
        evidence_log = FusionLogicService.extract_xai_evidence(
            audio_results.get("chunk_data", []), 
            video_results.get("emotion_data", []), 
            audio_metrics, 
            video_metrics, 
            fusion_results["alignment_score"],
            text_analysis_results
        )
        
        timeline_data = FusionLogicService.generate_timeline_data(
            audio_results.get("chunk_data", []),
            video_results.get("emotion_data", [])
        )

        # Step 5: Feedback Generation
        jobs_status[job_id].update({"progress": 90, "step": "Generating feedback"})
        ai_feedback = await FusionLogicService.generate_multimodal_feedback(
            audio_metrics, video_metrics, text_metrics, fusion_results["final_score"], fusion_results["alignment_score"]
        )

        # Attach rubrics to individual analyses for frontend compatibility
        video_results["rubrics"] = {k: {"score": v, "justification": fusion_results["fused_justifications"].get(k, "Analysis complete.")} for k, v in video_metrics.items()}
        # Add 'Overall' for Dashboard.tsx compatibility
        video_results["rubrics"]["Overall"] = {"score": video_metrics.get("Confidence", 50), "justification": "Overall visual performance."}
        
        audio_results["rubrics"] = {k: {"score": v, "justification": fusion_results["fused_justifications"].get(k, "Analysis complete.")} for k, v in audio_metrics.items()}
        audio_results["rubrics"]["Overall"] = {"score": audio_metrics.get("Vocal Confidence", 50), "justification": "Overall vocal performance."}
        
        text_results["rubrics"] = {k.replace("Textual ", ""): {"score": v, "justification": "Content analysis complete."} for k, v in text_metrics.items()}
        text_results["overall_score"] = text_overall_score

        # Final Response
        response = {
            "job_id": job_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "multimodal_score": fusion_results["final_score"],
            "overall_feedback": ai_feedback,
            "alignment_score": fusion_results["alignment_score"],
            "rubric_scores": fusion_results["fused_rubrics"],
            "rubric_justifications": fusion_results["fused_justifications"],
            "evidence_log": evidence_log,
            "timeline_data": timeline_data,
            "transcription": text_results.get("transcription", ""),
            "qa_pairs": text_results.get("qa_pairs", []),
            "individual_analysis": {
                "video": video_results,
                "audio": audio_results,
                "text": text_results
            }
        }

        # Save to disk
        result_file = os.path.join(TEMP_RESULTS_DIR, f"{job_id}.json")
        with open(result_file, "w") as f:
            json.dump(response, f)
            
        jobs_status[job_id] = {"status": "completed", "completed_at": datetime.now().isoformat()}
        logger.info(f"[{job_id}] Job completed.")

    except Exception as e:
        logger.error(f"[{job_id}] Job failed: {str(e)}")
        jobs_status[job_id] = {"status": "failed", "error": str(e), "failed_at": datetime.now().isoformat()}
    finally:
        if os.path.exists(video_path): os.remove(video_path)
        if os.path.exists(audio_path): os.remove(audio_path)

@app.get("/")
async def root():
    return {"status": "online", "version": "1.2.1"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/multimodal/analyze")
async def analyze_multimodal(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    """Async analysis endpoint - Returns job_id immediately"""
    job_id = str(uuid.uuid4())
    video_ext = video.filename.split('.')[-1]
    video_path = os.path.join(TEMP_VIDEO_DIR, f"{job_id}.{video_ext}")
    
    try:
        # Stream upload to disk
        with open(video_path, "wb") as f:
            while chunk := await video.read(1024 * 1024):
                f.write(chunk)
        await video.close()
        
        # Start background task
        background_tasks.add_task(process_analysis_job, job_id, video_path, video.filename)
        
        return {
            "job_id": job_id,
            "status": "accepted",
            "message": "Analysis started in background",
            "check_url": f"/api/multimodal/results/{job_id}"
        }
    except Exception as e:
        if os.path.exists(video_path): os.remove(video_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/multimodal/results/{job_id}")
async def get_results(job_id: str):
    """Check status or fetch results"""
    # Check in-memory status
    if job_id in jobs_status:
        job = jobs_status[job_id]
        if job["status"] == "completed":
            result_file = os.path.join(TEMP_RESULTS_DIR, f"{job_id}.json")
            if os.path.exists(result_file):
                with open(result_file, "r") as f:
                    return json.load(f)
        return job

    # Fallback to disk
    result_file = os.path.join(TEMP_RESULTS_DIR, f"{job_id}.json")
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            data = json.load(f)
            data["status"] = "completed"
            return data
            
    raise HTTPException(status_code=404, detail="Job not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
