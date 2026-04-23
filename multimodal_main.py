from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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

# Load environment variables from consolidated .env
load_dotenv()

# Add subdirectories to sys.path to allow imports from them
# This handles the refactored directory structure
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "video"))

# Import specialized services
from audio.services.audio_emotion_detector import AudioEmotionDetector
from video.services.optimized_emotion_detector import OptimizedEmotionDetector
from text.services.text_processor import TextProcessor
from services.fusion_logic import FusionLogicService

# Configure logging to a root app.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger("MultimodalAPI")

# FORCE TEMPORARY STORAGE TO PROJECT DIRECTORY (E: DRIVE)
# This prevents 400 Bad Request errors when the system C: drive is full
import tempfile
tempfile.tempdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_cache")
os.makedirs(tempfile.tempdir, exist_ok=True)
logger.info(f"Redirected system temp storage to: {tempfile.tempdir}")

app = FastAPI(
    title="Multimodal Emotion Analysis API",
    description="Unified Audio + Video emotion detection and grading",
    version="1.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory configuration (Using root-level temp folders)
TEMP_VIDEO_DIR = "temp_videos"
TEMP_AUDIO_DIR = "temp_audio"
TEMP_TRANS_DIR = "temp_trans"
TEMP_RESULTS_DIR = "temp_results"
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
os.makedirs(TEMP_TRANS_DIR, exist_ok=True)
os.makedirs(TEMP_RESULTS_DIR, exist_ok=True)

# Initialize global services once
audio_detector = AudioEmotionDetector()
text_processor = TextProcessor()
video_detector = OptimizedEmotionDetector()

def get_ffmpeg_path():
    """Robustly find FFmpeg path across Windows and Linux (Render)"""
    # 1. Check if it's already in the PATH (standard for Linux/Render)
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return 'ffmpeg'
    except:
        pass

    # 2. Windows-specific WinGet check (only if on Windows)
    if os.name == 'nt':
        username = os.getenv('USERNAME') or os.getenv('USER')
        winget_path = f"C:\\Users\\{username}\\AppData\\Local\\Microsoft\\WinGet\\Packages"
        if os.path.exists(winget_path):
            for root, dirs, files in os.walk(winget_path):
                if 'ffmpeg.exe' in files:
                    full_path = os.path.join(root, 'ffmpeg.exe')
                    logger.info(f"Auto-detected Windows FFmpeg at: {full_path}")
                    return full_path

    # 3. Last fallback
    return 'ffmpeg'

# Cache the detected path
FFMPEG_EXE = get_ffmpeg_path()

def extract_audio_from_video(video_path, audio_path):
    """Step 3: Extract Audio from Video using FFmpeg"""
    logger.info(f"Extracting audio using {FFMPEG_EXE}...")
    try:
        # -y to overwrite, -vn for no video, -acodec pcm_s16le for standard WAV, -ar 22050 for model compatibility
        command = [
            FFMPEG_EXE, '-y', '-i', video_path, 
            '-vn', '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1', 
            audio_path
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to run FFmpeg ({FFMPEG_EXE}): {str(e)}")
        return False

def safe_cleanup(file_path):
    """Attempt to delete a file with retries to handle Windows file locking"""
    if not os.path.exists(file_path):
        return
        
    for i in range(3):
        try:
            os.remove(file_path)
            return
        except PermissionError:
            time.sleep(0.5) # Wait for handles to release
        except Exception as e:
            logger.warning(f"Failed to delete {file_path}: {e}")
            break

@app.post("/api/multimodal/analyze")
async def analyze_multimodal(video: UploadFile = File(...)):
    """Unified Multimodal API Analysis Pipeline"""
    job_id = str(uuid.uuid4())
    logger.info(f"Starting multimodal job {job_id} for {video.filename}")
    
    # Step 2: Save Uploaded Video
    video_ext = video.filename.split('.')[-1]
    video_path = os.path.join(TEMP_VIDEO_DIR, f"{job_id}.{video_ext}")
    audio_path = os.path.join(TEMP_AUDIO_DIR, f"{job_id}.wav")
    
    try:
        # Save to disk
        content = await video.read()
        with open(video_path, "wb") as f:
            f.write(content)
        
        # IMPORTANT: Close the UploadFile handle immediately to avoid WinError 32
        await video.close()
            
        # Step 3: Extract Audio
        if not extract_audio_from_video(video_path, audio_path):
            raise HTTPException(status_code=500, detail="Failed to extract audio from video. Ensure FFmpeg is installed.")
            
        # Step 4 & 5: Parallel Model Inference
        logger.info(f"Running parallel inference for job {job_id}...")
        
        loop = asyncio.get_event_loop()
        
        # Audio, Video & Text tasks run concurrently
        video_task = loop.run_in_executor(None, video_detector.process_video, video_path)
        audio_task = loop.run_in_executor(None, audio_detector.process_audio, audio_path)
        text_task = loop.run_in_executor(None, text_processor.process_audio, audio_path)
        
        video_results, audio_results, text_results = await asyncio.gather(video_task, audio_task, text_task)
        
        # Step 6: Multi-Level Advanced Fusion (Intelligence 2.0)
        audio_probs = audio_results["probabilities"]
        video_probs_list = [f['probabilities'] for f in video_results.get('emotion_data', []) if 'probabilities' in f]
        video_labels_list = [f['emotion'] for f in video_results.get('emotion_data', [])]
        
        if not video_probs_list: 
            # Case with no faces, fallback to distribution mean mapping
            dist = video_results['emotion_distribution']
            # Map dictionary to list based on Standard Emotions order
            video_probs_list = [[(dist.get(emo, 0) / 100.0) for emo in FusionLogicService.STANDARD_EMOTIONS]]
        
        # Calculate specialized rubrics with Advanced Intelligence
        audio_labels = list(audio_results['emotion_distribution'].keys())
        audio_metrics = FusionLogicService.calculate_audio_rubrics(audio_probs, labels=audio_labels)
        
        # Video rubrics now return avg_probs for alignment check
        video_metrics, video_avg_probs = FusionLogicService.calculate_video_rubrics(video_probs_list, video_labels_list)
        
        # New Step: Analyze Text Quality (using Groq + Notebook Prompts)
        logger.info("Analyzing text quality (Textual Depth)...")
        text_analysis_results = text_processor.analyze_text_metrics(text_results.get("qa_pairs", []))
        duration = video_results.get("processing_stats", {}).get("duration", 0)
        text_metrics, text_overall_score, text_stats = FusionLogicService.calculate_text_rubrics(text_analysis_results, duration)

        # Performance Advanced Weighted Fusion + Alignment (Now with Text!)
        fusion_results = FusionLogicService.fuse_metrics(
            audio_metrics, video_metrics, text_metrics, audio_probs, video_avg_probs, text_overall_score
        )
        
        # Step 7: AI Feedback Generation (Groq/Llama 3 with Alignment Awareness)
        logger.info("Generating AI multimodal assessment...")
        
        # Step 9.5: Cross-Modal Alignment extraction (needed for feedback)
        audio_dom = audio_results["dominant_emotion"]
        video_dom = video_results["dominant_emotion"]
        
        def generate_mismatch_desc(a_dom, v_dom, align_score):
            if align_score >= 75:
                return "" # No major mismatch
            else:
                a_desc = "flat" if a_dom.lower() in ["neutral", "sad"] else a_dom.lower()
                v_desc = "positive" if v_dom.lower() in ["happy", "surprise"] else v_dom.lower()
                return f"Detected mismatch: {v_desc.capitalize()} facial expression but {a_desc} vocal tone."
        
        mismatch_example = generate_mismatch_desc(audio_dom, video_dom, fusion_results["alignment_score"])
        
        ai_feedback = await FusionLogicService.generate_multimodal_feedback(
            audio_metrics, 
            video_metrics, 
            text_metrics,
            fusion_results["final_score"], 
            fusion_results["alignment_score"],
            mismatch_flag=mismatch_example
        )

        # Step 8: Evidence-Based Snippets (XAI)
        logger.info("Extracting XAI behavioral evidence...")
        evidence_log = FusionLogicService.extract_xai_evidence(
            audio_results.get("chunk_data", []),
            video_results.get("emotion_data", []),
            audio_metrics,
            video_metrics,
            fusion_results["alignment_score"],
            text_analysis=text_analysis_results
        )
        
        # Step 9: Behavioral Timeline Data
        logger.info("Generating behavioral timeline series...")
        timeline_data = FusionLogicService.generate_timeline_data(
            audio_results.get("chunk_data", []),
            video_results.get("emotion_data", [])
        )
        
        cross_modal_alignment = {
            "score": fusion_results["alignment_score"],
            "audio_emotion": audio_dom.capitalize(),
            "facial_emotion": video_dom.capitalize(),
            "consistency_score": fusion_results["alignment_score"],
            "mismatch_example": mismatch_example or f"Consistent alignment. Vocal tone ({audio_dom}) and facial expression ({video_dom}) match well."
        }
        
        # Step 10: Final Response Assembly
        # Include the new behavioral metrics for Eye Contact and Gestures
        visual_behavioral = video_results.get("behavioral_metrics", {})
        
        # Calculate Audio highlights (Avg dB and Pitch Variance)
        audio_chunks = audio_results.get("chunk_data", [])
        pitches = [c.get('pitch', 0) for c in audio_chunks if c.get('pitch', 0) > 0]
        avg_db = sum([c.get('db', -60) for c in audio_chunks]) / len(audio_chunks) if audio_chunks else -60
        avg_pitch = sum(pitches) / len(pitches) if pitches else 0
        
        audio_behavioral = {
            "avg_db": round(avg_db, 1),
            "avg_pitch": round(avg_pitch, 1),
            "pitch_variation": round(float(np.std(pitches)), 1) if pitches else 0,
            "wpm": text_stats.get("wpm", 0),
            "filler_pct": text_stats.get("filler_pct", 0)
        }

        response = {
            "job_id": job_id,
            "timestamp": datetime.now().isoformat(),
            "filename": video.filename,
            "multimodal_score": fusion_results["final_score"],
            "individual_analysis": {
                "audio": {
                    "dominant": audio_results["dominant_emotion"],
                    "metrics": audio_results["probabilities"],
                    "behavioral": audio_behavioral,
                    "rubrics": {
                        name: {"score": score, "justification": FusionLogicService.get_metric_justification(name, score, audio_probs=audio_probs)}
                        for name, score in audio_metrics.items()
                    }
                },
                "video": {
                    "dominant": video_results["dominant_emotion"],
                    "metrics": video_avg_probs.tolist(),
                    "behavioral": visual_behavioral,
                    "rubrics": {
                        name: {"score": score, "justification": FusionLogicService.get_metric_justification(name, score, video_avg_probs=video_avg_probs)}
                        for name, score in video_metrics.items()
                    }
                },
                "text": {
                    "overall_score": int(round(text_overall_score)),
                    "behavioral": text_stats,
                    "rubrics": text_metrics,
                    "detailed_analysis": text_analysis_results
                }
            },
            "rubric_scores": fusion_results["fused_rubrics"],
            "text_rubrics": fusion_results["text_rubrics"],
            "rubric_justifications": {
                name: FusionLogicService.get_metric_justification(name, score, audio_probs, video_avg_probs)
                for name, score in fusion_results["fused_rubrics"].items()
            },
            "cross_modal_alignment": cross_modal_alignment,
            "ai_feedback": ai_feedback,
            "evidence_log": evidence_log,
            "timeline_data": timeline_data,
            "transcription": text_results.get("dialogue", []),
            "qa_pairs": text_results.get("qa_pairs", [])
        }
        
        # New: Save the full response as JSON for persistence
        results_path = os.path.join(TEMP_RESULTS_DIR, f"{job_id}.json")
        try:
            with open(results_path, "w") as f:
                json.dump(response, f, indent=4)
            logger.info(f"Saved multimodal results to {results_path}")
        except Exception as save_error:
            logger.warning(f"Failed to save results JSON: {save_error}")
        
        # Cleanup temp files
        safe_cleanup(video_path)
        safe_cleanup(audio_path)
        
        return response
        
    except Exception as e:
        logger.error(f"Multimodal job {job_id} catastrophic failure: {str(e)}")
        # Cleanup on failure
        safe_cleanup(video_path)
        safe_cleanup(audio_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """System health check including FFmpeg and Intelligence Version"""
    try:
        ffmpeg_check = subprocess.run([FFMPEG_EXE, '-version'], capture_output=True).returncode == 0
    except:
        ffmpeg_check = False
        
    return {
        "status": "active", 
        "version": "1.1.0 (Advanced Intelligence)",
        "ffmpeg_available": ffmpeg_check,
        "ffmpeg_path": FFMPEG_EXE,
        "models_loaded": audio_detector.model is not None,
        "alignment_engine": "ready"
    }

if __name__ == "__main__":
    import uvicorn
    # Final production port 8010
    uvicorn.run(app, host="0.0.0.0", port=8010)
