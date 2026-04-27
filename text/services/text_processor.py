import whisperx
import pandas as pd
import os
import logging
import uuid
import re
import requests
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# Use paths relative to this file to work on both Windows and Linux (Render)
TEXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = os.path.join(TEXT_ROOT, "models", "huggingface")
os.environ["XDG_CACHE_HOME"] = os.path.join(TEXT_ROOT, "models")
os.environ["PYTHONHASHSEED"] = "0"

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self):
        self.device = "cpu"
        self.hf_token = os.getenv("HF_TOKEN")
        self.compute_type = "int8"  # Optimized for CPU
        self.batch_size = 4        # Conservative for CPU
        self.model_name = "small"
        self.device = "cpu"
        self.model = None
        self.diarize_model = None
        self.align_model_cache = {} # Cache models for different languages
        
        # Ensure model caches are created in the text folder
        self.model_cache_dir = os.path.join(TEXT_ROOT, "models")
        os.makedirs(os.path.join(self.model_cache_dir, "huggingface"), exist_ok=True)
        
        # Direct paths for storage
        # Move to backend root temp_trans for consistency
        backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.transcription_dir = os.path.join(backend_root, "temp_trans")
        os.makedirs(self.transcription_dir, exist_ok=True)

    def _load_models(self):
        """Lazy load models to save memory until actually needed"""
        if self.model is None:
            logger.info(f"Loading WhisperX model '{self.model_name}' on {self.device}...")
            try:
                self.model = whisperx.load_model(self.model_name, self.device, compute_type=self.compute_type)
            except Exception as e:
                logger.error(f"Failed to load WhisperX model: {e}")
                raise

        if self.diarize_model is None:
            logger.info(f"Loading Diarization pipeline on {self.device}...")
            try:
                if not self.hf_token:
                    logger.warning("HF_TOKEN not found in environment. Diarization may fail.")
                self.diarize_model = whisperx.diarize.DiarizationPipeline(token=self.hf_token, device=self.device, cache_dir=self.model_cache_dir)
            except Exception as e:
                logger.error(f"Failed to load Diarization pipeline: {e}")
                raise

    def process_audio(self, audio_path: str):
        """
        Transcribes audio, performs alignment and diarization, 
        then groups text by speaker and saves to CSV.
        """
        logger.info(f"Starting text processing for {audio_path}")
        try:
            self._load_models()
            
            # 1. Transcribe
            logger.info("Step 1/4: Transcribing audio text...")
            audio = whisperx.load_audio(audio_path)
            result = self.model.transcribe(audio, batch_size=self.batch_size, language="en")
            language = result.get("language", "en")
            logger.info(f"Transcription complete (Language: {language})")
            
            # 2. Align
            logger.info(f"Step 2/4: Aligning words for {language}...")
            if language not in self.align_model_cache:
                self.align_model_cache[language] = whisperx.load_align_model(language_code=language, device=self.device)
            
            model_a, metadata = self.align_model_cache[language]
            result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
            logger.info("Alignment complete.")
            
            # 3. Diarization
            logger.info("Step 3/4: Identifying speakers (Diarization)...")
            diarize_segments = self.diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            logger.info("Speaker Identification complete.")
            
            # 4. Final Formatting
            logger.info("Step 4/4: Formatting final Q&A dialogue...")
            
            # 4. Group continuous segments by speaker (Logical grouping)
            dialogue = []
            speaker_map = {} # Maps original labels (e.g. SPEAKER_00) to Speaker 1, Speaker 2
            
            current_speaker = None
            current_text = ""
            
            for segment in result["segments"]:
                raw_speaker = segment.get("speaker", "UNKNOWN")
                text = segment["text"].strip()
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                
                if raw_speaker not in speaker_map:
                    # Map chronologically to Speaker 1, Speaker 2, etc.
                    speaker_map[raw_speaker] = f"Speaker {len(speaker_map) + 1}"
                
                clean_speaker = speaker_map[raw_speaker]
                
                if clean_speaker == current_speaker:
                    current_text += " " + text
                    # Update end time
                    current_end = end
                else:
                    if current_speaker is not None:
                        dialogue.append({
                            "speaker": current_speaker, 
                            "text": current_text.strip(),
                            "start": current_start,
                            "end": current_end
                        })
                    current_speaker = clean_speaker
                    current_text = text
                    current_start = start
                    current_end = end
            
            # Append last segment
            if current_speaker is not None:
                dialogue.append({
                    "speaker": current_speaker, 
                    "text": current_text.strip(),
                    "start": current_start,
                    "end": current_end
                })

            # 5. Format as Question/Answer (Intelligent Grouping)
            # Identifying the Interviewer (Speaker who started the convo or has more/shorter segments)
            qa_pairs = []
            if dialogue:
                interviewer = dialogue[0]["speaker"]
                current_q = None
                current_a_segments = []
                
                for seg in dialogue:
                    if seg["speaker"] == interviewer:
                        # If we have a previous Q, save it before starting a new one
                        if current_q is not None:
                            qa_pairs.append({
                                "Question": current_q["text"],
                                "Answer": " ".join(current_a_segments) if current_a_segments else "No response detected.",
                                "start": current_q["start"],
                                "end": seg["start"] # End of the answer is the start of next question
                            })
                        current_q = seg
                        current_a_segments = []
                    else:
                        current_a_segments.append(seg["text"])
                
                # Append final pair
                if current_q is not None:
                    qa_pairs.append({
                        "Question": current_q["text"],
                        "Answer": " ".join(current_a_segments) if current_a_segments else "No response detected.",
                        "start": current_q["start"],
                        "end": dialogue[-1]["end"]
                    })
            
            if not qa_pairs and dialogue:
                # Fallback if only one speaker detected
                qa_pairs.append({
                    "Question": "Monologue/Intro",
                    "Answer": " ".join([d["text"] for d in dialogue]),
                    "start": dialogue[0]["start"],
                    "end": dialogue[-1]["end"]
                })

            # 6. Save to CSV (Following user notebook logic)
            job_id = os.path.basename(audio_path).split('.')[0]
            csv_filename = f"transcription_{job_id}.csv"
            csv_path = os.path.join(self.transcription_dir, csv_filename)
            
            # Create a simplified DataFrame for the CSV
            df_data = []
            for pair in qa_pairs:
                df_data.append([pair["Question"], pair["Answer"]])
            
            df = pd.DataFrame(df_data, columns=["Question (Speaker 1)", "Answer (Speaker 2)"])
            df.to_csv(csv_path, index=False)
            
            logger.info(f"Transcription saved to {csv_path}")

            return {
                "dialogue": dialogue,
                "qa_pairs": qa_pairs,
                "csv_path": csv_path,
                "speaker_map": speaker_map
            }

        except Exception as e:
            logger.error(f"Error during audio processing: {e}")
            return {"error": str(e), "segments": [], "text": ""}

    def analyze_text_metrics(self, qa_pairs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Uses Groq (Llama-3-8b) to analyze each QA pair in parallel for maximum speed.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or not qa_pairs:
            logger.warning("GROQ_API_KEY missing or no QA pairs to analyze.")
            return []

        client = Groq(api_key=api_key)
        results = [None] * len(qa_pairs) # Pre-allocate for ordering

        def process_chunk(index, pair):
            question = pair.get("Question", "")
            answer = pair.get("Answer", "No response detected.")
            
            # Simple stats even for AI analysis
            fillers = ["uh", "um", "like", "actually", "basically", "literally", "you know"]
            words = answer.split()
            word_count = len(words)
            filler_count = sum(1 for w in words if w.lower().strip(',.?!') in fillers)
            complex_ratio = sum(1 for w in words if len(w) > 6) / word_count if word_count > 0 else 0
            vocab_level = "Beginner" if complex_ratio < 0.15 else "Intermediate" if complex_ratio < 0.3 else "Advanced"

            prompt = f"""For the question, "{question}", give a Score out of 100 each for Relevance, Clarity, Correctness, Structured answers, Fluency, Professionalism, No fillers, Focused, Authentic, Overall. 

            For EVERY score, provide a concise, high-fidelity explanation (exactly 2-3 lines) focused on the specific linguistic evidence in the answer: "{answer}"
            Respond only in the format: 'Metric Name: Score/100 Explanation'"""
            
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert linguistic analyst. Provide precise, evidence-based scoring."
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.1,
                    max_tokens=800
                )
                
                content = chat_completion.choices[0].message.content
                analysis = self._parse_groq_response(content)
                return {
                    "question": question,
                    "answer": answer,
                    "metrics": analysis,
                    "stats": {
                        "word_count": word_count,
                        "filler_count": filler_count,
                        "vocab_level": vocab_level
                    },
                    "start": pair.get("start", 0),
                    "end": pair.get("end", 0)
                }
            except Exception as e:
                logger.error(f"Groq analysis failed for index {index}: {e}")
                return None

        # Using a smaller number of workers for free tier rate limits
        # With TPM of 6000, we prefer sequential or very low concurrency
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_index = {executor.submit(process_chunk, i, pair): i for i, pair in enumerate(qa_pairs)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    if result:
                        results[index] = result
                    else:
                        # Fallback for failed analysis: keep the Q&A text but mark metrics as empty
                        logger.warning(f"Analysis failed for question {index}, providing fallback data.")
                        pair = qa_pairs[index]
                        results[index] = {
                            "question": pair.get("Question", ""),
                            "answer": pair.get("Answer", "No response detected."),
                            "metrics": {},
                            "stats": {
                                "word_count": len(pair.get("Answer", "").split()),
                                "filler_count": 0,
                                "vocab_level": "N/A"
                            },
                            "start": pair.get("start", 0),
                            "end": pair.get("end", 0)
                        }
                except Exception as e:
                    logger.error(f"Thread failed for index {index}: {e}")

        # Filter out failed ones but keep it sorted if possible
        return [r for r in results if r is not None]

    def _parse_groq_response(self, content: str) -> Dict[str, Any]:
        """
        Parses the Groq response into a structured dictionary of scores and explanations.
        Matches the logic found in the user's notebooks.
        """
        metrics = {}
        valid_keys = {
            'Relevance', 'Clarity', 'Correctness', 'Structured answers', 
            'Fluency', 'Professionalism', 'No fillers', 
            'Focused', 'Authentic', 'Overall'
        }
        
        # Normalize and split into lines
        lines = content.split('\n')
        current_metric = None
        
        # Pattern to find: "Metric Name: 80/100" (handles bolding and numbering)
        header_pattern = re.compile(r'(?:\d+\.\s*)?\**([\w\s]+)\**\s*[:\(]?\s*(\d+)/100')

        for line in lines:
            line = line.strip()
            if not line: continue
            
            match = header_pattern.search(line)
            if match:
                raw_key = match.group(1).strip()
                score = int(match.group(2))
                
                found_key = None
                normalized_raw = raw_key.lower().replace(" ", "")
                for vk in valid_keys:
                    if normalized_raw == vk.lower().replace(" ", ""):
                        found_key = vk
                        break
                
                if found_key:
                    after_score = line[match.end():].strip()
                    explanation = after_score.lstrip(':').lstrip('-').strip()
                    
                    metrics[found_key] = {
                        "score": score,
                        "explanation": explanation
                    }
                    current_metric = found_key
            else:
                if current_metric:
                    clean_line = line
                    if clean_line.lower().startswith("explanation:"):
                        clean_line = clean_line[len("explanation:"):].strip()
                    
                    if metrics[current_metric]["explanation"]:
                        metrics[current_metric]["explanation"] += " " + clean_line.strip()
                    else:
                        metrics[current_metric]["explanation"] = clean_line.strip()

        for k in metrics:
            metrics[k]["explanation"] = re.sub(r'\s+', ' ', metrics[k]["explanation"]).strip()
        
        return metrics

if __name__ == "__main__":
    # This block allows you to run this file directly to pre-load and save the models to the E: drive
    import logging
    logging.basicConfig(level=logging.INFO)
    processor = TextProcessor()
    print("--- Starting Model Pre-load ---")
    processor._load_models()
    print(f"--- Models are now saved and ready in: {processor.model_cache_dir} ---")
