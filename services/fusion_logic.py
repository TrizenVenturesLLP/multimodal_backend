import numpy as np
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class FusionLogicService:
    # Standard 7-emotion scale used for coaching formulas
    # 0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Neutral, 5: Sad, 6: Surprise
    STANDARD_EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    @staticmethod
    def scale(val, min_v, max_v):
        """
        Advanced Scaling logic (5-95%)
        Avoids extreme binary values to reflect realistic human communication psychology.
        """
        if max_v == min_v:
            return 50.0 # Neutral baseline
        
        # Calculate raw 0-100 score
        normalized_score = ((val - min_v) / (max_v - min_v)) * 100
        
        # Clamp between 5 and 95 and return as integer
        return int(round(max(5, min(95, normalized_score))))

    @classmethod
    def _normalize_probs(cls, probs, current_labels=None):
        """
        Intelligently maps any probability vector back to the standard 7-emotion scale.
        """
        if len(probs) == 7 and not current_labels:
            return probs
            
        final_probs = [0.0] * 7
        if current_labels:
            for i, score in enumerate(probs):
                if i < len(current_labels):
                    label = current_labels[i].capitalize()
                    if label in cls.STANDARD_EMOTIONS:
                        idx = cls.STANDARD_EMOTIONS.index(label)
                        final_probs[idx] = float(score)
            return final_probs
            
        for i in range(min(len(probs), 7)):
            final_probs[i] = float(probs[i])
        return final_probs

    @classmethod
    def calculate_alignment_score(cls, audio_probs, video_avg_probs):
        """
        STEP 2: Emotion Alignment (The 'Truthfulness' Metric)
        Compares the similarity between vocal and facial emotional distributions.
        High alignment = Voice and Face are telling the same story.
        """
        a = np.array(audio_probs)
        v = np.array(video_avg_probs)
        
        # Avoid division by zero
        if np.linalg.norm(a) == 0 or np.linalg.norm(v) == 0:
            return 50.0
            
        # Cosine Similarity (0 to 1)
        similarity = np.dot(a, v) / (np.linalg.norm(a) * np.linalg.norm(v))
        
        # Map to 5-95 scale
        return cls.scale(similarity, 0, 1)

    @classmethod
    def calculate_audio_rubrics(cls, probs, labels=None):
        """
        Calculates audio rubrics using adaptive emotion mapping and realistic scaling.
        """
        probs = cls._normalize_probs(probs, labels)
        angry, disgust, fear, happy, neutral, sad, surprise = probs
        
        metrics = {
            "Sentiment Score": cls.scale((happy + surprise) - (angry + disgust + fear + sad), -1, 1),
            "Professionalism Score": cls.scale(neutral + (1 - (angry + disgust + fear)), 0, 2),
            "Emotional Stability": cls.scale(neutral - (angry + fear + 0.5 * surprise), -1, 1),
            "Engagement Score": cls.scale(happy + surprise + 0.5 * neutral, 0, 1.5), # Adjusted formula
            "Calmness Score": cls.scale(neutral - (angry + fear + disgust + 0.3 * surprise), -1, 1),
            "Emotional Intensity": cls.scale(max(probs) if probs else 0, 0, 1),
            "Vocal Confidence": cls.scale(happy + neutral - (fear + sad + 0.5 * disgust), -1, 1),
            "Vocal Engagement": cls.scale(happy + surprise + 0.3 * neutral - 0.2 * sad, -0.2, 1.3),
            "Stress/Tension Score": cls.scale(fear + angry + sad + 0.5 * disgust, 0, 1.5) # Flipped for better dashboard viewing
        }
        return metrics

    @classmethod
    def calculate_video_rubrics(cls, probs_list, labels_list):
        """
        Calculates video rubrics using the 7 new metrics:
        Confidence, Enthusiasm, Stress, Stability, Engagement, Frustration, Anxiety.
        Includes spike detection for dynamic behavioral analysis.
        """
        if not probs_list:
            return {}
            
        norm_probs = [cls._normalize_probs(p) for p in probs_list]
        avg_probs = np.mean(norm_probs, axis=0)
        anger, disgust, fear, happy, neutral, sad, surprise = avg_probs
        
        # Calculate Surprise Spike (Max - Avg) for Stress/Anxiety formulas
        surprise_values = [p[6] for p in norm_probs]
        surprise_spike = max(0, max(surprise_values) - surprise) if surprise_values else 0
        
        # Implementation of User's Image Formulas
        metrics = {
            "Confidence": cls.scale((neutral * 0.6 + happy * 0.4) - (fear + sad + anger + disgust) * 0.5, -0.5, 1.2),
            "Enthusiasm": cls.scale((happy + 0.6 * surprise) - (anger + disgust + fear + sad) * 0.2, 0, 1.5),
            "Stress": cls.scale(fear * 1.0 + anger * 0.6 + disgust * 0.6 + sad * 0.5 + (surprise_spike if surprise_spike > 0.15 else 0) * 0.5 - neutral * 0.3, -0.2, 1.5),
            "Emotional Stability": cls.scale(neutral - (anger + disgust + fear + sad) * 0.4 - (surprise_spike if surprise_spike > 0.1 else 0) * 0.2, -0.2, 1.0),
            "Engagement": cls.scale(happy + surprise + 0.2 * (anger + fear), 0, 1.5),
            "Frustration": cls.scale(1.5 * (anger + disgust) + 0.5 * sad, 0, 1.5),
            "Anxiety": cls.scale(1.2 * fear + (1.0 if surprise_spike > 0.1 else 0) + 0.3 * sad, 0, 1.5)
        }
        
        return metrics, avg_probs

    @classmethod
    def calculate_text_rubrics(cls, text_analysis_results: List[Dict[str, Any]], duration_sec: float = 0):
        """
        Aggregates text metrics from multiple QA pairs into a unified score map.
        """
        if not text_analysis_results:
            return {}, 50.0, {}

        all_metrics = [r["metrics"] for r in text_analysis_results if "metrics" in r]
        all_stats = [r["stats"] for r in text_analysis_results if "stats" in r]
        
        if not all_metrics:
            return {}, 50.0, {}

        # Original 13-key rubrics
        tracked_keys = [
            'Relevance', 'Clarity', 'Correctness', 'Structured answers', 
            'Fluency', 'Professionalism', 'No fillers', 
            'Focused', 'Authentic', 'Overall'
        ]
        
        aggregated = {k: [] for k in tracked_keys}
        
        for m_set in all_metrics:
            for k in tracked_keys:
                if k in m_set:
                    aggregated[k].append(m_set[k]["score"])
        
        final_text_rubrics = {}
        for k, scores in aggregated.items():
            final_text_rubrics[f"Textual {k}"] = int(round(sum(scores) / len(scores))) if scores else 50
                
        # Calculate behavioral text statistics
        total_words = sum(s["word_count"] for s in all_stats)
        total_fillers = sum(s["filler_count"] for s in all_stats)
        duration_min = duration_sec / 60 if duration_sec > 0 else 1
        
        communication_summary = {
            "wpm": round(total_words / duration_min, 1),
            "filler_pct": round((total_fillers / total_words * 100), 1) if total_words > 0 else 0,
            "vocab_level": max(set([s["vocab_level"] for s in all_stats]), key=[s["vocab_level"] for s in all_stats].count, default="Intermediate") if all_stats else "Intermediate",
            "avg_response_length": round(total_words / len(text_analysis_results), 1) if text_analysis_results else 0
        }

        text_overall_avg = final_text_rubrics.get("Textual Overall", 50)
        
        return final_text_rubrics, text_overall_avg, communication_summary

    @classmethod
    def get_metric_justification(cls, name, score, audio_probs=None, video_avg_probs=None, text_results=None):
        """
        XAI 3.0: Clinical Behavioral Justification Engine.
        Returns diagnostic summaries for the 7 new rubrics.
        """
        def get_tone(s, high_desc, low_desc, mid_desc="Moderate"):
            if s > 75: return high_desc
            if s < 45: return low_desc
            return mid_desc

        if "Confidence" in name:
            if score > 70: return "Strong facial composure and positive engagement suggest high self-assurance."
            if score < 40: return "Detected signs of uncertainty or lower neutral presence, indicating low visual confidence."
            return "Professional facial composure with stable presence."

        if "Enthusiasm" in name:
            if score > 70: return "Frequent positive expressions and high arousal signals show great energy and passion."
            if score < 40: return "Limited variation in facial energy suggests a need for more dynamic expressiveness."
            return "Moderate level of energy and professional warmth detected."

        if "Stress" in name:
            if score > 60: return "Elevated indicators of facial tension and strain detected throughout the session."
            if score < 30: return "Facial muscles remained relaxed, showing a calm and controlled demeanor."
            return "Occasional stress fluctuations detected in facial presence."

        if "Stability" in name:
            if score > 75: return "Maintained a consistent and professional baseline demeanor with minimal volatility."
            if score < 45: return "Detected fluctuations in emotional state, suggesting room for more consistent delivery."
            return "Steady and professional facial consistency maintained."

        if "Engagement" in name:
            if score > 70: return "Active facial responsiveness and high presence indicate strong involvement."
            if score < 40: return "Lower facial activity levels suggest a need for more active visual participation."
            return "Consistent and attentive visual presence maintained."

        if "Frustration" in name:
            if score > 50: return "Occasional micro-expressions of annoyance or rejection were captured."
            return "Maintained a professional front with no significant signs of frustration."

        if "Anxiety" in name:
            if score > 50: return "Visual markers of worry or apprehension indicate underlying nervousness."
            return "Appeared visually comfortable with no significant signs of nervous apprehension."

        if "Professionalism" in name:
            return "Consistent professional tone and delivery maintained throughout."

        if "Text" in name or any(x in name for x in ["Relevance", "Fluency", "Clarity"]):
            if score > 80: return "High clarity and structural integrity in the spoken content."
            if score < 50: return "Sentence structure could be tightened for better message impact."
            return "Professional and coherent textual output."

        return "Results derived from time-synchronized behavioral probability distributions."

    @classmethod
    def fuse_metrics(cls, audio_metrics, video_metrics, text_metrics, audio_probs, video_avg_probs, text_overall_score):
        """
        STEP 4: Intelligence Fusion 3.0 (Multimodal)
        Combines modalities using Weighted Average + Alignment + Coherence.
        Now includes Textual Intelligence.
        """
        # Step A: Calculate Alignment (Cross-Modality Match)
        alignment_score = cls.calculate_alignment_score(
            cls._normalize_probs(audio_probs), 
            cls._normalize_probs(video_avg_probs)
        )
        
        # Step B: Correlation Fusion
        correlation_map = {
            "Sentiment Score": "Enthusiasm",
            "Emotional Stability": "Stability",
            "Engagement Score": "Engagement",
            "Calmness Score": "Confidence",
            "Vocal Confidence": "Confidence",
            "Stress/Tension Score": "Stress"
        }
        
        fused_results = {}
        for a_key, v_key in correlation_map.items():
            if a_key in audio_metrics and v_key in video_metrics:
                # 60/40 Split between Audio and Video
                fused_results[f"Multimodal {a_key.split(' ')[0]}"] = (audio_metrics[a_key] * 0.6) + (video_metrics[v_key] * 0.4)
        
        # Step C: Intelligent Additions
        fused_results["Multimodal Alignment"] = alignment_score
        fused_results["Temporal Consistency"] = video_metrics.get("Temporal Consistency", 0)
        
        # Step D: Coherent Engagement Calculation
        # Combines vocal energy, facial movement, and alignment truthfulness
        coherent_engagement = (
            audio_metrics.get("Engagement Score", 50) * 0.4 +
            video_metrics.get("Engagement Level", 50) * 0.4 +
            alignment_score * 0.2
        )
        fused_results["Coherent Engagement"] = coherent_engagement
        
        # Step E: Generate Justifications
        justifications = {}
        for name, score in fused_results.items():
            justifications[name] = cls.get_metric_justification(name, score, audio_probs, video_avg_probs)

        # Final Score: Custom Weighted Fusion
        # 60% Text + 20% Audio + 20% Video
        audio_avg = sum(audio_metrics.values()) / len(audio_metrics) if audio_metrics else 50
        video_avg = sum(video_metrics.values()) / len(video_metrics) if video_metrics else 50
        
        raw_final = (text_overall_score * 0.6) + (audio_avg * 0.2) + (video_avg * 0.2)
        
        return {
            "fused_rubrics": fused_results,
            "text_rubrics": text_metrics,
            "fused_justifications": justifications,
            "final_score": int(round(raw_final)),
            "alignment_score": int(round(alignment_score)),
            "text_overall_score": int(round(text_overall_score))
        }

    @classmethod
    def extract_xai_evidence(cls, audio_chunks, video_frames, audio_metrics, video_metrics, alignment_score, text_analysis=None):
        """
        Step 6: XAI Evidence Extraction 2.0 (PhD Upgrade)
        Identifies behavioral segments [START - END] and provides granular proofs.
        """
        evidence_log = []
        if not audio_chunks or not video_frames:
            # Still process text if available
            if not text_analysis: return []

        # --- CATEGORY 0: TEXTUAL (NEW) ---
        if text_analysis:
            try:
                # Find most high-impact textual moment
                best_answer = max(text_analysis, key=lambda x: x['metrics'].get('Clarity', {}).get('score', 0), default=None)
                if best_answer and best_answer['metrics'].get('Clarity', {}).get('score', 0) > 80:
                    s_text = best_answer.get('start', 0)
                    e_text = best_answer.get('end', 0)
                    evidence_log.append({
                        "timestamp": f"{fmt_time(s_text)} - {fmt_time(e_text)}",
                        "tag": "Text Clarity",
                        "benchmark_link": "Text Performance",
                        "proof": "Structured and logically consistent response.",
                        "evidence": f"Insight: Deep logical coherence in responses.",
                        "justification": "Your verbal structure during key responses showed high quality, significantly boosting your Master Score.",
                        "type": "success",
                        "impact": 15
                    })
            except Exception: pass

        # Helper: Format timestamp
        def fmt_time(sec):
            m, s = divmod(round(sec), 60)
            return f"{m:02d}:{s:02d}"

        # 1. TEMPORAL MAPPING: Map chunks to timestamps for windowing
        audio_map = {round(c['timestamp']): c for c in audio_chunks}
        
        # Helper: Find the bounds of a behavior around a peak
        def get_bounds(target_ts, signal_list, threshold_fn, max_window=10):
            start = target_ts
            end = target_ts
            # Scan backwards
            for i in range(1, max_window):
                prev_ts = target_ts - i
                if prev_ts < 0: break
                if not threshold_fn(prev_ts): break
                start = prev_ts
            # Scan forwards
            for i in range(1, max_window):
                next_ts = target_ts + i
                if not audio_map.keys() or next_ts > max(audio_map.keys()): break
                if not threshold_fn(next_ts): break
                end = next_ts
            
            # PhD Upgrade: Ensure a minimum "Reviewable Window" (at least 2s)
            if start == end:
                if start > 0: start -= 1
                if audio_map.keys() and end < max(audio_map.keys()): end += 1
                
            return start, end

        # --- CATEGORY 1: ENGAGEMENT (MOST IMPORTANT) ---
        engagement_signals = []
        for v in video_frames:
            ts = round(v['timestamp'])
            a = audio_map.get(ts)
            if a:
                # Signal: Neutral/Sad + Low Energy
                val = (v['probabilities'][4] + v['probabilities'][5] + a['probabilities'][4] + a['probabilities'][5]) / 4
                if a.get('energy', 1.0) < 0.02: val += 0.2 # Penalty for low energy
                engagement_signals.append((ts, val))
        
        if engagement_signals:
            ts, peak = max(engagement_signals, key=lambda x: x[1], default=(0, 0))
            if peak > 0.45: # Lowered from 0.55
                s, e = get_bounds(ts, engagement_signals, lambda t: any(x[1] > 0.35 for x in engagement_signals if x[0] == t))
                evidence_log.append({
                    "timestamp": f"{fmt_time(s)} - {fmt_time(e)}",
                    "tag": "Low Engagement",
                    "benchmark_link": "Multimodal Engagement",
                    "proof": "Monotone voice + low facial activity",
                    "evidence": "Synchronized drop in vocal and facial involvement detected.",
                    "justification": "This segment lowered your Engagement Score due to perceived lack of enthusiasm.",
                    "type": "warning",
                    "impact": -round(peak * 20)
                })

        # --- CATEGORY 2: STRESS / TENSION ---
        stress_signals = [(round(c['timestamp']), c['probabilities'][0] + c['probabilities'][2]) for c in audio_chunks]
        if stress_signals:
            ts, peak = max(stress_signals, key=lambda x: x[1], default=(0, 0))
            if peak > 0.25: # Lowered from 0.35
                s, e = get_bounds(ts, stress_signals, lambda t: any(x[1] > 0.2 for x in stress_signals if x[0] == t))
                evidence_log.append({
                    "timestamp": f"{fmt_time(s)} - {fmt_time(e)}",
                    "tag": "Stress Spike",
                    "benchmark_link": "Stress & Tension",
                    "proof": "High pitch variation + energy spikes",
                    "evidence": "Acoustic signatures matching physiological stress and vocal tension.",
                    "justification": "Elevated vocal signals match tension patterns, impacting overall calmness.",
                    "type": "warning",
                    "impact": -round(peak * 15)
                })

        # --- CATEGORY 3: ALIGNMENT (CORE FEATURE) ---
        if alignment_score < 80:
            mismatches = []
            for v in video_frames:
                ts = round(v['timestamp'])
                a = audio_map.get(ts)
                if a:
                    # Alignment check: Positive voice vs Negative/Neutral face
                    a_val = a['probabilities'][3] - (a['probabilities'][0] + a['probabilities'][2])
                    v_val = v['probabilities'][3] - (v['probabilities'][0] + v['probabilities'][5])
                    if (a_val > 0.2 and v_val < 0.1) or (a_val < -0.2 and v_val > -0.1):
                        mismatches.append((ts, abs(a_val - v_val)))
            if mismatches:
                ts, peak = max(mismatches, key=lambda x: x[1], default=(0, 0))
                s, e = get_bounds(ts, mismatches, lambda t: any(x[1] > 0.15 for x in mismatches if x[0] == t))
                evidence_log.append({
                    "timestamp": f"{fmt_time(s)} - {fmt_time(e)}",
                    "tag": "Emotion Mismatch",
                    "benchmark_link": "Multimodal Alignment",
                    "proof": "Tone/Face incongruence detected",
                    "evidence": "Modal Mismatch: Vocal indicators contradict non-verbal facial markers.",
                    "justification": "This discrepancy erodes perceived authenticity and trust with the audience.",
                    "type": "warning",
                    "impact": -round((peak / 2.0) * 30)
                })

        # --- CATEGORY 4: VIDEO (CONFIDENCE & ENTHUSIASM) ---
        confidence_signals = [(round(v['timestamp']), v['probabilities'][4]*0.6 + v['probabilities'][3]*0.4) for v in video_frames]
        if confidence_signals:
            ts, peak = max(confidence_signals, key=lambda x: x[1], default=(0, 0))
            if peak > 0.7:
                s, e = get_bounds(ts, confidence_signals, lambda t: any(x[1] > 0.6 for x in confidence_signals if x[0] == t))
                evidence_log.append({
                    "timestamp": f"{fmt_time(s)} - {fmt_time(e)}",
                    "tag": "High Confidence",
                    "benchmark_link": "Confidence",
                    "proof": "Steady gaze + professional composure",
                    "evidence": "Facial patterns indicating a professionally authoritative and confident posture.",
                    "justification": "Maintained exceptional composure during this segment, boosting your visual scores.",
                    "type": "success",
                    "impact": round(peak * 15)
                })

        # --- CATEGORY 5: VIDEO (ANXIETY/STRESS) ---
        # Detect Fear/Surprise spikes
        anxiety_signals = [(round(v['timestamp']), v['probabilities'][2]*1.2 + v['probabilities'][6]*0.5) for v in video_frames]
        if anxiety_signals:
            ts, peak = max(anxiety_signals, key=lambda x: x[1], default=(0, 0))
            if peak > 0.4:
                s, e = get_bounds(ts, anxiety_signals, lambda t: any(x[1] > 0.3 for x in anxiety_signals if x[0] == t))
                evidence_log.append({
                    "timestamp": f"{fmt_time(s)} - {fmt_time(e)}",
                    "tag": "Anxiety Detected",
                    "benchmark_link": "Anxiety",
                    "proof": "Micro-expressions of worry/startle",
                    "evidence": "Transient markers of nervous apprehension detected in facial presence.",
                    "justification": "Brief fluctuations in composure suggest a need for more steady breathing and focus.",
                    "type": "warning",
                    "impact": -round(peak * 20)
                })

        # --- CATEGORY 6: FRUSTRATION ---
        frustration_signals = [(round(v['timestamp']), v['probabilities'][0]*1.5 + v['probabilities'][1]*1.5) for v in video_frames]
        if frustration_signals:
            ts, peak = max(frustration_signals, key=lambda x: x[1], default=(0, 0))
            if peak > 0.35:
                s, e = get_bounds(ts, frustration_signals, lambda t: any(x[1] > 0.25 for x in frustration_signals if x[0] == t))
                evidence_log.append({
                    "timestamp": f"{fmt_time(s)} - {fmt_time(e)}",
                    "tag": "Frustration Alert",
                    "benchmark_link": "Frustration",
                    "proof": "Brow tension + rejection markers",
                    "evidence": "Subtle facial signals matching rejection or annoyance captured.",
                    "justification": "Detected indicators of frustration; try to maintain a neutral baseline during complex queries.",
                    "type": "warning",
                    "impact": -round(peak * 25)
                })
        
        # --- CATEGORY 7: CONFIDENCE DROP & STABLE PERFORMANCE ---
        if video_metrics.get("Confidence", 50) < 40:
             # Find a low confidence segment
             low_conf_signals = [(round(v['timestamp']), v['probabilities'][4]*0.6 + v['probabilities'][3]*0.4) for v in video_frames]
             if low_conf_signals:
                 ts, peak = min(low_conf_signals, key=lambda x: x[1], default=(0, 0))
                 s, e = get_bounds(ts, low_conf_signals, lambda t: any(x[1] < 0.5 for x in low_conf_signals if x[0] == t))
                 evidence_log.append({
                    "timestamp": f"{fmt_time(s)} - {fmt_time(e)}",
                    "tag": "Confidence Drop",
                    "benchmark_link": "Confidence",
                    "proof": "Decreased positive facial presence",
                    "evidence": "Detected a drop in self-assured visual markers and facial openness.",
                    "justification": "A more steady and open facial presence can help project greater authority.",
                    "type": "warning",
                    "impact": -15
                 })
        
        if video_metrics.get("Stability", 50) > 75:
            evidence_log.append({
                "timestamp": "Overall Session",
                "tag": "Stable Performance",
                "benchmark_link": "Stability",
                "proof": "Consistent emotional baseline",
                "evidence": "Maintained a highly controlled and professional emotional state throughout.",
                "justification": "Exceptional emotional control helps in maintaining rapport and authority.",
                "type": "success",
                "impact": 10
            })

        # Final cleanup and limit
        evidence_log.sort(key=lambda x: abs(x['impact']), reverse=True)
        return evidence_log[:6]

    @classmethod
    def generate_timeline_data(cls, audio_chunks, video_frames):
        """
        Step 6.5: Temporal Data Generation (Behavioral Timeline)
        Samples data every 2 seconds to create a continuous behavioral map.
        Enhanced with robustness to prevent hangs.
        """
        if not audio_chunks:
            return []

        try:
            timeline = []
            # Safety for max duration
            last_ts = audio_chunks[-1].get('timestamp', 0)
            max_duration = int(min(3600, round(last_ts))) # Cap at 1 hour for safety
            
            # Audio & Video direct maps for O(1) lookup
            audio_map = {round(c['timestamp']): c for c in audio_chunks if 'timestamp' in c}
            video_map = {round(v['timestamp']): v for v in video_frames if 'timestamp' in v}

            # Sample every 2-5 seconds based on duration to manage density
            step = 2 if max_duration < 300 else 5
            
            for ts in range(0, max_duration + 1, step):
                # Find closest audio chunk
                a = audio_map.get(ts) or audio_map.get(ts-1) or audio_map.get(ts+1)
                # Find closest video frame
                v = video_map.get(ts) or video_map.get(ts-1) or video_map.get(ts+1)
                
                if a and v:
                    a_p = a.get('probabilities', [0]*7)
                    v_p = v.get('probabilities', [0]*7)
                    
                    if len(a_p) >= 5 and len(v_p) >= 5:
                        # Formula 1: Stress (Based on new Video Stress formula)
                        stress_val = (a_p[2] + a_p[0]*0.6 + v_p[2]*1.0 + v_p[0]*0.6) * 50
                        
                        # Formula 2: Confidence (Based on new Video Confidence formula)
                        confidence_val = ((v_p[4]*0.6 + v_p[3]*0.4) * 0.7 + (a_p[3] + a_p[4]*0.5) * 0.3) * 100
                        if a.get('energy', 0) > 0.05: confidence_val += 10 
                        
                        # Formula 3: Engagement
                        engagement_val = (v_p[3] + v_p[6] + 0.2*(v_p[0]+v_p[2])) * 60 + (a.get('energy', 0) * 400)

                        m, s = divmod(ts, 60)
                        timeline.append({
                            "time": f"{m:02d}:{s:02d}",
                            "stress": round(float(stress_val), 1),
                            "confidence": round(float(confidence_val), 1),
                            "engagement": round(float(engagement_val), 1),
                            "pitch": round(float(a.get('pitch', 0)), 1)
                        })

            return timeline
        except Exception as e:
            logger.error(f"Critical error in timeline generation logic: {e}")
            return []

    @staticmethod
    async def generate_multimodal_feedback(audio_metrics, video_metrics, text_metrics, final_score, alignment_score, mismatch_flag=""):
        """
        Step 7: AI Feedback using Groq (Llama 3.1).
        Now includes 'Alignment' awareness AND 'Textual' insights using the provided Groq API key.
        """
        import os
        from groq import Groq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "AI feedback unavailable (Groq API key missing)."
            
        try:
            client = Groq(api_key=api_key)
            
            # Use rounded integers in the prompt to match UI
            audio_str = "\n".join([f"- {k}: {int(round(v))}" for k, v in audio_metrics.items()])
            video_str = "\n".join([f"- {k}: {int(round(v))}" for k, v in video_metrics.items()])
            # Include ALL text metrics for better context
            text_str = "\n".join([f"- {k}: {int(round(v))}" for k, v in text_metrics.items()])
            
            prompt = f"""
            You are an expert behavioral intelligence analyst and executive communication coach specializing in multimodal (audio + video + text) human communication.

            Analyze the following performance across all THREE channels (vocal acoustics, facial presence, and text analysis content):

            OVERALL PERFORMANCE: {int(round(final_score))}/100
            EMOTION ALIGNMENT (Face vs Voice): {int(round(alignment_score))}%
            MISMATCH DETECTED: {"YES" if mismatch_flag else "NO"}

            AUDIO (VOCAL) BENCHMARKS:
            {audio_str}

            VIDEO (FACIAL) BENCHMARKS:
            {video_str}
            
            TEXT ANALYSIS QUALITY:
            {text_str}

            Your task:
            Provide a precise, integrated assessment in 5–7 sentences that reflects deep behavioral insight across all three channels.

            STRICT ASSESSMENT RULES:
            1. ROUNDED VALUES: Only mention the rounded integer scores provided above. Never use decimals in your report.
            2. ALL MODALITIES: You MUST mention at least one specific insight regarding the TEXT ANALYSIS (content) quality (e.g., relevance, clarity, or correctness) alongside the audio/video metrics.
            3. MISMATCH ALERT: If MISMATCH DETECTED = YES, you MUST explicitly discuss the inconsistency between non-verbal signals.
            4. ACTIONABLE: Provide one highly specific, practical recommendation for improvement.

            Focus on the intersection of content and delivery. Use clear, human-like, professional coaching language.
            Write as a unified narrative paragraph—not bullet points.
            """
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a world-class executive communication coach and behavioral psychologist."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.1-8b-instant",
                temperature=0.6,
                max_tokens=800
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Multimodal Groq AI Feedback Error: {str(e)}")
            return "Failed to generate AI feedback using Groq."
