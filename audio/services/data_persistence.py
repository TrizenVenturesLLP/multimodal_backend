"""
Data Persistence Service
Handles storage and retrieval of video processing results and metadata
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class DataPersistenceService:
    """
    Service for persisting video processing data and results
    """
    
    def __init__(self, db_path: str = "video_analysis.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for data persistence"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create processing_jobs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER DEFAULT 0,
                    message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    processing_time_seconds REAL,
                    error_message TEXT,
                    client_ip TEXT,
                    user_agent TEXT
                )
            ''')
            
            # Create audio_results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audio_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    file_hash TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    audio_metadata TEXT NOT NULL,
                    emotion_analysis TEXT NOT NULL,
                    audio_score TEXT NOT NULL,
                    processing_mode TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES processing_jobs (job_id)
                )
            ''')
            
            # Create audio_analytics table for aggregated data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audio_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    total_audio_files INTEGER DEFAULT 0,
                    successful_processing INTEGER DEFAULT 0,
                    failed_processing INTEGER DEFAULT 0,
                    total_processing_time REAL DEFAULT 0,
                    average_processing_time REAL DEFAULT 0,
                    total_file_size BIGINT DEFAULT 0,
                    unique_audio_files INTEGER DEFAULT 0,
                    cache_hits INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_job_id ON processing_jobs(job_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_hash ON processing_jobs(file_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON processing_jobs(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON processing_jobs(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audio_results_job_id ON audio_results(job_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audio_results_file_hash ON audio_results(file_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_analytics_date ON audio_analytics(date)')
            
            conn.commit()
            conn.close()
            logger.info(f"Data persistence database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize data persistence database: {str(e)}")
            raise
    
    def save_processing_job(self, job_id: str, filename: str, file_hash: str, 
                          file_size: int, client_ip: str = None, user_agent: str = None) -> bool:
        """
        Save processing job metadata
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO processing_jobs 
                (job_id, filename, file_hash, file_size, status, client_ip, user_agent)
                VALUES (?, ?, ?, ?, 'queued', ?, ?)
            ''', (job_id, filename, file_hash, file_size, client_ip, user_agent))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved processing job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save processing job: {str(e)}")
            return False
    
    def update_job_status(self, job_id: str, status: str, progress: int = None, 
                         message: str = None, error_message: str = None) -> bool:
        """
        Update processing job status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build update query dynamically
            update_fields = ["status = ?"]
            params = [status]
            
            if progress is not None:
                update_fields.append("progress = ?")
                params.append(progress)
            
            if message is not None:
                update_fields.append("message = ?")
                params.append(message)
            
            if error_message is not None:
                update_fields.append("error_message = ?")
                params.append(error_message)
            
            # Set timestamps based on status
            if status == "processing":
                update_fields.append("started_at = CURRENT_TIMESTAMP")
            elif status in ["completed", "failed"]:
                update_fields.append("completed_at = CURRENT_TIMESTAMP")
            
            params.append(job_id)
            
            query = f"UPDATE processing_jobs SET {', '.join(update_fields)} WHERE job_id = ?"
            cursor.execute(query, params)
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update job status: {str(e)}")
            return False
    
    def save_audio_result(self, job_id: str, file_hash: str, filename: str,
                         audio_metadata: Dict, emotion_analysis: Dict, 
                         audio_score: Dict, processing_mode: str) -> bool:
        """
        Save audio processing results
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO audio_results 
                (job_id, file_hash, filename, audio_metadata, emotion_analysis, 
                 audio_score, processing_mode)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                job_id,
                file_hash,
                filename,
                json.dumps(audio_metadata),
                json.dumps(emotion_analysis),
                json.dumps(audio_score),
                processing_mode
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved audio result for job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save audio result: {str(e)}")
            return False
    
    def get_job_history(self, limit: int = 100, status: str = None) -> List[Dict]:
        """
        Get processing job history
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT job_id, filename, file_size, status, progress, message,
                       created_at, started_at, completed_at, processing_time_seconds,
                       error_message, client_ip
                FROM processing_jobs
            '''
            params = []
            
            if status:
                query += " WHERE status = ?"
                params.append(status)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            conn.close()
            
            # Convert to list of dictionaries
            jobs = []
            for row in results:
                jobs.append({
                    "job_id": row[0],
                    "filename": row[1],
                    "file_size": row[2],
                    "status": row[3],
                    "progress": row[4],
                    "message": row[5],
                    "created_at": row[6],
                    "started_at": row[7],
                    "completed_at": row[8],
                    "processing_time_seconds": row[9],
                    "error_message": row[10],
                    "client_ip": row[11]
                })
            
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to get job history: {str(e)}")
            return []
    
    def get_audio_result(self, job_id: str) -> Optional[Dict]:
        """
        Get audio processing result by job ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT file_hash, filename, audio_metadata, emotion_analysis,
                       audio_score, processing_mode, created_at
                FROM audio_results
                WHERE job_id = ?
            ''', (job_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    "file_hash": result[0],
                    "filename": result[1],
                    "audio_metadata": json.loads(result[2]),
                    "emotion_analysis": json.loads(result[3]),
                    "audio_score": json.loads(result[4]),
                    "processing_mode": result[5],
                    "created_at": result[6]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get audio result: {str(e)}")
            return None
    
    def get_analytics(self, days: int = 30) -> Dict:
        """
        Get processing analytics for the last N days
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get daily analytics
            cursor.execute('''
                SELECT date, total_audio_files, successful_processing, failed_processing,
                       total_processing_time, average_processing_time, total_file_size,
                       unique_audio_files, cache_hits
                FROM audio_analytics
                WHERE date >= date('now', '-{} days')
                ORDER BY date DESC
            '''.format(days))
            
            daily_analytics = cursor.fetchall()
            
            # Get overall statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_jobs,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    AVG(CASE WHEN completed_at IS NOT NULL AND started_at IS NOT NULL 
                        THEN (julianday(completed_at) - julianday(started_at)) * 86400 
                        ELSE NULL END) as avg_processing_time,
                    SUM(file_size) as total_file_size
                FROM processing_jobs
                WHERE created_at >= date('now', '-{} days')
            '''.format(days))
            
            overall_stats = cursor.fetchone()
            
            conn.close()
            
            return {
                "period_days": days,
                "overall_stats": {
                    "total_jobs": overall_stats[0] or 0,
                    "successful_processing": overall_stats[1] or 0,
                    "failed_processing": overall_stats[2] or 0,
                    "success_rate": round((overall_stats[1] or 0) / max(overall_stats[0] or 1, 1) * 100, 2),
                    "average_processing_time_seconds": round(overall_stats[3] or 0, 2),
                    "total_file_size_mb": round((overall_stats[4] or 0) / (1024 * 1024), 2)
                },
                "daily_analytics": [
                    {
                        "date": row[0],
                        "total_audio_files": row[1],
                        "successful_processing": row[2],
                        "failed_processing": row[3],
                        "total_processing_time": row[4],
                        "average_processing_time": row[5],
                        "total_file_size_mb": round(row[6] / (1024 * 1024), 2),
                        "unique_audio_files": row[7],
                        "cache_hits": row[8]
                    }
                    for row in daily_analytics
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get analytics: {str(e)}")
            return {}
    
    def update_daily_analytics(self, date: str = None) -> bool:
        """
        Update daily analytics (should be called periodically)
        """
        try:
            if not date:
                date = datetime.now().strftime('%Y-%m-%d')
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate analytics for the date
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_audio_files,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN completed_at IS NOT NULL AND started_at IS NOT NULL 
                        THEN (julianday(completed_at) - julianday(started_at)) * 86400 
                        ELSE 0 END) as total_processing_time,
                    AVG(CASE WHEN completed_at IS NOT NULL AND started_at IS NOT NULL 
                        THEN (julianday(completed_at) - julianday(started_at)) * 86400 
                        ELSE NULL END) as avg_processing_time,
                    SUM(file_size) as total_file_size,
                    COUNT(DISTINCT file_hash) as unique_audio_files
                FROM processing_jobs
                WHERE date(created_at) = ?
            ''', (date,))
            
            stats = cursor.fetchone()
            
            if stats and stats[0] > 0:  # Only update if there are jobs for this date
                cursor.execute('''
                    INSERT OR REPLACE INTO audio_analytics 
                    (date, total_audio_files, successful_processing, failed_processing,
                     total_processing_time, average_processing_time, total_file_size, unique_audio_files)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (date, *stats))
                
                conn.commit()
                logger.info(f"Updated daily analytics for {date}")
            
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update daily analytics: {str(e)}")
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """
        Clean up old processing data (keep only last N days)
        Returns number of records deleted
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete old processing jobs
            cursor.execute('''
                DELETE FROM processing_jobs 
                WHERE created_at < date('now', '-{} days')
            '''.format(days_to_keep))
            
            deleted_jobs = cursor.rowcount
            
            # Delete old audio results (orphaned)
            cursor.execute('''
                DELETE FROM audio_results 
                WHERE job_id NOT IN (SELECT job_id FROM processing_jobs)
            ''')
            
            deleted_results = cursor.rowcount
            
            # Delete old analytics
            cursor.execute('''
                DELETE FROM audio_analytics 
                WHERE date < date('now', '-{} days')
            '''.format(days_to_keep))
            
            deleted_analytics = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            total_deleted = deleted_jobs + deleted_results + deleted_analytics
            if total_deleted > 0:
                logger.info(f"Cleaned up {total_deleted} old records (jobs: {deleted_jobs}, results: {deleted_results}, analytics: {deleted_analytics})")
            
            return total_deleted
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {str(e)}")
            return 0
