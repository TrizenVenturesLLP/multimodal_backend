"""
Video Compression Service
Optional compression for very large videos (>500MB) to reduce processing load
"""

import os
import subprocess
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class VideoCompressionService:
    """Service for compressing large videos while maintaining quality for emotion analysis"""
    
    def __init__(self):
        self.compression_threshold_mb = 500  # Compress videos >500MB
        self.target_bitrate = "2M"  # Target bitrate for compression
        self.preset = "medium"  # FFmpeg preset for speed/quality balance
    
    def should_compress(self, file_path: str) -> bool:
        """Check if video should be compressed based on size"""
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            return file_size_mb > self.compression_threshold_mb
        except Exception as e:
            logger.error(f"Error checking file size: {e}")
            return False
    
    def compress_video(self, input_path: str, output_path: str) -> Tuple[bool, str]:
        """
        Compress video using FFmpeg while maintaining quality for emotion analysis
        
        Args:
            input_path: Path to input video
            output_path: Path for compressed video
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # FFmpeg command optimized for emotion analysis
            # - Preserves frame rate and resolution for face detection
            # - Uses H.264 codec with optimized settings
            # - Maintains audio track (if present)
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-c:v', 'libx264',  # H.264 codec
                '-b:v', self.target_bitrate,  # Target bitrate
                '-preset', self.preset,  # Encoding speed preset
                '-crf', '23',  # Constant Rate Factor (good quality)
                '-c:a', 'aac',  # Audio codec
                '-b:a', '128k',  # Audio bitrate
                '-movflags', '+faststart',  # Optimize for streaming
                '-y',  # Overwrite output file
                output_path
            ]
            
            logger.info(f"Compressing video: {input_path} -> {output_path}")
            
            # Run FFmpeg command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                # Check if compression was effective
                original_size = os.path.getsize(input_path)
                compressed_size = os.path.getsize(output_path)
                compression_ratio = compressed_size / original_size
                
                logger.info(f"Compression successful: {compression_ratio:.2%} of original size")
                
                if compression_ratio < 0.7:  # At least 30% reduction
                    return True, f"Compressed to {compression_ratio:.1%} of original size"
                else:
                    # Compression wasn't effective, use original
                    os.remove(output_path)
                    return False, "Compression not effective, using original video"
            else:
                logger.error(f"FFmpeg compression failed: {result.stderr}")
                return False, f"Compression failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            logger.error("Video compression timed out")
            return False, "Compression timed out"
        except Exception as e:
            logger.error(f"Error during video compression: {e}")
            return False, f"Compression error: {str(e)}"
    
    def get_compression_info(self, file_path: str) -> dict:
        """Get information about video compression potential"""
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            return {
                "file_size_mb": round(file_size_mb, 2),
                "should_compress": self.should_compress(file_path),
                "compression_threshold_mb": self.compression_threshold_mb,
                "estimated_compressed_size_mb": round(file_size_mb * 0.6, 2) if file_size_mb > self.compression_threshold_mb else file_size_mb
            }
        except Exception as e:
            logger.error(f"Error getting compression info: {e}")
            return {
                "file_size_mb": 0,
                "should_compress": False,
                "compression_threshold_mb": self.compression_threshold_mb,
                "estimated_compressed_size_mb": 0
            }
