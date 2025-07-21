"""
VideoEditor API - Production Grade Smart Memory Management & Portrait Format

Enterprise-quality video editor with intelligent resource management,
comprehensive error handling, and optimized portrait video generation.
"""
import logging
import os
import gc
import time
import psutil
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union
from contextlib import contextmanager

import numpy as np
from moviepy.editor import (
    VideoFileClip, AudioFileClip, TextClip,
    CompositeVideoClip, concatenate_videoclips,
    CompositeAudioClip, afx, VideoClip
)
from moviepy.video.fx.all import resize, crop

from config import OUTPUT_DIR, FPS
from models import TextOverlay
from utils import sanitize_filename

log = logging.getLogger(__name__)

# Constants for maintainability
class VideoFormat:
    """Standard video format configurations."""
    PORTRAIT = (1080, 1920)  # 9:16 aspect ratio
    LANDSCAPE = (1920, 1080)  # 16:9 aspect ratio
    SQUARE = (1080, 1080)    # 1:1 aspect ratio

class RenderQuality(Enum):
    """Render quality presets based on system capabilities."""
    HIGH = {"fps": 30, "threads": 4, "bitrate": "4000k", "preset": "medium"}
    MEDIUM = {"fps": 24, "threads": 2, "bitrate": "2500k", "preset": "fast"}
    LOW = {"fps": 24, "threads": 1, "bitrate": "1500k", "preset": "ultrafast"}

@dataclass
class SystemResourceMetrics:
    """System resource monitoring data."""
    total_ram_gb: float
    used_ram_gb: float
    available_ram_gb: float
    safe_limit_gb: float
    cpu_usage_percent: float
    
    @property
    def ram_usage_percent(self) -> float:
        return (self.used_ram_gb / self.total_ram_gb) * 100
    
    @property
    def is_memory_critical(self) -> bool:
        return self.used_ram_gb > (self.safe_limit_gb * 0.9)
    
    @property
    def recommended_max_clips(self) -> int:
        """Calculate optimal clip count based on available resources."""
        if self.available_ram_gb > 4.0:
            return 10
        elif self.available_ram_gb > 2.5:
            return 7
        elif self.available_ram_gb > 1.5:
            return 5
        elif self.available_ram_gb > 0.8:
            return 3
        else:
            return 2

class VideoProcessingError(Exception):
    """Custom exception for video processing errors."""
    pass

class SmartVideoEditor:
    """
    Production-grade video editor with intelligent resource management.
    
    Features:
    - Real-time memory monitoring and adaptive processing
    - Intelligent portrait format optimization with smart cropping
    - Dynamic quality adjustment based on system resources
    - Comprehensive error handling and recovery mechanisms
    - Performance metrics and detailed logging
    """
    
    def __init__(self, target_format: Tuple[int, int] = VideoFormat.PORTRAIT):
        """
        Initialize the smart video editor.
        
        Args:
            target_format: Target video dimensions (width, height)
        """
        self.target_format = target_format
        self.target_aspect_ratio = target_format[0] / target_format[1]
        
        # Initialize system monitoring
        self._initialize_system_metrics()
        
        # Performance tracking
        self.processing_stats = {
            "clips_processed": 0,
            "total_processing_time": 0.0,
            "memory_peak_usage": 0.0,
            "clips_failed": 0
        }
        
        log.info(f"SmartVideoEditor initialized")
        log.info(f"Target format: {target_format[0]}x{target_format[1]} (aspect: {self.target_aspect_ratio:.3f})")
        log.info(f"System: {self.system_metrics.total_ram_gb:.1f}GB RAM, Safe limit: {self.system_metrics.safe_limit_gb:.1f}GB")
    
    def _initialize_system_metrics(self) -> None:
        """Initialize system resource monitoring."""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        self.system_metrics = SystemResourceMetrics(
            total_ram_gb=memory.total / (1024**3),
            used_ram_gb=memory.used / (1024**3),
            available_ram_gb=(memory.available / (1024**3)),
            safe_limit_gb=(memory.total / (1024**3)) * 0.75,  # Use 75% max
            cpu_usage_percent=cpu_percent
        )
    
    def _update_system_metrics(self) -> SystemResourceMetrics:
        """Update current system resource metrics."""
        memory = psutil.virtual_memory()
        self.system_metrics.used_ram_gb = memory.used / (1024**3)
        self.system_metrics.available_ram_gb = memory.available / (1024**3)
        self.system_metrics.cpu_usage_percent = psutil.cpu_percent(interval=0.1)
        
        # Track peak memory usage
        if self.system_metrics.used_ram_gb > self.processing_stats["memory_peak_usage"]:
            self.processing_stats["memory_peak_usage"] = self.system_metrics.used_ram_gb
        
        return self.system_metrics
    
    @contextmanager
    def _memory_guard(self, operation_name: str):
        """Context manager for memory-safe operations."""
        metrics_before = self._update_system_metrics()
        
        if metrics_before.is_memory_critical:
            log.warning(f"Memory critical before {operation_name}. Forcing cleanup...")
            gc.collect()
            metrics_before = self._update_system_metrics()
        
        start_time = time.time()
        
        try:
            yield metrics_before
        finally:
            end_time = time.time()
            metrics_after = self._update_system_metrics()
            
            processing_time = end_time - start_time
            memory_delta = metrics_after.used_ram_gb - metrics_before.used_ram_gb
            
            log.debug(f"{operation_name} completed in {processing_time:.2f}s, "
                     f"RAM: {metrics_before.used_ram_gb:.1f}GB → {metrics_after.used_ram_gb:.1f}GB "
                     f"(Δ{memory_delta:+.1f}GB)")
    
    def _calculate_smart_crop_parameters(self, original_size: Tuple[int, int]) -> Dict[str, int]:
        """
        Calculate optimal crop parameters for aspect ratio conversion.
        
        Args:
            original_size: (width, height) of original video
            
        Returns:
            Dict with crop parameters: {'x1', 'y1', 'x2', 'y2'}
        """
        original_w, original_h = original_size
        target_w, target_h = self.target_format
        
        original_aspect = original_w / original_h
        
        if abs(original_aspect - self.target_aspect_ratio) < 0.05:
            # Aspect ratios are very close, no cropping needed
            return {}
        
        if original_aspect > self.target_aspect_ratio:
            # Video is too wide - crop sides (letterbox removal)
            new_width = int(original_h * self.target_aspect_ratio)
            x_center = original_w // 2
            x1 = max(0, x_center - new_width // 2)
            x2 = min(original_w, x1 + new_width)
            
            log.debug(f"Horizontal crop: {original_w}px → {new_width}px (x1={x1}, x2={x2})")
            return {'x1': x1, 'x2': x2}
        else:
            # Video is too tall - crop top/bottom (pillarbox removal)
            new_height = int(original_w / self.target_aspect_ratio)
            y_center = original_h // 2
            y1 = max(0, y_center - new_height // 2)
            y2 = min(original_h, y1 + new_height)
            
            log.debug(f"Vertical crop: {original_h}px → {new_height}px (y1={y1}, y2={y2})")
            return {'y1': y1, 'y2': y2}
    
    def _smart_resize_to_target_format(self, clip: VideoClip) -> VideoClip:
        """
        Intelligently resize and crop video to target format with optimal quality.
        
        Args:
            clip: Input video clip
            
        Returns:
            Processed video clip in target format
        """
        try:
            original_size = clip.size
            log.debug(f"Processing {original_size[0]}x{original_size[1]} → {self.target_format[0]}x{self.target_format[1]}")
            
            # Calculate crop parameters
            crop_params = self._calculate_smart_crop_parameters(original_size)
            
            # Apply cropping if needed
            if crop_params:
                clip = crop(clip, **crop_params)
                log.debug(f"Applied crop: {crop_params}")
            
            # Resize to target format
            resized_clip = resize(clip, newsize=self.target_format)
            
            # Verify final dimensions
            if resized_clip.size != self.target_format:
                log.warning(f"Resize verification failed: expected {self.target_format}, got {resized_clip.size}")
            
            return resized_clip
            
        except Exception as e:
            log.error(f"Smart resize failed: {e}. Using fallback basic resize.")
            return resize(clip, newsize=self.target_format)
    
    def _create_optimized_text_overlay(self, 
                                     text: str, 
                                     duration: float,
                                     font_size: int = 80,
                                     position: Union[str, Tuple] = 'center') -> Optional[TextClip]:
        """
        Create memory-optimized text overlay with proper styling.
        
        Args:
            text: Text content
            duration: Clip duration
            font_size: Font size (auto-adjusted for format)
            position: Text position
            
        Returns:
            Text clip or None if creation fails
        """
        try:
            # Adjust font size based on target format
            if self.target_format == VideoFormat.PORTRAIT:
                adjusted_font_size = min(font_size, 100)  # Limit for mobile readability
            else:
                adjusted_font_size = font_size
            
            # Create text clip with optimized settings
            text_clip = TextClip(
                text,
                fontsize=adjusted_font_size,
                color='white',
                font='Arial-Bold',
                stroke_color='black',
                stroke_width=3,
                method='caption',  # Better text rendering
                align='center'
            ).set_position(position).set_duration(duration)
            
            return text_clip
            
        except Exception as e:
            log.warning(f"Text overlay creation failed: {e}")
            return None
    
    def _create_performance_optimized_captions(self, 
                                             duration: float, 
                                             word_timings: List[Dict]) -> Optional[VideoClip]:
        """
        Create performance-optimized animated captions.
        
        Args:
            duration: Video duration
            word_timings: List of word timing data
            
        Returns:
            Caption video clip or None
        """
        try:
            if not word_timings:
                return None
            
            # Limit word count for performance
            max_words = 100 if self.system_metrics.available_ram_gb > 2.0 else 50
            limited_timings = word_timings[:max_words]
            
            # Pre-validate timing data
            valid_timings = []
            for word_data in limited_timings:
                if (isinstance(word_data, dict) and 
                    all(key in word_data for key in ['word', 'start', 'end']) and
                    isinstance(word_data['start'], (int, float)) and
                    isinstance(word_data['end'], (int, float))):
                    valid_timings.append(word_data)
            
            if not valid_timings:
                log.warning("No valid word timings found for captions")
                return None
            
            # Create optimized caption renderer
            def create_caption_frame(t: float) -> np.ndarray:
                """Create caption frame for given time."""
                try:
                    # Find active word
                    active_word = None
                    for word_data in valid_timings:
                        if word_data['start'] <= t < word_data['end']:
                            active_word = str(word_data['word']).upper().strip()[:30]  # Limit length
                            break
                    
                    if not active_word:
                        # Return transparent frame
                        frame_height = 120 if self.target_format == VideoFormat.PORTRAIT else 80
                        return np.zeros((frame_height, self.target_format[0], 3), dtype=np.uint8)
                    
                    # Create text frame
                    font_size = 70 if self.target_format == VideoFormat.PORTRAIT else 90
                    temp_text_clip = TextClip(
                        active_word,
                        fontsize=font_size,
                        color='yellow',
                        font='Arial-Bold',
                        stroke_color='black',
                        stroke_width=2
                    )
                    
                    frame = temp_text_clip.get_frame(0)
                    temp_text_clip.close()  # Immediate cleanup
                    
                    # Ensure proper format
                    if len(frame.shape) == 3 and frame.shape[2] > 3:
                        frame = frame[:, :, :3]
                    
                    return frame
                    
                except Exception as e:
                    log.debug(f"Caption frame creation failed at t={t:.2f}: {e}")
                    frame_height = 120 if self.target_format == VideoFormat.PORTRAIT else 80
                    return np.zeros((frame_height, self.target_format[0], 3), dtype=np.uint8)
            
            caption_clip = VideoClip(make_frame=create_caption_frame, duration=duration)
            return caption_clip
            
        except Exception as e:
            log.warning(f"Caption creation failed: {e}")
            return None
    
    def process_segment(self,
                       source_path: str,
                       duration: float,
                       overlay_plan: Optional[TextOverlay] = None,
                       caption_data: Optional[List[Dict[str, Any]]] = None) -> VideoClip:
        """
        Process video segment with comprehensive error handling and optimization.
        
        Args:
            source_path: Path to source video file
            duration: Target duration for the segment
            overlay_plan: Optional text overlay configuration
            caption_data: Optional caption timing data
            
        Returns:
            Processed video clip
            
        Raises:
            VideoProcessingError: If processing fails completely
        """
        with self._memory_guard(f"process_segment({os.path.basename(source_path)})"):
            start_time = time.time()
            
            try:
                # Validate inputs
                if not os.path.exists(source_path):
                    raise FileNotFoundError(f"Video file not found: {source_path}")
                
                if duration <= 0:
                    raise ValueError(f"Invalid duration: {duration}")
                
                # Load video with error handling
                try:
                    clip = VideoFileClip(source_path)
                    if not clip or not hasattr(clip, 'duration'):
                        raise VideoProcessingError("Failed to load video file properly")
                except Exception as e:
                    log.error(f"Video loading failed for {source_path}: {e}")
                    raise VideoProcessingError(f"Cannot load video: {e}")
                
                # Process video
                actual_duration = min(duration, clip.duration)
                clip = clip.subclip(0, actual_duration)
                clip = self._smart_resize_to_target_format(clip)
                clip.fps = 24  # Standard frame rate
                
                # Apply overlays based on memory availability
                current_metrics = self._update_system_metrics()
                
                if overlay_plan or caption_data:
                    if not current_metrics.is_memory_critical:
                        clip = self._apply_overlays_safely(clip, overlay_plan, caption_data)
                    else:
                        log.warning(f"Skipping overlays for {os.path.basename(source_path)} - memory critical")
                
                # Update statistics
                processing_time = time.time() - start_time
                self.processing_stats["clips_processed"] += 1
                self.processing_stats["total_processing_time"] += processing_time
                
                log.info(f"✅ Processed {os.path.basename(source_path)} in {processing_time:.2f}s")
                return clip
                
            except Exception as e:
                self.processing_stats["clips_failed"] += 1
                log.error(f"❌ Processing failed for {source_path}: {e}")
                return self._create_fallback_clip(duration, f"Failed: {os.path.basename(source_path)}")
    
    def _apply_overlays_safely(self, 
                              clip: VideoClip, 
                              overlay_plan: Optional[TextOverlay], 
                              caption_data: Optional[List[Dict[str, Any]]]) -> VideoClip:
        """Apply overlays with comprehensive error handling."""
        try:
            layers = [clip]
            
            # Add text overlay
            if overlay_plan and overlay_plan.text_content:
                text_overlay = self._create_optimized_text_overlay(
                    overlay_plan.text_content,
                    clip.duration,
                    overlay_plan.font_size
                )
                if text_overlay:
                    layers.append(text_overlay)
            
            # Add captions if memory allows
            current_metrics = self._update_system_metrics()
            if caption_data and current_metrics.available_ram_gb > 1.0:
                caption_overlay = self._create_performance_optimized_captions(
                    clip.duration, 
                    caption_data
                )
                if caption_overlay:
                    # Position captions appropriately for format
                    if self.target_format == VideoFormat.PORTRAIT:
                        position = ('center', 'bottom')
                    else:
                        position = ('center', 0.85)
                    
                    layers.append(caption_overlay.set_position(position, relative=True))
            
            # Composite layers if we have overlays
            if len(layers) > 1:
                composite = CompositeVideoClip(layers, size=self.target_format)
                composite.fps = clip.fps
                if hasattr(clip, 'audio') and clip.audio:
                    composite.audio = clip.audio
                return composite
            
            return clip
            
        except Exception as e:
            log.warning(f"Overlay application failed: {e}. Using clip without overlays.")
            return clip
    
    def _create_fallback_clip(self, duration: float, error_message: str = "Processing Failed") -> VideoClip:
        """Create a fallback clip for failed processing."""
        try:
            # Create simple colored background
            def make_fallback_frame(t):
                # Create a simple gradient background
                frame = np.zeros((self.target_format[1], self.target_format[0], 3), dtype=np.uint8)
                frame[:, :] = [40, 40, 80]  # Dark blue background
                return frame
            
            fallback_clip = VideoClip(make_frame=make_fallback_frame, duration=max(1.0, duration))
            fallback_clip.fps = 24
            
            # Try to add error message text
            try:
                error_text = self._create_optimized_text_overlay(
                    error_message,
                    fallback_clip.duration,
                    font_size=60,
                    position='center'
                )
                if error_text:
                    fallback_clip = CompositeVideoClip([fallback_clip, error_text], size=self.target_format)
                    fallback_clip.fps = 24
            except:
                pass  # Use plain background if text fails
            
            return fallback_clip
            
        except Exception as e:
            log.critical(f"Fallback clip creation failed: {e}")
            # Ultimate fallback - simple black clip
            return VideoClip(
                make_frame=lambda t: np.zeros((self.target_format[1], self.target_format[0], 3), dtype=np.uint8),
                duration=max(1.0, duration)
            ).set_fps(24)
    
    def _determine_optimal_render_settings(self) -> RenderQuality:
        """Determine optimal render settings based on system resources."""
        current_metrics = self._update_system_metrics()
        
        if (current_metrics.available_ram_gb > 3.0 and 
            current_metrics.cpu_usage_percent < 70):
            return RenderQuality.HIGH
        elif (current_metrics.available_ram_gb > 1.5 and 
              current_metrics.cpu_usage_percent < 85):
            return RenderQuality.MEDIUM
        else:
            return RenderQuality.LOW
    
    def render_video(self, 
                    clips: List[VideoClip], 
                    audio_track: Optional[AudioFileClip], 
                    title: str) -> str:
        """
        Render final video with intelligent resource management.
        
        Args:
            clips: List of video clips to concatenate
            audio_track: Optional audio track
            title: Video title for filename
            
        Returns:
            Path to rendered video file
            
        Raises:
            VideoProcessingError: If rendering fails completely
        """
        output_path = os.path.join(OUTPUT_DIR, f"{sanitize_filename(title)}.mp4")
        
        with self._memory_guard("video_render"):
            try:
                # Smart clip management
                current_metrics = self._update_system_metrics()
                max_clips = current_metrics.recommended_max_clips
                
                log.info(f"Render setup: {len(clips)} clips requested, "
                        f"{current_metrics.available_ram_gb:.1f}GB available, "
                        f"max {max_clips} clips recommended")
                
                # Filter and limit clips
                valid_clips = [c for c in clips 
                              if c is not None and hasattr(c, 'duration') and c.duration > 0]
                
                if len(valid_clips) > max_clips:
                    log.warning(f"Limiting clips from {len(valid_clips)} to {max_clips} for memory safety")
                    valid_clips = valid_clips[:max_clips]
                
                if not valid_clips:
                    log.error("No valid clips for rendering")
                    valid_clips = [self._create_fallback_clip(5.0, "No Content Available")]
                
                # Smart duration management
                total_duration = sum(c.duration for c in valid_clips)
                max_duration = 120.0 if current_metrics.available_ram_gb > 2.0 else 60.0
                
                if total_duration > max_duration:
                    log.info(f"Video duration ({total_duration:.1f}s) exceeds limit ({max_duration}s). Scaling clips.")
                    scale_factor = max_duration / total_duration
                    scaled_clips = []
                    
                    for clip in valid_clips:
                        new_duration = max(0.5, clip.duration * scale_factor)
                        try:
                            scaled_clip = clip.subclip(0, new_duration)
                            scaled_clip.fps = 24
                            scaled_clips.append(scaled_clip)
                        except Exception as e:
                            log.warning(f"Clip scaling failed: {e}")
                            scaled_clips.append(clip)  # Keep original if scaling fails
                    
                    valid_clips = scaled_clips
                
                # Concatenate clips
                log.info(f"Concatenating {len(valid_clips)} clips...")
                final_video = concatenate_videoclips(valid_clips)
                final_video.fps = 24
                
                # Add audio if available and memory permits
                if audio_track and current_metrics.available_ram_gb > 1.0:
                    try:
                        # Sync audio duration with video
                        if audio_track.duration > final_video.duration:
                            audio_track = audio_track.subclip(0, final_video.duration)
                        elif audio_track.duration < final_video.duration * 0.8:
                            # Loop audio if significantly shorter
                            loops_needed = int(np.ceil(final_video.duration / audio_track.duration))
                            audio_track = afx.concatenate_audioclips([audio_track] * loops_needed)
                            audio_track = audio_track.subclip(0, final_video.duration)
                        
                        final_video.audio = audio_track
                        log.info("✅ Audio track added successfully")
                        
                    except Exception as e:
                        log.warning(f"Audio processing failed: {e}. Rendering without audio.")
                
                # Determine render settings
                render_quality = self._determine_optimal_render_settings()
                render_settings = render_quality.value
                
                log.info(f"Rendering with {render_quality.name} quality: {render_settings}")
                
                # Perform render
                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    logger='bar',
                    **render_settings
                )
                
                # Cleanup and final verification
                final_video.close()
                for clip in valid_clips:
                    try:
                        clip.close()
                    except Exception:
                        pass
                
                gc.collect()
                
                # Verify output
                if not os.path.exists(output_path):
                    raise VideoProcessingError("Output file was not created")
                
                file_size = os.path.getsize(output_path)
                if file_size < 1000:  # Less than 1KB indicates failure
                    raise VideoProcessingError(f"Output file too small ({file_size} bytes)")
                
                # Log final statistics
                final_metrics = self._update_system_metrics()
                log.info(f"✅ Video rendered successfully: {output_path}")
                log.info(f"📊 Processing stats: {self.processing_stats['clips_processed']} clips, "
                        f"{self.processing_stats['clips_failed']} failed, "
                        f"peak RAM: {self.processing_stats['memory_peak_usage']:.1f}GB")
                
                return output_path
                
            except Exception as e:
                log.error(f"❌ Video rendering failed: {e}")
                
                # Emergency render attempt
                try:
                    log.info("🚨 Attempting emergency render...")
                    emergency_clip = self._create_fallback_clip(10.0, "Render Failed - Emergency Mode")
                    emergency_clip.write_videofile(
                        output_path,
                        fps=24,
                        preset='ultrafast',
                        threads=1,
                        logger=None
                    )
                    emergency_clip.close()
                    return output_path
                    
                except Exception as emergency_error:
                    log.critical(f"❌ Emergency render also failed: {emergency_error}")
                    raise VideoProcessingError(f"Complete render failure: {e}")

# Maintain backward compatibility
VideoEditor = SmartVideoEditor