"""
VideoEditor API - Re-architected for Stability

This module provides a robust, high-level interface for video manipulation.
It enforces uniform sizing and color formatting to prevent common rendering errors.
"""
# --- Standard Library Imports ---
import logging
import os
from typing import List, Dict, Any, Optional, Tuple

# --- Third-Party Imports ---
import numpy as np
from moviepy.editor import (
    VideoFileClip, AudioFileClip, TextClip,
    CompositeVideoClip, concatenate_videoclips,
    CompositeAudioClip, afx, VideoClip
)
from moviepy.video.fx.all import resize

# --- Local Application Imports ---
from config import OUTPUT_DIR
from models import TextOverlay
from utils import sanitize_filename

log = logging.getLogger(__name__)

# --- Core Configuration ---
PROJECT_SIZE = (1920, 1080)

class VideoEditor:
    """A robust, re-architected video editor for stable automated production."""

    def process_segment(
        self,
        source_path: str,
        duration: float,
        overlay_plan: Optional[TextOverlay] = None,
        caption_data: Optional[List[Dict[str, Any]]] = None,
    ) -> VideoClip:
        """Processes a single video segment from start to finish."""
        log.info(f"Processing segment from '{os.path.basename(source_path)}'")
        try:
            with VideoFileClip(source_path) as clip:
                sub_clip = clip.subclip(0, duration)
                resized_clip = resize(sub_clip, newsize=PROJECT_SIZE)

                if overlay_plan or caption_data:
                    final_clip = self._apply_overlays(resized_clip, overlay_plan, caption_data)
                else:
                    final_clip = resized_clip

                if not final_clip.audio and clip.audio:
                    final_clip.audio = clip.audio.subclip(0, duration)
                return final_clip
        except Exception as e:
            log.error(f"Failed to process segment {source_path}: {e}")
            return VideoClip(make_frame=lambda t: np.zeros((PROJECT_SIZE[1], PROJECT_SIZE[0], 3), dtype=np.uint8), duration=duration)

    def _apply_overlays(
        self,
        base_clip: VideoClip,
        overlay_plan: Optional[TextOverlay],
        caption_data: Optional[List[Dict[str, Any]]]
    ) -> VideoClip:
        """A dedicated, safe method for adding overlays to a clip."""
        base_clip_rgba = base_clip.fl_image(
            lambda frame: np.dstack([frame, np.full(frame.shape[:2], 255, dtype=np.uint8)]) if frame.shape[2] == 3 else frame
        )
        layers = [base_clip_rgba]

        if overlay_plan:
            txt_clip = TextClip(
                overlay_plan.text_content, fontsize=overlay_plan.font_size, color='white',
                font='Arial-Bold', stroke_color='black', stroke_width=2
            ).set_pos(overlay_plan.position).set_duration(base_clip.duration)
            layers.append(txt_clip)

        if caption_data:
            caption_layer = self._create_caption_layer(base_clip.duration, caption_data, base_clip.size)
            layers.append(caption_layer.set_position(("center", 0.8), relative=True))

        composite_clip = CompositeVideoClip(layers, size=PROJECT_SIZE)
        if base_clip.audio:
            composite_clip.audio = base_clip.audio
            
        return composite_clip.fl_image(lambda frame: frame[:, :, :3])

    def _create_caption_layer(self, duration: float, word_timings: List, size: Tuple[int, int]) -> VideoClip:
        """Helper to generate a transparent VideoClip with animated captions."""
        font_kwargs = {
            "fontsize": 90, "color": "white", "font": "Arial-Bold",
            "stroke_color": "black", "stroke_width": 3,
        }
        word_cache = {}
        def make_frame(t):
            active_word = next((w['word'] for w in word_timings if w['start'] <= t < w['end']), None)
            if not active_word:
                return np.zeros((size[1], size[0], 4), dtype=np.uint8)
            active_word = active_word.upper()
            if active_word in word_cache:
                return word_cache[active_word]
            with TextClip(active_word, **font_kwargs) as txt_clip:
                frame = txt_clip.get_frame(0)
                word_cache[active_word] = frame
                return frame
        return VideoClip(make_frame=make_frame, duration=duration, ismask=False)

    def render_video(self, clips: List[VideoClip], audio_track: AudioFileClip, title: str) -> str:
        """Concatenates uniform clips and renders the final video with CPU control."""
        output_path = os.path.join(OUTPUT_DIR, f"{sanitize_filename(title)}.mp4")
        log.info(f"All segments processed. Concatenating final video.")

        with concatenate_videoclips(clips) as final_video:
            final_video.audio = audio_track
            render_preset = 'fast'
            log.info(f"Rendering final video to '{output_path}' using preset '{render_preset}'.")

            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                threads=os.cpu_count(),
                preset=render_preset,
                logger='bar'
            )
        log.info(f"✅ Video successfully rendered: {output_path}")
        for clip in clips:
            clip.close()
        return output_path