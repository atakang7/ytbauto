"""
This service handles the assembly and rendering of a remixed video using
the VideoEditor API for the Transformative Mode.
"""
# --- Standard Library Imports ---
import asyncio
import logging
import os
from typing import Dict, List

# --- Third-Party Imports ---
from moviepy.editor import AudioFileClip, concatenate_audioclips, concatenate_videoclips

# --- Local Application Imports ---
from config import TEMP_ASSETS_DIR
from models import RemixPlan, AnalysedScene
from services.media_service import MediaService
from services.audio_service import AudioService
from utils import sanitize_filename
from video_processing.editor import VideoEditor

log = logging.getLogger(__name__)

class RemixAssemblyService:
    def __init__(self, media_service: MediaService, audio_service: AudioService, editor: VideoEditor):
        self.media_service = media_service
        self.audio_service = audio_service
        self.editor = editor

    async def _generate_voiceovers(self, plan: RemixPlan) -> Dict[int, str]:
        if not plan.voiceover_segments:
            return {}
        
        log.info(f"Generating {len(plan.voiceover_segments)} voiceover segments...")
        tasks = []
        base_filename = sanitize_filename(plan.remix_video_title)
        
        for vo_segment in plan.voiceover_segments:
            scene_data = {
                "id": f"vo_{vo_segment.scene_id}",
                "narration": vo_segment.narration_text,
                "emotion": "neutral"
            }
            output_base = os.path.join(TEMP_ASSETS_DIR, f"tts_{base_filename}_{scene_data['id']}")
            tasks.append(
                self.audio_service._generate_single_tts_segment(scene_data, output_base, "openai", {})
            )
            
        generated_paths = await asyncio.gather(*tasks)
        
        vo_audio_paths = {}
        for vo_segment, path in zip(plan.voiceover_segments, generated_paths):
            if path:
                vo_audio_paths[vo_segment.scene_id] = path
        return vo_audio_paths

    async def assemble_and_render_remix(self, plan: RemixPlan, scenes: List[AnalysedScene]):
        log.info(f"🚀 Starting hybrid remix assembly for '{plan.remix_video_title}'...")
        
        vo_audio_paths, music_path = await asyncio.gather(
            self._generate_voiceovers(plan),
            self.media_service._fetch_music_from_pixabay(plan.background_music_suggestion)
        )

        scene_map = {s.scene_id: s for s in scenes}
        processed_clips = []
        primary_audio_clips = []

        for scene_id in plan.scene_ids_to_include:
            if scene_id not in scene_map:
                log.warning(f"Scene ID {scene_id} from plan not found. Skipping.")
                continue

            scene_info = scene_map[scene_id]
            clip = self.editor.create_clip_from_source(
                plan.source_video_path,
                scene_info.start_time_seconds,
                scene_info.end_time_seconds
            )
            
            if plan.text_overlays:
                for overlay in plan.text_overlays:
                    if overlay.scene_id == scene_id:
                        clip = self.editor.add_static_overlay(clip, overlay)
            
            processed_clips.append(clip)

            if scene_id in vo_audio_paths:
                primary_audio_clips.append(AudioFileClip(vo_audio_paths[scene_id]))
            elif clip.audio is not None:
                primary_audio_clips.append(clip.audio)

        if not processed_clips:
            log.error("No valid scenes found to create a remix. Aborting."); return

        final_video_track = concatenate_videoclips(processed_clips)
        
        primary_audio_track = None
        if primary_audio_clips:
            primary_audio_track = concatenate_audioclips(primary_audio_clips)

        if music_path and primary_audio_track:
            final_audio_track = self.editor.mix_audio(primary_audio_track, music_path)
            final_video_track.audio = final_audio_track
        elif primary_audio_track:
            final_video_track.audio = primary_audio_track
        
        self.editor.render_video(final_video_track, plan.remix_video_title)