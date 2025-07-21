"""
This service handles the final assembly for the Generative Mode.
It uses the re-architected VideoEditor to compose and render the video reliably.
"""
# --- Standard Library Imports ---
import logging
from typing import Dict, Any, Optional

# --- Third-Party Imports ---
from moviepy.editor import AudioFileClip, concatenate_audioclips

# --- Local Application Imports ---
# We now need SubScene and TextOverlay to manually build the CTA scene
from models import VideoPlan, SubScene, TextOverlay
from video_processing.editor import VideoEditor
from services.media_service import MediaService
from services.audio_service import AudioService

log = logging.getLogger(__name__)

class GenerativeAssemblyService:
    def __init__(self, editor: VideoEditor, media_service: MediaService, audio_service: AudioService):
        self.editor = editor
        self.media_service = media_service
        self.audio_service = audio_service

    async def assemble_video(
        self, plan: VideoPlan, processed_audio: Dict, media_assets: Dict
    ):
        """
        Assembles the video using the re-architected VideoEditor.
        This method processes each scene into a uniform clip, mixes the final audio,
        and then renders the complete video.
        """
        log.info(f"🚀 Starting generative assembly for '{plan.video_title}'...")

        # --- 1. Create a flattened list of all scenes to process ---
        scene_map = []
        for i, section in enumerate(plan.sections):
            for j, sub_scene in enumerate(section.sub_scenes):
                scene_map.append({"scene": sub_scene, "id": f"{i}_{j}"})

        # Manually create the Call to Action scene object
        if plan.call_to_action_text:
            # **THE FIX**: Use integer 999 instead of string "cta" for scene_id
            cta_overlay = TextOverlay(
                text_content=plan.call_to_action_text,
                scene_id=999
            )
            cta_scene = SubScene(
                narration_text=plan.call_to_action_text,
                visual_search_query="abstract background",
                emotion="upbeat"
            )
            scene_map.append({"scene": cta_scene, "id": "cta"})

        # --- 2. Process all video segments into a uniform list ---
        processed_segments = []
        narration_clips = []
        for scene_item in scene_map:
            scene_obj = scene_item["scene"]
            scene_id = scene_item["id"]

            video_path = media_assets.get("visuals", {}).get(scene_id)
            audio_path = processed_audio.get(scene_id, {}).get("filepath")
            
            if not video_path or not audio_path:
                log.warning(f"Missing video or audio asset for scene {scene_id}. Skipping.")
                continue
            
            narration_clip = AudioFileClip(audio_path)
            narration_clips.append(narration_clip)
            
            # Don't pass text_overlay since SubScene doesn't have that attribute
            segment = self.editor.process_segment(
                source_path=video_path,
                duration=narration_clip.duration,
                overlay_plan=cta_overlay if scene_id == "cta" else None,
                caption_data=processed_audio.get(scene_id, {}).get("asr_word_timings")
            )
            processed_segments.append(segment)
        
        if not processed_segments:
            log.error("No clips were assembled. Aborting render.")
            return

        # --- 3. Create the final audio track ---
        full_narration = concatenate_audioclips(narration_clips)
        music_path = media_assets.get("music")
        final_audio_track = self.audio_service.mix_audio_with_narration(full_narration, music_path)
        
        # --- 4. Render the final video ---
        self.editor.render_video(
            clips=processed_segments,
            audio_track=final_audio_track,
            title=plan.video_title
        )

        # Clean up audio clips to free memory
        full_narration.close()
        final_audio_track.close()
        for clip in narration_clips:
            clip.close()