"""
This service handles the analysis of a source video file, including property
extraction, scene detection, and AI-powered content tagging.
"""
# --- Standard Library Imports ---
import asyncio
import base64
import logging
import os
from typing import Any, Dict, List

# --- Third-Party Imports ---
from openai import OpenAI
from moviepy.editor import VideoFileClip
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from tqdm.asyncio import tqdm as asyncio_tqdm

# --- Local Application Imports ---
from config import TEMP_ASSETS_DIR, SCENE_DETECT_THRESHOLD
from models import AnalysedScene

log = logging.getLogger(__name__)

class VideoAnalysisService:
    def __init__(self, video_path: str, openai_client: OpenAI):
        self.video_path = video_path
        self.clip = VideoFileClip(video_path)
        self.openai_client = openai_client

    def get_video_properties(self) -> Dict[str, Any]:
        props = {
            "duration": self.clip.duration,
            "fps": self.clip.fps,
            "resolution": self.clip.size,
        }
        log.info(f"Video Properties: {props}")
        return props

    def detect_scenes(self) -> List[AnalysedScene]:
        log.info("Starting scene detection...")
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=SCENE_DETECT_THRESHOLD))
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager, show_progress=True)
        scene_list_raw = scene_manager.get_scene_list()

        if not scene_list_raw:
            log.warning("No scenes detected. Treating the whole video as a single scene.")
            return [
                AnalysedScene(
                    scene_id=0,
                    start_time_seconds=0.0,
                    end_time_seconds=self.clip.duration,
                    duration_seconds=self.clip.duration,
                )
            ]

        analysed_scenes = [
            AnalysedScene(
                scene_id=i,
                start_time_seconds=start.get_seconds(),
                end_time_seconds=end.get_seconds(),
                duration_seconds=(end - start).get_seconds(),
            )
            for i, (start, end) in enumerate(scene_list_raw)
        ]
        log.info(f"Detected {len(analysed_scenes)} scenes.")
        return analysed_scenes

    def _extract_frame_as_base64(self, time_seconds: float) -> str:
        frame_path = os.path.join(TEMP_ASSETS_DIR, f"frame_at_{time_seconds:.2f}.jpg")
        self.clip.save_frame(frame_path, t=time_seconds)
        with open(frame_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        os.remove(frame_path)
        return encoded_string

    async def _tag_single_scene(self, scene: AnalysedScene) -> List[str]:
        midpoint_time = scene.start_time_seconds + (scene.duration_seconds / 2)
        try:
            base64_frame = await asyncio.to_thread(
                self._extract_frame_as_base64, midpoint_time
            )
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image using 3-5 concise keywords. Focus on objects, actions, and mood. Example: 'person smiling, office, bright'. Respond with only the keywords separated by commas.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_frame}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=50,
            )
            tags = [
                tag.strip() for tag in response.choices[0].message.content.split(",")
            ]
            return tags
        except Exception as e:
            log.error(f"Could not tag scene {scene.scene_id}: {e}")
            return []

    async def tag_scenes_with_vision(
        self, scenes: List[AnalysedScene]
    ) -> List[AnalysedScene]:
        log.info(f"Starting AI vision tagging for {len(scenes)} scenes...")
        tagging_coroutines = [self._tag_single_scene(scene) for scene in scenes]
        
        all_tags = await asyncio_tqdm.gather(*tagging_coroutines, desc="Tagging video scenes")
        
        for scene, tags in zip(scenes, all_tags):
            scene.tags = tags
        return scenes