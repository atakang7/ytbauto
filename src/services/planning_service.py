"""
This service handles all AI-driven planning for both Generative and
Transformative modes using an LLM.
"""
# --- Standard Library Imports ---
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

# --- Third-Party Imports ---
from openai import OpenAI
from pydantic import ValidationError

# --- Local Application Imports ---
from config import MAX_RETRIES
from models import AnalysedScene, RemixPlan, VideoPlan

log = logging.getLogger(__name__)

class PlanningService:
    def __init__(self, client: OpenAI):
        self.client = client

    def load_brand_persona(self, persona_file_path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(persona_file_path):
            log.error(f"Persona file not found at '{persona_file_path}'.")
            return None
        try:
            with open(persona_file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log.error(f"Error loading or parsing persona file '{persona_file_path}': {e}")
            return None

    def create_generative_plan(
        self, user_prompt: str, persona: Dict[str, Any]
    ) -> Optional[VideoPlan]:
        log.info("Starting AI video planning process...")
        plan_schema = VideoPlan.schema()
        raw_json = self._run_ai_director(user_prompt, persona, plan_schema)

        if not raw_json:
            log.error("AI director returned no content after multiple retries.")
            return None
        try:
            video_plan = VideoPlan.parse_obj(raw_json)
            log.info("AI plan successfully validated against data structure.")
            return video_plan
        except ValidationError as e:
            log.error(f"AI-generated plan failed validation: {e}")
            return None

    def _run_ai_director(
        self, user_prompt: str, persona: Dict[str, Any], schema: Dict
    ) -> Optional[Dict]:
        log.info("Generating scene-by-scene plan with AI Director...")
        supported_emotions = list(
            persona.get("speech_settings", {}).get("emotion_prosody_map", {}).keys()
        )
        system_prompt = f"""
        You are a meticulous video director for engaging, short-form vertical videos. Your primary task is to generate a complete JSON plan that results in a dynamic and visually interesting video.

        Follow these rules strictly:
        1. Break the script down into individual sentences. Each sentence will become a 'SubScene'.
        2. For each 'SubScene', you MUST create a specific and descriptive 'visual_search_query' for Pexels that perfectly matches the narration's content and emotion.
        3. For each 'SubScene', assign a 'motion_type' from the allowed values to create a dynamic Ken Burns effect. Vary the motions.
        4. The final output MUST be a single, valid JSON object that strictly adheres to the provided JSON Schema.

        JSON Schema: {json.dumps(schema, indent=2)}
        Valid Emotions for the 'emotion' field: {supported_emotions}
        """
        for attempt in range(MAX_RETRIES):
            try:
                log.info(f"AI Director Attempt {attempt + 1}/{MAX_RETRIES}...")
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Create a video plan for: '{user_prompt}'"},
                    ],
                    response_format={"type": "json_object"},
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                log.warning(f"OpenAI Director attempt {attempt + 1} FAILED: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(4 + attempt * 2)
        return None

    async def create_remix_plan(
        self, remix_query: str, tagged_scenes: List[AnalysedScene], source_video_path: str
    ) -> Optional[RemixPlan]:
        log.info("Creating Remix Plan with AI Video Editor...")
        scenes_summary = "\n".join(
            [
                f"Scene {s.scene_id} (duration {s.duration_seconds:.1f}s): tags={s.tags}"
                for s in tagged_scenes
            ]
        )
        plan_schema = RemixPlan.schema()

        system_prompt = f"""
        You are an expert video editor. Your task is to create a compelling short video by selecting scenes, and optionally adding new voiceovers and text overlays.
        You will be given a user's request and a list of available scenes with their duration and AI-generated descriptive tags.

        Your Logic:
        1.  **Select Scenes**: Based on the user's request, choose the `scene_ids_to_include` in the order they should appear.
        2.  **Suggest Music**: Suggest a `background_music_suggestion` genre suitable for the user's request.
        3.  **Add Voiceovers (Optional)**: If the user's request implies narration or summary (e.g., "summarize this video," "explain the key moments"), generate concise `voiceover_segments`. For a simple highlight reel, this can be null.
        4.  **Add Text Overlays (Optional)**: If the user's request implies highlighting or titles (e.g., "show the best parts," "add titles to each section"), generate `text_overlays`. For a simple music-driven edit, this can be null.

        You MUST return ONLY a valid JSON object that adheres to this schema. The `text_overlays` and `voiceover_segments` fields are optional.
        JSON Schema: {json.dumps(plan_schema, indent=2)}
        """
        user_content = (
            f"User Request: '{remix_query}'\n\nAvailable Scenes:\n{scenes_summary}"
        )

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
            )
            plan_data = json.loads(response.choices[0].message.content)
            plan_data["source_video_path"] = source_video_path
            return RemixPlan.parse_obj(plan_data)
        except Exception as e:
            log.error(f"Failed to create remix plan: {e}")
            return None