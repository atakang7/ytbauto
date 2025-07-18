"""
This service handles all audio-related tasks, including Text-to-Speech (TTS)
generation and Automatic Speech Recognition (ASR) for captions.
"""
# --- Standard Library Imports ---
import asyncio
import base64
import html
import logging
import os
from typing import Any, Dict, List, Optional

# --- Third-Party Imports ---
from openai import OpenAI
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from tqdm.asyncio import tqdm as asyncio_tqdm
from moviepy.editor import AudioFileClip, CompositeAudioClip, afx

# --- Local Application Imports ---
from config import (
    OPENAI_TTS_MODEL, OPENAI_TTS_VOICE, SPEECHIFY_DEFAULT_VOICE_ID,
    TEMP_ASSETS_DIR
)
from models import VideoPlan
from utils import sanitize_filename

log = logging.getLogger(__name__)

# Optional dependencies are handled locally
try:
    import whisper
except ImportError:
    whisper = None
try:
    from speechify import Speechify
except ImportError:
    Speechify = None

class AudioService:
    def __init__(self, openai_client: OpenAI, speechify_client: Optional[Speechify]):
        self.openai_client = openai_client
        self.speechify_client = speechify_client
        self.asr_model = self._load_asr_model()

    def mix_audio_with_narration(
        self,
        primary_audio: AudioFileClip,
        music_path: Optional[str],
        ducking_level: float = 0.15
        ) -> CompositeAudioClip:
            """Mixes a primary audio track (like narration) with background music."""
            if not music_path or not os.path.exists(music_path):
                log.warning("Background music path not found or not provided. Skipping audio mix.")
                return primary_audio

            log.info("Mixing final audio track with background music...")
            with AudioFileClip(music_path) as music_clip:
                processed_music = music_clip.fx(afx.audio_normalize).volumex(ducking_level)
                
                if processed_music.duration > primary_audio.duration:
                    processed_music = processed_music.subclip(0, primary_audio.duration)
                else:
                    processed_music = afx.audio_loop(processed_music, duration=primary_audio.duration)
                    
                return CompositeAudioClip([primary_audio, processed_music])
    def _load_asr_model(self) -> Optional[Any]:
        if not whisper:
            return None
        log.info("Loading ASR model...")
        try:
            model = whisper.load_model("small", device="cpu")
            log.info("ASR model loaded successfully.")
            return model
        except Exception as e:
            log.error(f"Could not load ASR model: {e}")
            return None

    def get_audio_duration(self, filepath: str) -> float:
        try:
            if filepath.lower().endswith(".mp3"):
                audio = MP3(filepath)
            elif filepath.lower().endswith(".wav"):
                audio = WAVE(filepath)
            else:
                return 0.0
            return audio.info.length
        except Exception:
            return 0.0

    async def generate_and_process_audio(
        self, plan: VideoPlan, persona: Dict[str, Any]
    ) -> Dict[str, Dict]:
        base_filename = sanitize_filename(plan.video_title)
        tts_preference = "speechify" if self.speechify_client else "openai"
        all_sub_scenes_with_ids = self._get_all_sub_scenes_with_ids(plan)

        coroutines = [
            self._generate_single_tts_segment(
                scene,
                os.path.join(TEMP_ASSETS_DIR, f"tts_{base_filename}_{scene['id']}"),
                tts_preference,
                persona,
            )
            for scene in all_sub_scenes_with_ids
        ]
        results = await asyncio_tqdm.gather(*coroutines, desc="Generating TTS audio")
        processed_segments = {}
        for seg_data, filepath in zip(all_sub_scenes_with_ids, results):
            scene_id = seg_data["id"]
            if filepath and (duration := self.get_audio_duration(filepath)) > 0:
                processed_segments[scene_id] = {
                    **seg_data,
                    "filepath": os.path.abspath(filepath),
                    "duration": duration,
                }
        self._transcribe_audio_segments(list(processed_segments.values()))
        return processed_segments

    def _get_all_sub_scenes_with_ids(self, plan: VideoPlan) -> List[Dict]:
        all_sub_scenes = []
        for i, section in enumerate(plan.sections):
            for j, sub_scene in enumerate(section.sub_scenes):
                all_sub_scenes.append(
                    {
                        "id": f"{i}_{j}",
                        "narration": sub_scene.narration_text,
                        "emotion": sub_scene.emotion,
                        "keywords": sub_scene.keywords_for_highlighting,
                    }
                )
        if plan.call_to_action_text:
            all_sub_scenes.append(
                {
                    "id": "cta",
                    "narration": plan.call_to_action_text,
                    "emotion": "upbeat",
                    "keywords": [],
                }
            )
        return all_sub_scenes

    def _construct_ssml(
        self, script_text: str, emotion: str, persona: Dict[str, Any]
    ) -> str:
        prosody_map = persona.get("speech_settings", {}).get("emotion_prosody_map", {})
        prosody_attrs = prosody_map.get(emotion, {})
        rate, pitch = prosody_attrs.get("rate", "medium"), prosody_attrs.get(
            "pitch", "medium"
        )
        ssml_body = f'<prosody rate="{rate}" pitch="{pitch}">{html.escape(script_text)}</prosody>'
        return f"<speak>{ssml_body}</speak>"

    async def _generate_single_tts_segment(
        self, scene_data: Dict, output_base: str, tts_pref: str, persona: Dict[str, Any]
    ) -> Optional[str]:
        if not (narration_text := scene_data.get("narration")):
            return None
        if tts_pref == "speechify" and self.speechify_client:
            output_filename = f"{output_base}_speechify.wav"
            ssml_input = self._construct_ssml(
                narration_text, scene_data.get("emotion", "neutral"), persona
            )
            if await asyncio.to_thread(
                self._speechify_tts_sync, ssml_input, SPEECHIFY_DEFAULT_VOICE_ID, output_filename
            ):
                return output_filename
            log.warning("Speechify SSML TTS failed. Falling back to OpenAI TTS.")
        output_filename = f"{output_base}_openai.mp3"
        try:
            response = await asyncio.to_thread(
                self.openai_client.audio.speech.create,
                model=OPENAI_TTS_MODEL,
                voice=OPENAI_TTS_VOICE,
                input=narration_text,
                response_format="mp3",
            )
            with open(output_filename, "wb") as f:
                f.write(response.content)
            if os.path.exists(output_filename) and os.path.getsize(output_filename) > 100:
                return output_filename
        except Exception as e:
            log.error(f"Failed to generate OpenAI TTS: {e}")
        return None

    def _speechify_tts_sync(
        self, ssml: str, voice_id: str, output_path: str
    ) -> bool:
        try:
            response = self.speechify_client.tts.audio.speech(
                input=ssml, voice_id=voice_id, audio_format="wav"
            )
            if response and response.audio_data:
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(response.audio_data))
                if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                    return True
        except Exception as e:
            log.warning(f"Speechify TTS error: {e}")
        return False

    def _transcribe_audio_segments(self, audio_segments: List[Dict]):
        if not self.asr_model:
            return
        for seg in audio_segments:
            if not (filepath := seg.get("filepath")):
                continue
            try:
                result = self.asr_model.transcribe(
                    filepath, word_timestamps=True, fp16=False
                )
                seg["asr_word_timings"] = [
                    word
                    for s in result.get("segments", [])
                    for word in s.get("words", [])
                ]
            except Exception as e:
                log.error(f"ASR transcription failed for {filepath}: {e}")
                seg["asr_word_timings"] = []