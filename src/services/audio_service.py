"""
This service handles all audio-related tasks, including Text-to-Speech (TTS)
generation and Automatic Speech Recognition (ASR) for captions.

Fixed to ensure reliable subtitle generation while maintaining memory efficiency.
"""
# --- Standard Library Imports ---
import asyncio
import base64
import gc
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
        self.asr_model = None  # Lazy load when needed

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
            music_clip = None
            try:
                music_clip = AudioFileClip(music_path)
                processed_music = music_clip.fx(afx.audio_normalize).volumex(ducking_level)
                
                if processed_music.duration > primary_audio.duration:
                    processed_music = processed_music.subclip(0, primary_audio.duration)
                else:
                    processed_music = afx.audio_loop(processed_music, duration=primary_audio.duration)
                    
                result = CompositeAudioClip([primary_audio, processed_music])
                return result
            except Exception as e:
                log.error(f"Audio mixing failed: {e}")
                return primary_audio
            finally:
                # Clean up music clip immediately
                if music_clip:
                    try:
                        music_clip.close()
                    except:
                        pass

    def _get_or_load_asr_model(self) -> Optional[Any]:
        """Get ASR model, loading it if necessary."""
        if self.asr_model is not None:
            return self.asr_model
            
        if not whisper:
            log.warning("Whisper not available - subtitles will be disabled")
            return None
            
        log.info("Loading Whisper ASR model for subtitles...")
        try:
            # Use 'base' model for better accuracy while keeping memory reasonable
            self.asr_model = whisper.load_model("base", device="cpu")
            log.info("✅ Whisper model loaded successfully - subtitles enabled")
            return self.asr_model
        except Exception as e:
            log.error(f"❌ Failed to load Whisper model: {e} - subtitles disabled")
            return None

    def _unload_asr_model_if_needed(self):
        """Unload ASR model only when we're completely done with all transcription."""
        # Keep model loaded during processing - only unload at the very end
        pass

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
        
        # STRICT SINGLE PROVIDER: Choose once and NEVER switch
        if self.speechify_client:
            tts_provider = "speechify"
            log.info("🎤 LOCKED to Speechify for entire video - no fallbacks allowed")
        else:
            tts_provider = "openai"
            log.info("🎤 LOCKED to OpenAI TTS for entire video - no fallbacks allowed")
        
        all_sub_scenes_with_ids = self._get_all_sub_scenes_with_ids(plan)
        log.info(f"Processing {len(all_sub_scenes_with_ids)} segments with SINGLE provider: {tts_provider}")

        # Generate TTS for all segments with SAME provider - FAIL if provider fails
        coroutines = [
            self._generate_single_tts_segment_with_retries(
                scene,
                os.path.join(TEMP_ASSETS_DIR, f"tts_{base_filename}_{scene['id']}"),
                tts_provider,
                persona,
            )
            for scene in all_sub_scenes_with_ids
        ]
        results = await asyncio_tqdm.gather(*coroutines, desc=f"Generating TTS with {tts_provider}")
        
        # Build processed segments dict - track failures
        processed_segments = {}
        failed_segments = 0
        
        for seg_data, filepath in zip(all_sub_scenes_with_ids, results):
            scene_id = seg_data["id"]
            if filepath and (duration := self.get_audio_duration(filepath)) > 0:
                processed_segments[scene_id] = {
                    **seg_data,
                    "filepath": os.path.abspath(filepath),
                    "duration": duration,
                }
            else:
                failed_segments += 1
                log.error(f"❌ Scene {scene_id} FAILED - no audio generated with {tts_provider}")

        # Report TTS results
        total_segments = len(all_sub_scenes_with_ids)
        success_segments = len(processed_segments)
        
        log.info(f"🎤 TTS Results: {success_segments}/{total_segments} segments successful with {tts_provider}")
        
        if failed_segments > 0:
            log.warning(f"⚠️ {failed_segments} segments failed with {tts_provider} - NO FALLBACK USED")
        
        if success_segments == 0:
            log.error(f"❌ ALL TTS FAILED with {tts_provider} - Video cannot be created")
            return {}

        # Generate subtitles for successful segments only
        if processed_segments:
            log.info(f"🎬 Generating subtitles for {len(processed_segments)} successful segments...")
            self._transcribe_audio_segments_reliably(list(processed_segments.values()))
        
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

    def _construct_consistent_ssml(
        self, script_text: str, emotion: str, persona: Dict[str, Any]
    ) -> str:
        """Create SSML with consistent speed - no dynamic rate changes."""
        prosody_map = persona.get("speech_settings", {}).get("emotion_prosody_map", {})
        prosody_attrs = prosody_map.get(emotion, {})
        
        # CONSISTENT SPEED: Use default rate for all segments
        default_rate = persona.get("speech_settings", {}).get("default_rate", "medium")
        pitch = prosody_attrs.get("pitch", "medium")
        
        # Build SSML with consistent rate
        ssml_body = f'<prosody rate="{default_rate}" pitch="{pitch}">{html.escape(script_text)}</prosody>'
        return f"<speak>{ssml_body}</speak>"

    async def _generate_single_tts_segment_with_retries(
        self, scene_data: Dict, output_base: str, tts_provider: str, persona: Dict[str, Any]
    ) -> Optional[str]:
        """Generate TTS with 3 retries - NEVER switches providers, fails if provider doesn't work."""
        narration_text = scene_data.get("narration")
        if not narration_text:
            return None
        
        scene_id = scene_data.get("id", "unknown")
        max_retries = 3
        
        log.debug(f"🎙️ Generating TTS for {scene_id} using ONLY {tts_provider} (max {max_retries} attempts)")
        
        for attempt in range(max_retries):
            try:
                if tts_provider == "speechify":
                    if not self.speechify_client:
                        log.error(f"❌ {scene_id}: Speechify chosen but client not available - FAILING")
                        return None
                    
                    output_filename = f"{output_base}_speechify.wav"
                    
                    # Use consistent SSML with stable speed
                    ssml_input = self._construct_consistent_ssml(
                        narration_text, scene_data.get("emotion", "neutral"), persona
                    )
                    
                    success = await asyncio.to_thread(
                        self._speechify_tts_sync_with_retries, 
                        ssml_input, 
                        SPEECHIFY_DEFAULT_VOICE_ID, 
                        output_filename,
                        attempt + 1
                    )
                    
                    if success:
                        log.debug(f"✅ {scene_id}: Speechify TTS succeeded on attempt {attempt + 1}")
                        return output_filename
                    else:
                        log.warning(f"⚠️ {scene_id}: Speechify attempt {attempt + 1} failed")
                
                elif tts_provider == "openai":
                    output_filename = f"{output_base}_openai.mp3"
                    
                    success = await asyncio.to_thread(
                        self._openai_tts_sync_with_retries,
                        narration_text,
                        output_filename,
                        attempt + 1
                    )
                    
                    if success:
                        log.debug(f"✅ {scene_id}: OpenAI TTS succeeded on attempt {attempt + 1}")
                        return output_filename
                    else:
                        log.warning(f"⚠️ {scene_id}: OpenAI attempt {attempt + 1} failed")
                else:
                    log.error(f"❌ {scene_id}: Unknown TTS provider '{tts_provider}' - FAILING")
                    return None
                
                # Wait before retry (exponential backoff)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    log.debug(f"⏳ {scene_id}: Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    
            except Exception as e:
                log.error(f"❌ {scene_id}: TTS attempt {attempt + 1} error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        # All retries failed - NO FALLBACK TO OTHER PROVIDER
        log.error(f"❌ {scene_id}: All {max_retries} attempts FAILED with {tts_provider} - NO FALLBACK, SEGMENT FAILED")
        return None

    def _speechify_tts_sync_with_retries(
        self, ssml: str, voice_id: str, output_path: str, attempt_num: int
    ) -> bool:
        """Speechify TTS with detailed error logging."""
        try:
            log.debug(f"📡 Speechify API call attempt {attempt_num}")
            response = self.speechify_client.tts.audio.speech(
                input=ssml, voice_id=voice_id, audio_format="wav"
            )
            
            if response and response.audio_data:
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(response.audio_data))
                
                # Verify file was created and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                    log.debug(f"✅ Speechify file created: {os.path.getsize(output_path)} bytes")
                    return True
                else:
                    log.warning(f"⚠️ Speechify created invalid file (size: {os.path.getsize(output_path) if os.path.exists(output_path) else 0})")
                    return False
            else:
                log.warning(f"⚠️ Speechify returned empty response")
                return False
                
        except Exception as e:
            log.warning(f"⚠️ Speechify API error on attempt {attempt_num}: {e}")
            return False

    def _openai_tts_sync_with_retries(
        self, text: str, output_path: str, attempt_num: int
    ) -> bool:
        """OpenAI TTS with detailed error logging."""
        try:
            log.debug(f"📡 OpenAI TTS API call attempt {attempt_num}")
            response = self.openai_client.audio.speech.create(
                model=OPENAI_TTS_MODEL,
                voice=OPENAI_TTS_VOICE,
                input=text,
                response_format="mp3",
            )
            
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            # Verify file was created and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                log.debug(f"✅ OpenAI file created: {os.path.getsize(output_path)} bytes")
                return True
            else:
                log.warning(f"⚠️ OpenAI created invalid file (size: {os.path.getsize(output_path) if os.path.exists(output_path) else 0})")
                return False
                
        except Exception as e:
            log.warning(f"⚠️ OpenAI API error on attempt {attempt_num}: {e}")
            return False

    def _transcribe_audio_segments_reliably(self, audio_segments: List[Dict]):
        """
        Transcribe audio segments with reliable subtitle generation.
        
        This method ensures every segment gets word timings for subtitles,
        with smart memory management and error recovery.
        """
        if not audio_segments:
            log.warning("No audio segments to transcribe")
            return

        # Load ASR model
        model = self._get_or_load_asr_model()
        if not model:
            log.error("❌ Cannot generate subtitles - Whisper model unavailable")
            # Set empty timings to prevent crashes
            for seg in audio_segments:
                seg["asr_word_timings"] = []
            return

        log.info(f"🎙️ Transcribing {len(audio_segments)} audio files for subtitles...")
        
        successful_transcriptions = 0
        failed_transcriptions = 0
        
        for i, seg in enumerate(audio_segments):
            filepath = seg.get("filepath")
            scene_id = seg.get("id", f"segment_{i}")
            
            if not filepath or not os.path.exists(filepath):
                log.warning(f"⚠️ Audio file missing for {scene_id} - no subtitles for this segment")
                seg["asr_word_timings"] = []
                failed_transcriptions += 1
                continue
            
            try:
                log.debug(f"Transcribing {scene_id}: {os.path.basename(filepath)}")
                
                # Transcribe with optimized settings for subtitle generation
                result = model.transcribe(
                    filepath,
                    word_timestamps=True,
                    fp16=False,  # More stable on CPU
                    language='en',  # Specify language for better performance
                    condition_on_previous_text=False  # Each segment independent
                )
                
                # Extract word-level timings
                word_timings = []
                for segment in result.get("segments", []):
                    for word in segment.get("words", []):
                        if all(key in word for key in ["word", "start", "end"]):
                            # Clean and validate word data
                            word_data = {
                                "word": str(word["word"]).strip(),
                                "start": float(word["start"]),
                                "end": float(word["end"])
                            }
                            # Basic validation
                            if (word_data["word"] and 
                                0 <= word_data["start"] < word_data["end"] <= 30.0):  # Reasonable bounds
                                word_timings.append(word_data)
                
                seg["asr_word_timings"] = word_timings
                successful_transcriptions += 1
                
                if word_timings:
                    log.debug(f"✅ {scene_id}: Generated {len(word_timings)} subtitle words")
                else:
                    log.warning(f"⚠️ {scene_id}: Transcription succeeded but no word timings extracted")
                
                # Periodic memory cleanup during long transcription sessions
                if i % 3 == 0:  # Every 3 transcriptions
                    gc.collect()
                    
            except Exception as e:
                log.error(f"❌ Transcription failed for {scene_id}: {e}")
                seg["asr_word_timings"] = []
                failed_transcriptions += 1
                continue
        
        # Final transcription summary
        total_segments = len(audio_segments)
        log.info(f"🎬 Subtitle generation complete: {successful_transcriptions}/{total_segments} successful")
        
        if failed_transcriptions > 0:
            log.warning(f"⚠️ {failed_transcriptions} segments have no subtitles")
        
        if successful_transcriptions == 0:
            log.error("❌ No subtitles generated - all transcriptions failed")
        
        # Keep the model loaded for potential future use in the same session
        # It will be cleaned up when the service is destroyed