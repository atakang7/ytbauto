# src/audio.py
"""
Manages audio generation (TTS), processing (trimming), and analysis (ASR).
"""
import asyncio
import re
import os
import base64
from moviepy.editor import AudioFileClip

# Import from our own modules
from utils import print_status, print_error, print_warning, sanitize_filename
import config

# Conditional Imports
try:
    from speechify import Speechify
except ImportError:
    Speechify = None
try:
    import whisper
except ImportError:
    whisper = None
try:
    import librosa
    import soundfile as sf
except ImportError:
    librosa = None
    sf = None

# --- Audio Processing Utility ---
def trim_audio_silence(audio_path: str, top_db=25):
    """Trims leading and trailing silence from an audio file for better sync."""
    if not librosa or not sf:
        print_warning("Librosa/Soundfile not installed. Skipping silence trimming.")
        return
        
    try:
        y, sr = librosa.load(audio_path, sr=None)
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        if len(y_trimmed) > 0:
            sf.write(audio_path, y_trimmed, sr)
            print_status(f"Trimmed silence from '{os.path.basename(audio_path)}'.")
        else:
            print_warning(f"Audio trimming resulted in an empty clip for '{audio_path}'.")
    except Exception as e:
        print_error(f"Could not trim silence from '{audio_path}': {e}")

# --- TTS Generation ---
def _speechify_tts_sync(client: Speechify, text: str, voice_id: str, output_path: str) -> float | None:
    """Synchronous helper for Speechify TTS call, handling SSML."""
    try:
        response = client.tts.audio.speech(input=text, voice_id=voice_id, audio_format="wav")
        if response and response.audio_data:
            audio_data_bytes = base64.b64decode(response.audio_data)
            with open(output_path, "wb") as f: f.write(audio_data_bytes)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                with AudioFileClip(output_path) as clip:
                    return clip.duration
    except Exception as e:
        error_body = e.body if hasattr(e, 'body') else ''
        print_error(f"Speechify TTS (SSML) error: {e}, body: {error_body}")
    return None

async def _generate_single_tts_segment(
    text_to_speak: str, output_filepath_base: str, tts_preference: str,
    speechify_client: Speechify | None, openai_client
) -> tuple[str | None, float]:
    """Generates a single audio segment."""
    if not text_to_speak.strip(): return None, 0.0

    if tts_preference == "speechify" and speechify_client:
        output_filename = f"{output_filepath_base}_speechify.wav"
        ssml_input = text_to_speak.strip()
        if not ssml_input.lower().startswith('<speak>'):
            ssml_input = f'<speak>{ssml_input}</speak>'
            
        print_status(f"Attempting Speechify SSML TTS for: '{ssml_input[:70]}...'")
        duration = await asyncio.to_thread(
            _speechify_tts_sync, speechify_client, ssml_input, config.SPEECHIFY_DEFAULT_VOICE_ID, output_filename
        )
        if duration:
            return output_filename, duration
        print_warning("Speechify SSML TTS failed. Falling back to OpenAI TTS.")

    output_filename = f"{output_filepath_base}_openai.mp3"
    clean_text = re.sub(r'<[^>]+>', '', text_to_speak).strip()
    if not clean_text:
        print_warning(f"Text empty after stripping SSML: '{text_to_speak}'"); return None, 0.0
        
    print_status(f"Using OpenAI TTS for: '{clean_text[:40]}...'")
    try:
        response = openai_client.audio.speech.create(
            model=config.OPENAI_TTS_MODEL, voice=config.OPENAI_TTS_VOICE,
            input=clean_text, response_format="mp3")
        response.stream_to_file(output_filename)
        if os.path.exists(output_filename) and os.path.getsize(output_filename) > 100:
            with AudioFileClip(output_filename) as clip: return output_filename, clip.duration
    except Exception as e:
        print_error(f"Failed to generate OpenAI TTS: {e}")
    return None, 0.0

async def generate_and_process_audio(tts_segments: list, tts_preference: str, speechify_client, openai_client, plan_title: str) -> list:
    """Generates all TTS audio, trims silence, and returns data for assembly."""
    base_filename = sanitize_filename(plan_title)
    coroutines = [_generate_single_tts_segment(
        s["text"], os.path.join(config.TEMP_ASSETS_DIR, f"tts_{base_filename}_{s['id']}"),
        tts_preference, speechify_client, openai_client
    ) for s in tts_segments]
    
    print_status(f"Generating {len(coroutines)} audio segments concurrently...")
    results = await asyncio.gather(*coroutines)

    processed_segments = []
    for seg_data, (fpath, dur) in zip(tts_segments, results):
        if fpath and dur > 0:
            trim_audio_silence(fpath)
            with AudioFileClip(fpath) as clip: final_duration = clip.duration
            processed_segments.append({**seg_data, "filepath": fpath, "duration": final_duration})
        else:
            print_warning(f"TTS failed for segment '{seg_data['id']}'. It will be silent.")
    return processed_segments

# --- ASR / Transcription ---
def transcribe_audio_segments(audio_segments: list) -> None:
    """Transcribes all audio segments in-place to add 'asr_word_timings'."""
    if not whisper:
        print_warning("Whisper not installed. Skipping transcription."); return
    
    try:
        asr_model = whisper.load_model("small", device="cpu")
        print_status("ASR model loaded.")
    except Exception as e:
        print_error(f"Could not load ASR model: {e}"); return

    print_status("Transcribing audio segments for caption synchronization...")
    for seg in audio_segments:
        if not seg.get("filepath"): continue
        try:
            print_status(f"  Transcribing {os.path.basename(seg['filepath'])}...")
            result = asr_model.transcribe(seg['filepath'], word_timestamps=True)
            words = [word for s in result.get('segments', []) for word in s.get('words', [])]
            if words:
                seg['asr_word_timings'] = words
                print_status(f"  Transcription successful. Found {len(words)} words.")
            else:
                seg['asr_word_timings'] = []
        except Exception as e:
            print_error(f"ASR transcription failed for {seg['filepath']}: {e}")
            seg['asr_word_timings'] = []