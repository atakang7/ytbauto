# ielts.py

import os
import json
import asyncio
import random
import time 
from dotenv import load_dotenv
from moviepy.editor import (
    AudioFileClip, CompositeAudioClip, CompositeVideoClip, 
    vfx, afx, concatenate_audioclips
)
from openai import OpenAI
import traceback

# --- Hume AI SDK ---
try:
    from hume import AsyncHumeClient
    # Based on Hume's TTS docs, this is the most likely correct path.
    # User to verify against their specific installed SDK version if issues arise.
    from hume.tts import PostedUtterance, PostedUtteranceVoiceWithName 
    HUME_SDK_AVAILABLE = True
except ImportError:
    HUME_SDK_AVAILABLE = False

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

import base64

# --- Import from utils.py ---
# This script REQUIRES utils.py to be in the same directory.
from utils import (
    print_status, print_error, print_warning, 
    sanitize_filename, cleanup_temp_assets, setup_pillow_antialias,
    create_processed_visual_clip,
    fetch_from_pexels,
    fetch_from_freesound,
    create_animated_karaoke_captions
)

# --- Configuration & Initialization ---
load_dotenv()
setup_pillow_antialias() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUME_API_KEY = os.getenv("HUME_API_KEY")
HUME_DEFAULT_VOICE_ID = os.getenv("HUME_TTS_VOICE_ID") 

PEXELS_API_KEY_ENV = os.getenv("PEXELS_API_KEY")
FREESOUND_API_KEY_ENV = os.getenv("FREESOUND_API_KEY")

OUTPUT_DIR = "generated_videos_outstanding" 
TEMP_ASSETS_DIR_MAIN = "temp_video_assets_outstanding"
PLANS_DIR = "video_plans_outstanding"

VIDEO_DIMENSIONS_CONFIG = (1080, 1920) 
TARGET_FPS_CONFIG = 24
MIN_CLIP_DURATION_CONFIG = 2.0 
MAX_STOCK_VIDEO_DURATION_SECONDS_CONFIG = 12 
VIDEO_TRANSITION_DURATION_CONFIG = 0.3 

CAPTION_STYLE_CONFIG = {
    "font_path": os.getenv("CAPTIONS_FONT_PATH", "Komika-Axis.ttf"), # User should provide a bold, impactful font
    "font_size": 90,
    "color": "white",
    "accent_color": "#FFFF00", # Yellow for keywords
    "stroke_color": "black",
    "stroke_width": 5.0
}

OPENAI_TTS_VOICE_MODEL_CONFIG = "tts-1-hd" 
OPENAI_TTS_VOICE_PERSON_CONFIG = "shimmer" 
GPU_ACCELERATED_CODEC_CONFIG = os.getenv("GPU_CODEC")

if not OPENAI_API_KEY:
    print_error("OPENAI_API_KEY not found. Required for planning and fallback TTS.")
    exit()
openai_client = OpenAI(api_key=OPENAI_API_KEY)

if HUME_SDK_AVAILABLE: print_status("Hume SDK detected.")
else: print_warning("Hume SDK not found. Hume TTS will fallback to OpenAI.")
if AIOFILES_AVAILABLE: print_status("aiofiles library detected.")
elif HUME_SDK_AVAILABLE and HUME_API_KEY: print_warning("'aiofiles' not found. Hume TTS saving will fail. Install 'pip install aiofiles'.")

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_ASSETS_DIR_MAIN, exist_ok=True)
    os.makedirs(PLANS_DIR, exist_ok=True)
except OSError as e:
    print_error(f"Error creating directories: {e}"); exit()

# --- AI Content Planning: The Creator & The Critic ---

def plan_video_content_draft(user_prompt: str) -> dict | None:
    """
    Step 1: The 'Creator' AI generates a first draft of the video plan.
    """
    print_status("Planning video content with OpenAI (Creator Draft)...")
    system_message = """
    You are an AI Video Production Strategist. Your goal is to quickly generate a functional first draft of a JSON video plan based on a user's prompt.
    The plan should include a hook, 2-3 main sections, and a CTA. Focus on creating a logical structure and filling all required fields.
    This draft will be reviewed and improved by a senior creative director AI.
    Output a valid JSON object ONLY. The structure must match the final JSON structure for video generation.
    Ensure 'duration_estimate_seconds' is a number.
    """
    # Using the full detailed prompt structure for the first draft ensures the Critic has all fields to work with.
    full_prompt_structure_for_draft = """
    You are an AI Video Production Strategist, modeling the creative output of top-tier, viral short-form video creators.
    Your goal is to transform a user's idea into a comprehensive, actionable JSON plan for a "fire solid," highly attractive, and emotionally engaging video.
    The video MUST feel authentically human-created, incredibly dynamic, and provide clear, memorable value with a strong narrative drive from the very first second.
    Output a valid JSON object ONLY. No explanations before or after the JSON.

    JSON Structure:
    {
      "video_title": "Viral-worthy, ultra-clickable title (max 60 chars, SEO-friendly, high intrigue, hints at extraordinary value/stakes).",
      "target_audience_persona": "Deeply empathetic profile of the target audience: their core desires, unvoiced frustrations, what makes them INSTANTLY scroll away, and what content makes them feel immediately captivated, understood, and excited.",
      "overall_tone_emotion": "Describe the primary emotional journey FOR THE VIEWER (e.g., 'from sudden shock/disbelief to fascinated curiosity and an 'aha!' moment of empowerment', 'from relatable daily grind to hilarious escapism and shared joy', 'intense intrigue building to a satisfying, unexpected payoff'). This MUST dictate voice, music, pacing, and visual style with precision.",
      "hook_strategy": {
        "hook_type": "visual_first_mystery_audio_sting | action_packed_cold_open | high_emotion_narration_burst | provocative_question_text_overlay | pattern_interrupt_unexpected_sound",
        "duration_seconds": 2.5,
        "visual_sequence_description": "Hyper-detailed description of the visuals for these first seconds. Think dynamic camera work, rapid cuts, compelling imagery. E.g., 'Extreme close-up on an eye snapping open, quick cut to a mysterious object, rapid zoom out to reveal a surprising scene. High contrast, moody lighting.' OR 'Fast-paced montage (3-4 quick shots) of [action related to topic] building intensity.'",
        "stock_media_search_query_hook": "Specific Pexels query for the hook's visuals (e.g., 'eye opening extreme close up dark mysterious', 'fast car chase night drone view', 'abstract glitch intro pattern')",
        "narration_script_hook": "IF narration is used in hook (can be empty for visual-first), EXTREMELY short (2-6 words). Must be impactful. E.g., 'You won't believe this.', 'It all changed when...', 'Stop scrolling if...'",
        "hume_acting_prompt_hook": "IF narration_script_hook exists: Ultra-specific Hume acting instruction for these few words. E.g., 'Whispered with intense urgency and a hint of disbelief.', 'Shouted with explosive excitement!', 'Delivered with a deadpan, ironic tone.'",
        "hume_speed_hook": 1.2,
        "hume_trailing_silence_hook_sec": 0.1,
        "sfx_hook": "Crucial SFX for the hook. E.g., 'dramatic_riser_and_impact', 'short_glitch_sfx_then_whoosh'",
        "music_cue_hook": "How music starts or changes for the hook. E.g., 'Silence then sudden BLAST of energetic track', 'Intriguing synth pulse starts immediately', 'Music swells dramatically with visual reveal.'"
      },
      "caption_style_suggestion": "Suggest a caption style inspired by top engaging creators (e.g., 'MrBeast_energetic_yellow_bold_black_thick_outline_subtle_pop_animation', 'Modern_clean_white_heavy_sans_serif_drop_shadow_gentle_fade', 'Kinetic_text_color_changing_per_word_playful_font'). Include font type idea, main color, outline/shadow, and simple animation hint.",
      "sections": [
        {
          "section_number": 1,
          "section_title": "Very short, impactful theme (2-3 words) that logically follows the hook.",
          "duration_estimate_seconds": 4.0,
          "narrative_script": "Voice-over script. High energy or deeply engaging, human, and direct. Extremely natural, conversational language. Highly concise (8-18 words for shorts).",
          "keywords_for_highlighting": ["key_word1", "key_word2"],
          "hume_acting_prompt": "ULTRA-specific acting instruction for Hume AI for THIS segment. (e.g., 'Speak with rapidly building excitement, almost breathless, emphasizing the keyword with a slight upward pitch and a knowing pause right after')",
          "hume_speed": 1.0,
          "hume_trailing_silence_sec": 0.0,
          "visual_concept_description": "Vivid, high-impact visual description for this section. Focus on dynamic action, strong emotional cues, or crystal-clear illustration.",
          "stock_media_search_query": "Highly specific Pexels query for visually STUNNING footage. (e.g., 'cinematic time-lapse bustling city night high_angle')",
          "sfx_cues": [ ], "timed_text_overlays": [ ], "tool_suggestions": [ ]
        }
      ],
      "call_to_action_script": "Craft a CTA that is NOT generic. (Max 10 words). E.g., 'Think you know? Comment your guess!'",
      "hume_cta_acting_prompt": "Acting instruction for CTA: (e.g., 'Speak with infectious URGENCY and excitement')",
      "hume_cta_speed": 1.15,
      "background_music_suggestion": "Specific mood, genre, and energy evolution. (e.g., 'High-energy electronic trap beat with heavy bass, constant build-up, drops synchronized with key visual moments')"
    }
    """
    for attempt in range(3):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": full_prompt_structure_for_draft}, {"role": "user", "content": f"User's Core Idea: \"{user_prompt}\""}],
                temperature=0.7, response_format={"type": "json_object"} )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print_error(f"OpenAI Draft Planner attempt {attempt + 1} FAILED: {e}")
            if attempt < 2: time.sleep(random.uniform(3,5))
            else: print_error("Max retries for Draft Planner reached."); return None
    return None

def critique_and_refine_plan(draft_plan_json: str) -> dict | None:
    """
    Step 2: The 'Critic' AI reviews the draft plan and rewrites it for higher quality and impact.
    """
    print_status("Refining draft plan with AI Creative Director...")
    system_message = """
    You are a world-class short-form video producer and content strategist, known for turning good ideas into viral, highly engaging masterpieces.
    You will be given a DRAFT JSON video plan. Your task is to analyze it with a critical eye and then REWRITE the ENTIRE JSON object to make it 10x better.

    Your refinement MUST address these points:
    1.  **Narrative Depth & Length:** The draft might be too short or shallow. Flesh out the story. If needed, add one more 'section' to better develop the middle of the video. Ensure the total video length feels substantial, aiming for a 35-55 second sweet spot.
    2.  **Hook Authenticity:** Is the hook truly unskippable and non-generic? Rewrite it to be more creative, impactful, and authentic. Avoid cliches.
    3.  **Script Quality:** Are the `narrative_script` entries too robotic? Rewrite them to be more conversational, human, and emotionally compelling. Add rhetorical questions or more engaging phrasing.
    4.  **Hume AI Direction:** Are the `hume_acting_prompt`s nuanced enough? Make them ultra-specific, like a director giving precise instructions to a voice actor for rhythm, tone, and emotional delivery. Add or adjust `hume_speed` and `hume_trailing_silence_sec` to create a more dynamic vocal rhythm.
    5.  **Visual Creativity:** Are the `visual_concept_description` and `stock_media_search_query` generic? Make them more creative, specific, and emotionally aligned with the narration. Suggest dynamic visuals over static ones.
    6.  **CTA Impact:** Is the `call_to_action_script` weak? Rewrite it to be a compelling invitation or challenge, not a cheap plug.

    You MUST output the complete, REWRITTEN, and improved JSON object. Do not just list suggestions; provide the final, polished JSON plan.
    """
    for attempt in range(3):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini", # Using a capable model for critique is important
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Here is the draft JSON plan to critique and rewrite:\n\n{draft_plan_json}"}
                ],
                temperature=0.6, # Lower temp for more focused, high-quality refinement
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print_error(f"OpenAI Critic Planner attempt {attempt + 1} FAILED: {e}")
            if attempt < 2: time.sleep(random.uniform(4,8))
            else: print_error("Max retries for Critic Planner reached."); return None
    return None

# --- TTS, Media Fetching, Clip & Caption Creation (from utils.py) ---
# These functions are now assumed to be in utils.py and are imported at the top.

# --- Main Video Assembly Function ---
def create_video_from_plan_enhanced(video_plan_data: dict, segment_audio_data: list):
    # This function is now cleaner, focusing on assembly.
    # It assumes TTS is pre-generated and passed in segment_audio_data.
    # It calls utilities for fetching and creating clips.
    # (The full, robust content of this function from your last working script goes here)
    print_status("Assembling video from advanced plan...")
    if not video_plan_data: print_error("Video plan is empty."); return None
    
    total_narration_duration = sum(item['duration'] for item in segment_audio_data)
    print_status(f"Total narration duration from all segments: {total_narration_duration:.2f}s.")

    all_sub_clips, all_audio_mix_tracks, temp_closeables = [], [], []

    if segment_audio_data:
        try:
            narration_clips_to_concat = [AudioFileClip(p['filepath']) for p in segment_audio_data]
            temp_closeables.extend(narration_clips_to_concat)
            full_narration_clip = concatenate_audioclips(narration_clips_to_concat)
            temp_closeables.append(full_narration_clip)
            all_audio_mix_tracks.append(full_narration_clip)
            print_status(f"Main narration audio track created (Duration: {full_narration_clip.duration:.2f}s).")
        except Exception as e_concat:
            print_error(f"Could not concatenate narration segments: {e_concat}"); return None
    
    bg_music_fpath = fetch_from_freesound(
        video_plan_data["background_music_suggestion"], FREESOUND_API_KEY_ENV, TEMP_ASSETS_DIR_MAIN,
        total_narration_duration or 30 )
    if bg_music_fpath:
        try:
            bg_raw = AudioFileClip(bg_music_fpath); temp_closeables.append(bg_raw)
            target_bg_dur = total_narration_duration if total_narration_duration > 0.01 else bg_raw.duration
            bg_proc = bg_raw.fx(afx.volumex, 0.07).set_duration(target_bg_dur) 
            if bg_raw.duration < target_bg_dur and target_bg_dur > 0: bg_proc = bg_proc.fx(vfx.loop, duration=target_bg_dur)
            if bg_proc.duration > 0.1: all_audio_mix_tracks.append(bg_proc); print_status(f"BG music loaded ({bg_proc.duration:.2f}s).")
        except Exception as e: print_error(f"Could not load/process BG music {bg_music_fpath}: {e}")
    
    timeline_pos = 0.0
    
    all_content_segments = []
    hook_plan = video_plan_data.get("hook_strategy", {})
    if hook_plan:
        hook_plan['is_hook'] = True; all_content_segments.append(hook_plan)
    all_content_segments.extend(video_plan_data.get("sections", []))
    if video_plan_data.get("call_to_action_script"):
        all_content_segments.append({
            'is_cta': True, 'narrative_script': video_plan_data['call_to_action_script'],
            'keywords_for_highlighting': [],
            'stock_media_search_query': video_plan_data.get("overall_tone_emotion", "modern tech") + " conclusion"
        })
    
    audio_segment_index = 0
    for i, segment_data in enumerate(all_content_segments):
        is_hook = segment_data.get('is_hook', False)
        segment_id = "hook" if is_hook else segment_data.get('section_title', f"segment_{i}")
        print_status(f"Processing segment: '{segment_id}'")

        segment_narration_text = segment_data.get("narration_script_hook") if is_hook else segment_data.get("narrative_script")
        
        visual_duration = 0.0
        if segment_narration_text and audio_segment_index < len(segment_audio_data):
            visual_duration = segment_audio_data[audio_segment_index]['duration']
            audio_segment_index += 1
        else:
            visual_duration = float(segment_data.get("duration_seconds") or segment_data.get("duration_estimate_seconds") or MIN_CLIP_DURATION_CONFIG)
            if segment_narration_text: print_warning(f"Audio segment data out of sync for '{segment_id}'. Using estimated duration.")

        visual_duration = max(MIN_CLIP_DURATION_CONFIG, visual_duration)
        query = segment_data.get("stock_media_search_query_hook") if is_hook else segment_data.get("stock_media_search_query")
        
        media_info = fetch_from_pexels(query, segment_id, PEXELS_API_KEY_ENV, TEMP_ASSETS_DIR_MAIN)
        
        vis_clip = create_processed_visual_clip(
            media_info, visual_duration, VIDEO_DIMENSIONS_CONFIG, TARGET_FPS_CONFIG, MAX_STOCK_VIDEO_DURATION_SECONDS_CONFIG)
        if not vis_clip or vis_clip.duration <= 0.01:
            print_error(f"Visual clip failed for '{segment_id}'."); continue

        vis_clip_processed = vis_clip.set_start(timeline_pos).set_duration(vis_clip.duration)
        temp_closeables.append(vis_clip_processed)
        if i > 0: vis_clip_processed = vis_clip_processed.fx(vfx.fadein, VIDEO_TRANSITION_DURATION_CONFIG)
        all_sub_clips.append(vis_clip_processed)

        if segment_narration_text:
            karaoke_clips = create_animated_karaoke_captions(
                script=segment_narration_text, audio_duration=vis_clip_processed.duration, 
                timeline_start_time=timeline_pos, video_width=VIDEO_DIMENSIONS_CONFIG[0],
                keywords=segment_data.get("keywords_for_highlighting", []),
                style_config=CAPTION_STYLE_CONFIG
            )
            if karaoke_clips:
                all_sub_clips.extend(karaoke_clips); temp_closeables.extend(karaoke_clips)
        
        timeline_pos += vis_clip_processed.duration
    
    if not all_sub_clips: print_error("No visual clips processed. Aborting."); return None
    
    final_vid_duration = max(timeline_pos, total_narration_duration)
    if final_vid_duration <= 0.1: print_warning(f"Final video duration is very short: {final_vid_duration:.2f}s"); return None
    
    print_status(f"Final Compositing. Target duration: {final_vid_duration:.2f}s. Sub-clips: {len(all_sub_clips)}")
    final_composite = CompositeVideoClip(all_sub_clips, size=VIDEO_DIMENSIONS_CONFIG).set_duration(final_vid_duration)
    temp_closeables.append(final_composite)
    if final_composite.duration > VIDEO_TRANSITION_DURATION_CONFIG:
        final_composite = final_composite.fx(vfx.fadeout, VIDEO_TRANSITION_DURATION_CONFIG)

    if all_audio_mix_tracks:
        try:
            final_mix = CompositeAudioClip(all_audio_mix_tracks).set_duration(final_composite.duration)
            temp_closeables.append(final_mix)
            final_composite = final_composite.set_audio(final_mix)
            if final_composite.audio: print_status(f"Final audio mix applied ({final_composite.audio.duration:.2f}s)")
        except Exception as e: print_error(f"Audio composition error: {e}."); final_composite = final_composite.without_audio()
    else: print_warning("No audio tracks. Video will be silent."); final_composite = final_composite.without_audio()
    
    out_fpath = os.path.join(OUTPUT_DIR, sanitize_filename(video_plan_data.get("video_title", "generated_video")) + ".mp4")
    codec = GPU_ACCELERATED_CODEC_CONFIG or "libx264"
    try:
        print_status(f"Writing video: {out_fpath} (Codec: {codec}, Dur: {final_composite.duration:.2f}s)...")
        final_composite.write_videofile(
            out_fpath, codec=codec, audio_codec="aac", fps=TARGET_FPS_CONFIG,
            threads=os.cpu_count() if codec=="libx264" else None, preset="fast",
            logger='bar' if final_composite.duration > 1 else None, bitrate="6500k")
        print_status(f"Video created: {out_fpath}")
        return out_fpath
    except Exception as e_wr:
        print_error(f"Write error with {codec}: {e_wr}\n{traceback.format_exc()}")
        if GPU_ACCELERATED_CODEC_CONFIG and codec != "libx264":
            print_warning("Attempting CPU fallback (libx264)...")
            try:
                final_composite.write_videofile(out_fpath, codec="libx264", audio_codec="aac", fps=TARGET_FPS_CONFIG, threads=os.cpu_count(), preset="medium", logger='bar', bitrate="6000k")
                print_status(f"Video created with CPU fallback: {out_fpath}"); return out_fpath
            except Exception as e_cpu: print_error(f"CPU fallback failed: {e_cpu}\n{traceback.format_exc()}")
    finally: 
        print_status(f"Closing {len(temp_closeables)} MoviePy objects...")
        for obj in reversed(temp_closeables):
            try: 
                if obj and hasattr(obj, 'close') and callable(obj.close): obj.close()
            except: pass

# --- Text-to-Speech Management ---
async def generate_tts_audio_segment(
    text_to_speak: str,
    output_filepath_base: str, # Base path, e.g., "temp_dir/tts_segment_1"
    tts_preference: str = "hume",
    hume_client: 'AsyncHumeClient' = None,
    hume_acting_prompt: str = None,
    hume_voice_name_or_id: str = None,
    hume_speed: float = None, 
    hume_trailing_silence: float = None
    ) -> tuple[str | None, float]:
    """
    Generates a single audio segment using the preferred TTS service (Hume or OpenAI fallback).
    Hume generates .wav by default; the fallback generates .mp3.
    Returns the actual path to the generated audio file and its duration.
    """
    if not text_to_speak.strip():
        print_warning(f"TTS input (base: {output_filepath_base}) is empty.")
        return None, 0.0

    output_filename_hume_wav = f"{output_filepath_base}_hume.wav"
    output_filename_openai_mp3 = f"{output_filepath_base}_openai.mp3"

    # --- Attempt Hume AI TTS ---
    if tts_preference == "hume" and HUME_SDK_AVAILABLE and AIOFILES_AVAILABLE and HUME_API_KEY and hume_client:
        effective_hume_voice = hume_voice_name_or_id or HUME_DEFAULT_VOICE_ID
        
        print_status(f"Attempting Hume TTS for: '{text_to_speak[:40]}...'")
        if effective_hume_voice: print_status(f"  Hume Voice: '{effective_hume_voice}'")
        else: print_status("  Hume Voice: Default/Inferred by Hume")
        print_status(f"  Hume Acting Prompt: '{hume_acting_prompt or 'None'}'")
        if hume_speed is not None: print_status(f"  Hume Speed: {hume_speed}")
        if hume_trailing_silence is not None: print_status(f"  Hume Trailing Silence: {hume_trailing_silence}s")
        
        try:
            # This is where you would have your uncommented, live Hume SDK calls.
            # Building the configuration for the Hume API call:
            params_for_utterance = {"text": text_to_speak}
            if hume_acting_prompt and hume_acting_prompt.lower() not in ['neutral delivery', ''] and hume_acting_prompt.strip():
                params_for_utterance["description"] = hume_acting_prompt
            if effective_hume_voice:
                params_for_utterance["voice"] = PostedUtteranceVoiceWithName(name=effective_hume_voice)
            if hume_speed is not None:
                params_for_utterance["speed"] = hume_speed
            if hume_trailing_silence is not None and hume_trailing_silence > 0:
                params_for_utterance["trailing_silence"] = hume_trailing_silence
            
            utterance_config = PostedUtterance(**params_for_utterance)
            
            print_status("  Calling hume.tts.synthesize_json...")
            
            speech_result = await hume_client.tts.synthesize_json(
                utterances=[utterance_config]
            )
            print_status("  Hume API call returned.")

            if speech_result and hasattr(speech_result, 'generations') and speech_result.generations and \
               hasattr(speech_result.generations[0], 'audio') and speech_result.generations[0].audio:
                
                base64_encoded_audio = speech_result.generations[0].audio
                audio_data = base64.b64decode(base64_encoded_audio)
                
                if not AIOFILES_AVAILABLE:
                    raise ImportError("aiofiles library not found, required for saving Hume TTS output.")

                async with aiofiles.open(output_filename_hume_wav, "wb") as f:
                    await f.write(audio_data)
                
                if os.path.exists(output_filename_hume_wav) and os.path.getsize(output_filename_hume_wav) > 100:
                    with AudioFileClip(output_filename_hume_wav) as clip: duration = clip.duration
                    if duration and duration > 0.1:
                        print_status(f"Hume TTS segment successfully generated: {output_filename_hume_wav} (Duration: {duration:.2f}s)")
                        return output_filename_hume_wav, duration
                    print_error(f"Hume TTS .wav file '{output_filename_hume_wav}' invalid duration after saving.")
                else:
                    print_error(f"Hume TTS .wav file '{output_filename_hume_wav}' empty or not found after saving.")
            else:
                error_detail = "No audio data in response or generation failed." 
                if speech_result:
                    for attr_name in ['error_message', 'error', 'detail', 'message', 'id']:
                        if hasattr(speech_result, attr_name) and getattr(speech_result, attr_name):
                            error_detail = f"{attr_name}: {str(getattr(speech_result, attr_name))}"; break 
                print_error(f"Hume TTS API did not return valid audio. Details: {error_detail}")
            
            # This line will only be hit if the above block is still commented out.
            # Delete or comment out the line below once you've uncommented the actual Hume calls:
            raise NotImplementedError("Actual Hume TTS call is commented out. Please uncomment the block above.")

        except NotImplementedError as nie: 
             print_error(f"Hume TTS: {nie}") 
             print_warning("Falling back to OpenAI TTS (Hume part not live or placeholder).")
        except Exception as e_hume:
            print_error(f"Error during Hume TTS attempt: {e_hume}\n{traceback.format_exc()}")
            print_warning("Falling back to OpenAI TTS for this segment.")
        # Clean up failed Hume attempt before falling back
        if os.path.exists(output_filename_hume_wav): os.remove(output_filename_hume_wav)


    # --- Fallback to OpenAI TTS ---
    output_filename_to_use = output_filename_openai_mp3
    if tts_preference == "hume": # Message indicates it's a fallback
        print_status(f"Falling back to OpenAI TTS for: '{text_to_speak[:40]}...'")
    else: # OpenAI was the preference from the start, or Hume not configured
        print_status(f"Using OpenAI TTS for: '{text_to_speak[:40]}...' (Voice: {OPENAI_TTS_VOICE_PERSON_CONFIG})")
    
    try:
        response_openai = openai_client.audio.speech.create(
            model=OPENAI_TTS_VOICE_MODEL_CONFIG, voice=OPENAI_TTS_VOICE_PERSON_CONFIG,
            input=text_to_speak, response_format="mp3" )
        response_openai.stream_to_file(output_filename_to_use)
        if os.path.exists(output_filename_to_use) and os.path.getsize(output_filename_to_use) > 100:
            with AudioFileClip(output_filename_to_use) as clip: duration_openai = clip.duration
            if duration_openai and duration_openai > 0.1:
                print_status(f"OpenAI TTS segment saved: {output_filename_to_use} (Duration: {duration_openai:.2f}s)")
                return output_filename_to_use, duration_openai
            print_error(f"OpenAI TTS file '{output_filename_to_use}' invalid duration.")
        else: print_error(f"OpenAI TTS generation failed or produced empty file: {output_filename_to_use}")
    except Exception as e_openai:
        print_error(f"Failed to generate OpenAI TTS: {e_openai}\n{traceback.format_exc()}")
    
    if os.path.exists(output_filename_to_use): os.remove(output_filename_to_use)
    return None, 0.0

async def main_async_logic():
    cleanup_temp_assets(TEMP_ASSETS_DIR_MAIN) 
    user_prompt = input("Enter video idea: ").strip()
    if not user_prompt: print_error("No prompt. Exiting."); return

    # --- New Two-Step Planning Process ---
    plan_fname_base = sanitize_filename(user_prompt[:40], "video_plan")
    
    # Step 1: Creator generates the draft
    draft_plan = plan_video_content_draft(user_prompt)
    if not draft_plan: print_error("Initial draft plan generation failed. Exiting."); return
    
    draft_plan_path = os.path.join(PLANS_DIR, f"{plan_fname_base}_draft.json")
    with open(draft_plan_path, 'w', encoding='utf-8') as f:
        json.dump(draft_plan, f, indent=2, ensure_ascii=False)
    print_status(f"Draft plan saved to: {draft_plan_path}")

    # Step 2: Critic refines the draft
    final_plan = critique_and_refine_plan(json.dumps(draft_plan))
    if not final_plan: 
        print_warning("AI Critic failed to refine the plan. Proceeding with the original draft.")
        final_plan = draft_plan
    
    final_plan_path = os.path.join(PLANS_DIR, f"{plan_fname_base}_final.json")
    with open(final_plan_path, 'w', encoding='utf-8') as f:
        json.dump(final_plan, f, indent=2, ensure_ascii=False)
    print_status(f"Final, refined plan saved to: {final_plan_path}")
    print_status("Proceeding with video generation using the AI-refined plan.")

    # --- End New Two-Step Planning Process ---
    
    hume_cli = None; tts_pref = "openai"
    if HUME_API_KEY:
        if HUME_SDK_AVAILABLE and AIOFILES_AVAILABLE:
            try:
                hume_cli = AsyncHumeClient(api_key=HUME_API_KEY)
                print_status("Hume AI Client initialized. Will attempt Hume TTS first."); tts_pref = "hume"
            except Exception as e: print_error(f"Hume Client init failed: {e}. Falling back to OpenAI TTS."); tts_pref = "openai"
        else: print_warning("Hume API Key set but dependencies missing. Falling back to OpenAI TTS."); tts_pref = "openai"
    else: print_status("HUME_API_KEY not set. Using OpenAI TTS."); tts_pref = "openai"

    tts_segments_to_generate = []
    
    hook_s_data = final_plan.get("hook_strategy", {})
    hook_s_narr = hook_s_data.get("narration_script_hook", "")
    if hook_s_narr.strip():
        tts_segments_to_generate.append({
            "text":hook_s_narr, "hume_prompt":hook_s_data.get("hume_acting_prompt_hook",""), 
            "hume_speed": hook_s_data.get("hume_speed_hook"), 
            "hume_trailing_silence": hook_s_data.get("hume_trailing_silence_hook_sec"), "id":"hook" })

    for i_s, sec in enumerate(final_plan.get("sections",[])):
        tts_segments_to_generate.append({
            "text":sec.get("narrative_script",""), "hume_prompt":sec.get("hume_acting_prompt",""), 
            "hume_speed": sec.get("hume_speed"), "hume_trailing_silence": sec.get("hume_trailing_silence_sec"),
            "id":f"s{i_s+1}", "keywords": sec.get("keywords_for_highlighting", [])
            })
    
    cta_text = final_plan.get("call_to_action_script","")
    if cta_text.strip(): 
        tts_segments_to_generate.append({
            "text":cta_text, "hume_prompt":final_plan.get("hume_cta_acting_prompt",""), 
            "hume_speed": final_plan.get("hume_cta_speed"), "hume_trailing_silence": None, "id":"cta"})

    if not any(s["text"].strip() for s in tts_segments_to_generate): print_error("No narrative script in plan. Exiting."); return

    generated_audio_segments = []
    for seg_data in tts_segments_to_generate:
        if not seg_data["text"].strip(): continue
        tts_fbase = os.path.join(TEMP_ASSETS_DIR_MAIN, f"tts_{sanitize_filename(final_plan.get('video_title','vid_aud'))}_{seg_data['id']}")
        gen_fpath, seg_dur = await generate_tts_audio_segment(
            seg_data["text"], tts_fbase, tts_pref, hume_cli, seg_data["hume_prompt"], HUME_DEFAULT_VOICE_ID,
            seg_data["hume_speed"], seg_data["hume_trailing_silence"])
        if seg_dur > 0 and gen_fpath:
            generated_audio_segments.append({"filepath": gen_fpath, "duration": seg_dur, "script": seg_data["text"], "keywords": seg_data.get("keywords", [])})
        else: print_warning(f"TTS failed for segment '{seg_data['id']}'. It will be silent.")
    
    video_output_file = create_video_from_plan_enhanced(final_plan, generated_audio_segments)
    
    if video_output_file: print_status(f"===> Output Video Ready: {video_output_file} <===")
    else: print_error("Video generation ultimately FAILED.")

    cleanup = input("Clean up temporary files? (yes/no) [yes]: ").strip().lower()
    if cleanup not in ["no","n"]: cleanup_temp_assets(TEMP_ASSETS_DIR_MAIN)
    else: print_status(f"Temporary assets retained: {TEMP_ASSETS_DIR_MAIN}")
    print_status("Script execution finished.")

if __name__ == "__main__":
    try: asyncio.run(main_async_logic())
    except KeyboardInterrupt: print_status("\nProcess interrupted by user. Exiting gracefully.")
    except Exception as e_fatal:
        print_error(f"FATAL UNHANDLED EXCEPTION in main: {e_fatal}")
        traceback.print_exc()