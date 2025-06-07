# src/main.py
"""
Main entry point and orchestrator for the AI Video Generation script.

This script ties together all the modules to perform the following steps:
1.  Load configuration and the brand persona.
2.  Get a video idea from the user.
3.  Use an AI to plan the video script and visual choices.
4.  Generate narration audio using a TTS service.
5.  Generate caption timings using ASR.
6.  Fetch all visual and audio assets.
7.  Assemble all elements into a final video file.
"""
import asyncio
import argparse
import json
import os
import traceback
from openai import OpenAI

# Import our own modules
import config
from utils import print_status, print_error, print_warning, sanitize_filename, setup_project_directories, setup_pillow_antialias
from planning import load_brand_persona, plan_video_content
from audio import generate_and_process_audio, transcribe_audio_segments
from visuals import create_processed_visual_clip, create_asr_synced_captions, fetch_from_pexels, fetch_music_from_pixabay

# MoviePy imports for the final assembly
from moviepy.editor import AudioFileClip, CompositeAudioClip, CompositeVideoClip, vfx, afx, concatenate_audioclips

def initialize_clients():
    """Initializes and returns API clients based on .env settings."""
    if not config.OPENAI_API_KEY:
        print_error("OPENAI_API_KEY not found in .env file. Exiting.")
        return None, None
    openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    speechify_client = None
    if config.SPEECHIFY_API_KEY:
        try:
            from speechify import Speechify
            speechify_client = Speechify(token=config.SPEECHIFY_API_KEY)
            print_status("Speechify SDK detected and client initialized.")
        except ImportError:
            print_warning("Speechify SDK not installed, but API key is present. Run 'pip install speechify-api'")
        except Exception as e:
            print_warning(f"Speechify client initialization failed: {e}")
            
    return openai_client, speechify_client

async def main_async_logic(persona_file: str):
    """Orchestrates the entire video creation process."""
    setup_pillow_antialias()
    setup_project_directories(
        dirs_to_create=[config.OUTPUT_DIR, config.TEMP_ASSETS_DIR, config.PLANS_DIR],
        temp_dir_to_clean=config.TEMP_ASSETS_DIR
    )

    # 1. Load Persona & Get Topic
    brand_persona = load_brand_persona(persona_file)
    if not brand_persona: exit()
    
    user_prompt = input("\nEnter video idea (or a detailed plan): ").strip()
    if not user_prompt: print_error("No prompt. Exiting."); return

    # 2. Initialize Clients
    openai_client, speechify_client = initialize_clients()
    if not openai_client: exit()
    
    # 3. AI Planning
    final_plan = plan_video_content(user_prompt, brand_persona, openai_client)
    if not final_plan: print_error("Video planning failed. Exiting."); return
    
    plan_fname = f"{sanitize_filename(final_plan.get('video_title', user_prompt[:40]))}.json"
    plan_path = os.path.join(config.PLANS_DIR, plan_fname)
    with open(plan_path, 'w', encoding='utf-8') as f: json.dump(final_plan, f, indent=4)
    print_status(f"Final plan saved to '{plan_path}'")

    # 4. Audio Generation & Processing
    tts_segments = []
    for sec in final_plan.get("sections", []):
        if sec.get("narrative_script"):
            tts_segments.append({"text": sec["narrative_script"], "id": sanitize_filename(sec.get("section_title")), "keywords": sec.get("keywords_for_highlighting", [])})
    if final_plan.get("call_to_action_script"):
        tts_segments.append({"text": final_plan["call_to_action_script"], "id": "cta", "keywords": []})
    
    tts_preference = "speechify" if speechify_client else "openai"
    processed_audio = await generate_and_process_audio(tts_segments, tts_preference, speechify_client, openai_client, final_plan.get('video_title'))
    transcribe_audio_segments(processed_audio)

    # 5. Video Assembly
    print_status("Beginning video assembly process...")
    all_clips = []
    timeline_pos = 0.0
    narration_clips = [AudioFileClip(s['filepath']) for s in processed_audio if s.get('filepath')]
    
    # Create master narration track and calculate total duration
    full_narration_clip = concatenate_audioclips(narration_clips) if narration_clips else None
    total_duration = full_narration_clip.duration if full_narration_clip else 0

    # Prepare background music
    music_clip = None
    music_path = fetch_music_from_pixabay(final_plan.get("background_music_suggestion"), config.PIXABAY_API_KEY, config.TEMP_ASSETS_DIR)
    if music_path:
        try:
            music_clip = AudioFileClip(music_path).fx(vfx.loop, duration=total_duration).fx(afx.audio_fadein, 1.0).fx(afx.volumex, 0.08)
            print_status("Background music loaded and processed.")
        except Exception as e:
            print_error(f"Could not process music: {e}")

    # Create visual and caption clips for each segment
    for seg_data in final_plan.get("sections", []) + [{"section_title": "CTA", "narrative_script": final_plan.get("call_to_action_script"), "stock_media_search_query": "abstract gradient background"}]:
        if not seg_data.get("narrative_script"): continue
        
        segment_id = sanitize_filename(seg_data.get('section_title'))
        audio_seg = next((s for s in processed_audio if s["id"] == segment_id), None)
        duration = audio_seg['duration'] if audio_seg else config.MIN_CLIP_DURATION
        
        media_info = fetch_from_pexels(seg_data.get("stock_media_search_query"), segment_id, config.PEXELS_API_KEY, config.TEMP_ASSETS_DIR)
        if media_info and media_info.get("path"):
            vis_clip = create_processed_visual_clip(media_info["path"], duration)
            if vis_clip:
                all_clips.append(vis_clip.set_start(timeline_pos))
                if audio_seg and audio_seg.get("asr_word_timings"):
                    captions = create_asr_synced_captions(audio_seg["asr_word_timings"], seg_data.get("keywords_for_highlighting", []))
                    for cap in captions:
                        all_clips.append(cap.set_start(cap.start + timeline_pos))
        
        timeline_pos += duration

    # 6. Final Composition and Render
    if not all_clips:
        print_error("No visual clips were created. Aborting."); return
        
    final_audio_tracks = [track for track in [full_narration_clip, music_clip] if track]
    final_audio = CompositeAudioClip(final_audio_tracks).set_duration(total_duration) if final_audio_tracks else None
    
    final_video = CompositeVideoClip(all_clips, size=config.VIDEO_DIMS).set_duration(total_duration).set_audio(final_audio)

    out_fpath = os.path.join(config.OUTPUT_DIR, sanitize_filename(final_plan.get("video_title")) + ".mp4")
    try:
        print_status(f"Writing final video to '{out_fpath}'...")
        final_video.write_videofile(
            out_fpath, codec="libx264", audio_codec="aac", fps=config.FPS,
            threads=os.cpu_count(), preset="fast", logger='bar')
        print_status(f"Video created: {out_fpath}")
    except Exception as e:
        print_error(f"Failed to write video file: {e}")
    finally:
        # Gracefully close all clips
        for clip in all_clips + final_audio_tracks:
            if hasattr(clip, 'close'): clip.close()
        print_status("Cleaned up video and audio clips.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-powered video creation script.")
    parser.add_argument(
        "-p", "--persona",
        default="brand_persona.json",
        help="Path to the brand persona JSON file (default: brand_persona.json)"
    )
    args = parser.parse_args()
    try:
        asyncio.run(main_async_logic(args.persona))
    except KeyboardInterrupt:
        print_status("\nProcess interrupted by user. Exiting gracefully.")
    except Exception as e_fatal:
        print_error(f"FATAL UNHANDLED EXCEPTION in main: {e_fatal}")
        traceback.print_exc()