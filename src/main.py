"""
Main Orchestrator for the AI Video Generation Script

This script initializes the system, presents the user with creation modes, and
delegates the chosen workflow to the appropriate services.
"""
# --- Standard Library Imports ---
import asyncio
import argparse
import json
import logging
import os
import shutil
import sys
import traceback
from typing import Optional

# --- Third-Party Imports ---
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# --- Local Application Imports ---
import config
from services import (
    PlanningService, AudioService, MediaService,
    GenerativeAssemblyService, RemixAssemblyService, VideoAnalysisService
)
from models import VideoPlan, RemixPlan
from utils import sanitize_filename
from video_processing.editor import VideoEditor

# ======================================================================================
# --- 1. Core Setup: Logging and Dependency Checks ---
# ======================================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)
load_dotenv()

def check_dependencies():
    """
    Checks for critical libraries and external tools at startup and exits if not found.
    """
    log.info("Checking for critical dependencies...")
    try:
        import moviepy.editor
        import scenedetect
        import mutagen
        import whisper
        import cv2
        import numpy
        import tqdm
        log.info("All critical Python libraries are installed.")
    except ImportError as e:
        log.critical(f"Missing critical dependency: {e.name}.")
        log.critical("Please run 'poetry install' to install all required packages.")
        sys.exit(1)

    if not shutil.which("ffmpeg"):
        log.critical("FFmpeg not found in system PATH.")
        log.critical("FFmpeg is required by MoviePy for video processing.")
        log.critical("Please install FFmpeg (e.g., 'sudo apt install ffmpeg' or 'brew install ffmpeg') and ensure it is in your PATH.")
        sys.exit(1)
    
    log.info("External dependency 'ffmpeg' found.")


# ======================================================================================
# --- 2. Mode-Specific Workflows ---
# ======================================================================================

async def run_generative_mode(persona_file: str):
    log.info("🚀 Starting Generative Mode...")
    openai_client, speechify_client = initialize_clients()
    planner = PlanningService(openai_client)
    if not (brand_persona := planner.load_brand_persona(persona_file)): return
    if not (user_prompt := input("\nEnter video idea: ").strip()): return
    
    final_plan: Optional[VideoPlan] = await asyncio.to_thread(planner.create_generative_plan, user_prompt, brand_persona)
    if not final_plan: return

    plan_path = os.path.join(config.PLANS_DIR, f"{sanitize_filename(final_plan.video_title)}.json")
    with open(plan_path, 'w') as f: f.write(final_plan.json(indent=2))
    log.info(f"Plan saved to '{plan_path}'")
    
    editor = VideoEditor()
    audio_service = AudioService(openai_client, speechify_client)
    media_service = MediaService()
    assembly_service = GenerativeAssemblyService(editor, media_service, audio_service)

    processed_audio, media_assets = await asyncio.gather(
        audio_service.generate_and_process_audio(final_plan, brand_persona),
        media_service.get_assets_for_plan(final_plan)
    )
    
    await assembly_service.assemble_video(final_plan, processed_audio, media_assets)

async def run_transformative_mode(persona_file: str):
    log.info("🚀 Starting Transformative Mode...")
    try:
        openai_client, speechify_client = initialize_clients()
        planner = PlanningService(openai_client)
        
        video_path = input("\nEnter full path to source video: ").strip()
        if not video_path or not os.path.exists(video_path):
            log.error(f"File not found or path is empty: {video_path}"); return
        
        analysis_service = VideoAnalysisService(video_path, openai_client)
        analysis_service.get_video_properties()
        scenes = analysis_service.detect_scenes()
        tagged_scenes = await analysis_service.tag_scenes_with_vision(scenes)
        
        print("\n--- AI Scene Tagging Complete ---")
        for scene in tagged_scenes: print(f"  Scene {scene.scene_id}: Tags = {scene.tags}")
        
        if not (remix_query := input("\nEnter remix request (e.g., 'summarize this video'): ").strip()): return
        
        remix_plan: Optional[RemixPlan] = await planner.create_remix_plan(remix_query, tagged_scenes, video_path)
        if not remix_plan: return
        
        plan_path = os.path.join(config.PLANS_DIR, f"{sanitize_filename(remix_plan.remix_video_title)}.json")
        with open(plan_path, 'w') as f: f.write(remix_plan.json(indent=2))
        log.info(f"Plan saved to '{plan_path}'")
        
        editor = VideoEditor()
        media_service = MediaService()
        audio_service = AudioService(openai_client, speechify_client)
        remix_assembly_service = RemixAssemblyService(media_service, audio_service, editor)
        
        await remix_assembly_service.assemble_and_render_remix(remix_plan, tagged_scenes)
    except Exception as e:
        log.critical(f"Fatal error in Transformative Mode: {e}"); traceback.print_exc()

async def run_render_from_file_mode():
    log.info("🚀 Starting Render From Plan File Mode...")
    try:
        plan_path = input("\nEnter the full path to your plan JSON file: ").strip()
        if not plan_path or not os.path.exists(plan_path):
            log.error(f"Plan file not found: {plan_path}"); return
        
        with open(plan_path, 'r') as f:
            data = json.load(f)

        openai_client, speechify_client = initialize_clients()
        editor = VideoEditor()
        audio_service = AudioService(openai_client, speechify_client)
        media_service = MediaService()
        planner = PlanningService(openai_client)

        if 'video_title' in data and 'sections' in data:
            log.info("Detected a Generative Plan (VideoPlan).")
            final_plan = VideoPlan.parse_obj(data)
            brand_persona = planner.load_brand_persona("brand_persona.json")
            if not brand_persona: 
                log.error("Could not load default brand persona 'brand_persona.json'."); return

            assembly_service = GenerativeAssemblyService(editor, media_service, audio_service)
            processed_audio, media_assets = await asyncio.gather(
                audio_service.generate_and_process_audio(final_plan, brand_persona),
                media_service.get_assets_for_plan(final_plan)
            )
            await assembly_service.assemble_video(final_plan, processed_audio, media_assets)

        elif 'remix_video_title' in data and 'source_video_path' in data:
            log.info("Detected a Transformative Plan (RemixPlan).")
            remix_plan = RemixPlan.parse_obj(data)
            
            analysis_service = VideoAnalysisService(remix_plan.source_video_path, openai_client)
            scenes = analysis_service.detect_scenes()
            
            remix_assembly_service = RemixAssemblyService(media_service, audio_service, editor)
            await remix_assembly_service.assemble_and_render_remix(remix_plan, scenes)
        
        else:
            log.error("Could not determine plan type from the provided JSON file.")

    except Exception as e:
        log.critical(f"Fatal error during render from file: {e}"); traceback.print_exc()


# ======================================================================================
# --- 3. Main Orchestrator and Execution ---
# ======================================================================================

def setup_directories():
    log.info("Setting up project directories...")
    if os.path.exists(config.TEMP_ASSETS_DIR): shutil.rmtree(config.TEMP_ASSETS_DIR)
    dirs_to_create = [config.OUTPUT_DIR, config.TEMP_ASSETS_DIR, config.PLANS_DIR]
    for path in dirs_to_create:
        os.makedirs(path, exist_ok=True)

def initialize_clients() -> tuple[OpenAI, Optional['Speechify']]:
    if not (api_key := config.OPENAI_API_KEY): raise ValueError("OPENAI_API_KEY not found.")
    openai_client = OpenAI(api_key=api_key)
    speechify_client = None
    try:
        from speechify import Speechify
        if config.SPEECHIFY_API_KEY:
            speechify_client = Speechify(token=config.SPEECHIFY_API_KEY)
    except ImportError:
        log.info("Speechify SDK not installed, it will not be available.")
    except Exception as e:
        log.warning(f"Speechify client failed to initialize: {e}")
    return openai_client, speechify_client

async def main_orchestrator(persona_file: str):
    while True:
        print("\nSelect Mode: [1] Generative [2] Transformative [3] Render From File [q] Quit")
        choice = input("> ").strip().lower()
        if choice == '1': await run_generative_mode(persona_file); break
        elif choice == '2': await run_transformative_mode(persona_file); break
        elif choice == '3': await run_render_from_file_mode(); break
        elif choice in ['q', 'quit']: break
        else: log.warning("Invalid choice.")

if __name__ == "__main__":
    check_dependencies()
    
    parser = argparse.ArgumentParser(description="AI-powered video creation orchestrator.")
    parser.add_argument("-p", "--persona", default="brand_persona.json", help="Path to brand persona JSON file (relative to src).")
    args = parser.parse_args()
    
    try:
        setup_directories()
        script_dir = os.path.dirname(__file__)
        persona_path = os.path.join(script_dir, args.persona)
        asyncio.run(main_orchestrator(persona_path))
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("\nProcess interrupted.")
    except Exception as e:
        log.critical(f"A fatal unhandled exception occurred: {e}"); traceback.print_exc()