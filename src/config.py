"""
Configuration module for the AI Video Generator.

This is the single source of truth for all configuration variables.
"""
import os
from typing import Optional, Tuple
from dotenv import load_dotenv

# Load environment variables from a .env file at the project root
load_dotenv()

# --- Directory Settings ---
OUTPUT_DIR: str = "generated_videos"
TEMP_ASSETS_DIR: str = "temp_video_assets"
PLANS_DIR: str = "video_plans"

# --- API Keys ---
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
SPEECHIFY_API_KEY: Optional[str] = os.getenv("SPEECHIFY_API_KEY")
PEXELS_API_KEY: Optional[str] = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY: Optional[str] = os.getenv("PIXABAY_API_KEY")

# --- Video Settings ---
VIDEO_DIMS: Tuple[int, int] = (1080, 1920)
FPS: int = 30

# --- Text-to-Speech (TTS) Settings ---
SPEECHIFY_DEFAULT_VOICE_ID: str = os.getenv("SPEECHIFY_DEFAULT_VOICE_ID", "Matthew")
OPENAI_TTS_MODEL: str = "tts-1-hd"
OPENAI_TTS_VOICE: str = "shimmer"
MAX_SEGMENT_DURATION: float = 15.0

# --- Transformative Mode Settings ---
SCENE_DETECT_THRESHOLD: int = 27

# --- Network Settings ---
REQUEST_TIMEOUT: int = 45
MAX_RETRIES: int = 3