# src/config.py
"""
Configuration module for the AI Video Generator.

This file centralizes all settings, API keys from the .env file,
and style configurations to make the project easy to manage.
"""
import os
from dotenv import load_dotenv

# Load environment variables from a .env file at the project root
load_dotenv()

# --- Directory Settings ---
OUTPUT_DIR = "generated_videos"
TEMP_ASSETS_DIR = "temp_video_assets"
PLANS_DIR = "video_plans"

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SPEECHIFY_API_KEY = os.getenv("SPEECHIFY_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")

# --- Video Settings ---
VIDEO_DIMS = (1080, 1920)
FPS = 24
MIN_CLIP_DURATION = 2.0
MAX_STOCK_VIDEO_DURATION = 12.0
VIDEO_TRANSITION_DURATION = 0.3

# --- Text-to-Speech (TTS) Settings ---
SPEECHIFY_DEFAULT_VOICE_ID = os.getenv("SPEECHIFY_DEFAULT_VOICE_ID", "Matthew")
OPENAI_TTS_MODEL = "tts-1-hd"
OPENAI_TTS_VOICE = "shimmer"

# --- Caption Style Settings ---
CAPTION_STYLE = {
    "font_path": os.getenv("CAPTIONS_FONT_PATH", "Arial-Bold"),
    "font_size": 95,
    "color": "white",
    "accent_color": "#FFFF00",
    "stroke_color": "black",
    "stroke_width": 5,
}