# src/utils.py
"""
Generic, low-level utility functions used across the project.

This module contains helpers for logging, filesystem operations,
and other basic tasks that are not specific to any one domain.
"""
import os
import shutil
import re
from PIL import Image as PILImage

# --- Logging Helper Functions ---
def print_status(message: str):
    print(f"[INFO] {message}")

def print_error(message: str):
    print(f"[ERROR] {message}")

def print_warning(message: str):
    print(f"[WARNING] {message}")

# --- Filesystem Utilities ---
def sanitize_filename(text: str, default_name: str = "file") -> str:
    """Cleans a string to be a valid filename."""
    if not text: text = default_name
    text = str(text).strip().replace(" ", "_")
    text = re.sub(r'(?u)[^-\w.]', '', text)
    return text[:100] if text else default_name

def setup_project_directories(dirs_to_create: list[str], temp_dir_to_clean: str):
    """Ensures all necessary project directories exist and cleans the temp folder."""
    try:
        if os.path.exists(temp_dir_to_clean):
            shutil.rmtree(temp_dir_to_clean)
        
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
        print_status("Project directories ensured and temp assets cleaned.")
    except OSError as e:
        print_error(f"Error setting up project directories: {e}")
        raise

# --- Compatibility Helper ---
def setup_pillow_antialias():
    """
    Addresses Pillow version incompatibility with MoviePy. Newer Pillow versions
    (10.0.0+) rename ANTIALIAS to LANCZOS. This patches the old attribute back.
    """
    if not hasattr(PILImage, 'ANTIALIAS'):
        try:
            PILImage.ANTIALIAS = PILImage.Resampling.LANCZOS
            print_status("Compatibility patch for Pillow/MoviePy applied (ANTIALIAS).")
        except AttributeError:
            print_error("Could not apply Pillow compatibility patch. Video processing might fail.")