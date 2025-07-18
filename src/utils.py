"""
Generic, low-level utility functions used across the project.
"""
import os
import re
import shutil

from PIL import Image as PILImage


# --- Filesystem Utilities ---
def sanitize_filename(text: str, default_name: str = "file") -> str:
    """Cleans a string to be a valid filename."""
    if not text:
        text = default_name
    text = str(text).strip().replace(" ", "_")
    text = re.sub(r'(?u)[^-\w.]', '', text)
    return text[:100] if text else default_name

# --- Compatibility Helper ---
def setup_pillow_antialias():
    """
    Addresses Pillow version incompatibility with MoviePy. Newer Pillow versions
    (10.0.0+) rename ANTIALIAS to LANCZOS. This patches the old attribute back.
    """
    if not hasattr(PILImage, 'ANTIALIAS'):
        try:
            PILImage.ANTIALIAS = PILImage.Resampling.LANCZOS
        except AttributeError:
            # For even newer versions where Resampling might be removed
            pass