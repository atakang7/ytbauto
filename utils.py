# utils.py

import os
import shutil
import re
import random
import requests
from PIL import Image as PILImage, ImageFont, ImageDraw
import numpy as np
from moviepy.editor import (
    ImageClip, VideoFileClip, AudioFileClip, 
    TextClip, CompositeVideoClip, vfx, afx
)
from moviepy.video.fx.all import crop, resize
import traceback

# --- Logging Helper Functions ---
def print_status(message: str):
    """Prints an informational message."""
    print(f"[INFO] {message}")

def print_error(message: str):
    """Prints an error message."""
    print(f"[ERROR] {message}")

def print_warning(message: str):
    """Prints a warning message."""
    print(f"[WARNING] {message}")

# --- Filesystem Helper Functions ---
def sanitize_filename(text: str, default_name: str = "file") -> str:
    if not text: text = default_name
    text = str(text)
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    text = re.sub(r'[-\s]+', '-', text)
    return text if text else default_name

def cleanup_temp_assets(folder_path: str):
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print_status(f"Temporary assets directory '{folder_path}' cleaned up.")
        os.makedirs(folder_path, exist_ok=True)
    except Exception as e:
        print_error(f"Error cleaning up temporary assets in '{folder_path}': {e}")

# --- Environment/Library Setup ---
def setup_pillow_antialias():
    if not hasattr(PILImage, 'ANTIALIAS'):
        try: PILImage.ANTIALIAS = PILImage.Resampling.LANCZOS
        except AttributeError:
            try: PILImage.ANTIALIAS = PILImage.LANCZOS
            except AttributeError: print_warning("Could not set PIL.Image.ANTIALIAS.")

# --- Media Fetching Utilities ---
def fetch_from_pexels(
    query: str, section_num_str: str, pexels_api_key: str, temp_assets_dir: str,
    media_type: str = "videos", per_page: int = 5, orientation: str = "portrait"
) -> dict | None:
    if not pexels_api_key: print_warning("PEXELS_API_KEY not provided. Skipping Pexels."); return None
    headers = {"Authorization": pexels_api_key}
    search_url = f"https://api.pexels.com/{media_type}/search" if media_type == "videos" else f"https://api.pexels.com/v1/search"
    params = {"query": query, "per_page": per_page, "orientation": orientation}
    if media_type == "videos": params["size"] = "medium"
    
    print_status(f"Pexels: '{query}' (sec: {section_num_str}, type: {media_type})...")
    try:
        r = requests.get(search_url, headers=headers, params=params, timeout=25); r.raise_for_status()
        data = r.json(); items = data.get(media_type if media_type == "videos" else "photos", [])
        if not items and media_type == "videos":
            print_status(f"Pexels: No videos for '{query}'. Trying images."); 
            return fetch_from_pexels(query, section_num_str, pexels_api_key, temp_assets_dir, "photos", per_page, orientation)
        if not items: print_warning(f"Pexels: No {media_type} for '{query}'."); return None
        
        item = random.choice(items); item_id = item['id']; dl_url = None
        ext = ".mp4" if media_type == "videos" else ".jpg"
        if media_type == "videos":
            vfs = item.get('video_files', [])
            pvids = [vf for vf in vfs if vf.get('width',0) < vf.get('height',0) and '.mp4' in vf.get('link','').lower()]
            amp4s = sorted([vf for vf in vfs if '.mp4' in vf.get('link','').lower()], key=lambda x: x.get('height', 0), reverse=True)
            dl_url = (sorted(pvids,key=lambda x:x.get('height',0),reverse=True)[0]['link'] if pvids else amp4s[0]['link'] if amp4s else None)
            if not dl_url: print_warning(f"Pexels: No .mp4 for video ID {item_id}."); return None
        else: 
            src = item.get('src',{}); dl_url = src.get('portrait') or src.get('large2x') or src.get('large') or src.get('original')
            if dl_url: p_ext=os.path.splitext(dl_url.split('?')[0])[1]; ext = p_ext if p_ext and 2<len(p_ext)<6 else ext
            else: print_warning(f"Pexels: No image URL for photo ID {item_id}."); return None
        if not dl_url: print_warning(f"No download URL for Pexels item ID {item_id}."); return None

        mr = requests.get(dl_url, stream=True, timeout=70); mr.raise_for_status()
        fpath = os.path.join(temp_assets_dir, f"pexels_{sanitize_filename(query)}_{section_num_str}_{item_id}{ext}")
        with open(fpath, "wb") as f: 
            for chunk in mr.iter_content(chunk_size=1024*1024*2): f.write(chunk)
        print_status(f"Pexels: Downloaded {os.path.basename(fpath)}")
        return {"path":fpath, "type":media_type[:-1] if media_type.endswith('s') else media_type}
    except Exception as e: print_error(f"Pexels error ('{query}'): {e}"); return None

def fetch_from_freesound(
    query: str, freesound_api_key: str, temp_assets_dir: str,
    target_duration_sec: float = 45.0
) -> str | None:
    if not freesound_api_key: print_warning("FREESOUND_API_KEY not provided. Skipping background music."); return None
    
    url = "https://freesound.org/apiv2/search/text/"
    params = {
        "query": query,
        "token": freesound_api_key,
        "filter": f"duration:[{max(15, target_duration_sec - 15)} TO {target_duration_sec + 45}] license:\"Creative Commons 0\"",
        "fields": "id,name,previews,duration,username",
        "sort": "rating_desc",
        "page_size": 15
    }
    print_status(f"Freesound: Searching for music with query '{query}'...")
    try:
        r = requests.get(url, params=params, timeout=25); r.raise_for_status()
        data = r.json(); tracks = data.get('results', [])
        if not tracks: print_warning(f"Freesound: No music found for query: '{query}'."); return None

        track = random.choice(tracks)
        track_id = track['id']
        dl_url = track.get('previews', {}).get('preview-hq-mp3')
        if not dl_url: print_warning(f"Freesound: No HQ MP3 preview found for track ID {track_id}."); return None
        
        mr = requests.get(dl_url, stream=True, timeout=100); mr.raise_for_status()
        fpath = os.path.join(temp_assets_dir, f"freesound_music_{track_id}.mp3")
        with open(fpath, "wb") as f: 
            for chunk in mr.iter_content(chunk_size=1024*1024): f.write(chunk)
        
        with AudioFileClip(fpath) as clip: dur = clip.duration
        if dur and dur > 0.1: 
            print_status(f"Freesound: Downloaded '{track['name']}' by '{track['username']}' ({dur:.2f}s)")
            return fpath
        else: 
            print_error(f"Freesound: File '{fpath}' invalid duration."); 
            if os.path.exists(fpath): os.remove(fpath)
    except requests.exceptions.RequestException as e: print_error(f"Freesound API error ('{query}'): {e}")
    except Exception as e: print_error(f"Freesound fetch/download error: {e}\n{traceback.format_exc()}")
    return None

# --- Visual & Caption Clip Creation Utilities ---
def create_processed_visual_clip(
    media_info: dict | None, target_duration: float, video_dimensions: tuple[int, int],
    target_fps: int, max_stock_video_duration: int
) -> VideoFileClip | ImageClip | None:
    if not media_info or not media_info.get("path"):
        print_warning(f"No media_info.path. Creating blank clip for {target_duration:.2f}s.")
        blank_frame = np.zeros((video_dimensions[1], video_dimensions[0], 3), dtype=np.uint8)
        return ImageClip(blank_frame, ismask=False).set_duration(target_duration).set_fps(target_fps)

    path = media_info["path"]; media_type = media_info["type"]
    try:
        if media_type == "video":
            source_clip = VideoFileClip(path, audio=False, target_resolution=(video_dimensions[1], None))
            use_duration = min(source_clip.duration, target_duration, max_stock_video_duration)
            start_time = (source_clip.duration - use_duration) / 2 if source_clip.duration > use_duration else 0
            processed_clip = source_clip.subclip(start_time, start_time + use_duration).set_duration(use_duration)
        elif media_type == "image":
            source_clip = ImageClip(path)
            processed_clip = source_clip.set_duration(target_duration)
        else: raise ValueError(f"Unsupported media_type '{media_type}'")
        
        current_w, current_h = processed_clip.size
        target_w, target_h = video_dimensions
        aspect_ratio_current = float(current_w) / current_h if current_h > 0 else 1.0
        aspect_ratio_target = float(target_w) / target_h if target_h > 0 else 1.0
        if abs(aspect_ratio_current - aspect_ratio_target) < 0.01:
            final_visual = processed_clip.resize(video_dimensions)
        elif aspect_ratio_current > aspect_ratio_target:
            resized_clip = processed_clip.resize(height=target_h)
            final_visual = resized_clip.fx(crop, x_center=resized_clip.w / 2, width=target_w)
        else: 
            resized_clip = processed_clip.resize(width=target_w)
            final_visual = resized_clip.fx(crop, y_center=resized_clip.h / 2, height=target_h)
        return final_visual.set_fps(target_fps)
    except Exception as e:
        print_error(f"Error processing visual '{path}': {e}\n{traceback.format_exc()}")
        blank_frame = np.zeros((video_dimensions[1], video_dimensions[0], 3), dtype=np.uint8)
        return ImageClip(blank_frame, ismask=False).set_duration(target_duration).set_fps(target_fps)

def create_animated_karaoke_captions(
    script: str, audio_duration: float, timeline_start_time: float,
    video_width: int, keywords: list, style_config: dict
) -> list[CompositeVideoClip]:
    clips = []
    words = script.split()
    if not words: return []
    
    avg_word_duration = audio_duration / len(words) if len(words) > 0 else 0
    current_word_start_time = 0.0
    
    font_path = style_config.get("font_path", "DejaVu-Sans-Bold")
    font_size = style_config.get("font_size", 85)
    stroke_width = style_config.get("stroke_width", 4.5)
    
    for word in words:
        is_keyword = word.strip(".,!?").lower() in [k.lower() for k in keywords]
        
        text_color = style_config.get("accent_color", "yellow") if is_keyword else style_config.get("color", "white")
        
        try:
            word_clip = TextClip(
                word, fontsize=font_size, color=text_color, font=font_path,
                stroke_color=style_config.get("stroke_color", "black"),
                stroke_width=stroke_width, method='caption'
            )
        except Exception as e:
            print_error(f"Could not create TextClip for word '{word}': {e}")
            continue # Skip this word if it fails

        # Keyword animation: slight pop
        if is_keyword:
            word_clip = word_clip.fx(vfx.resize, lambda t: 1 + 0.1 * np.sin(t * np.pi * 5))

        # Heuristic for word duration based on character length
        char_based_duration = 0.08 * len(word) + 0.1 # Base time + time per char
        word_duration = max(avg_word_duration * 0.75, char_based_duration)
        word_duration = min(word_duration, avg_word_duration * 1.5) # Cap at 1.5x average

        word_clip = word_clip.set_position(('center', 'center')) \
                             .set_start(timeline_start_time + current_word_start_time) \
                             .set_duration(word_duration)

        clips.append(word_clip)
        current_word_start_time += word_duration

    return clips