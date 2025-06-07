# src/visuals.py
"""
Handles the fetching and processing of all visual elements,
including stock videos, music, and animated captions.
"""
import os
import random
import requests
from moviepy.editor import (
    VideoFileClip, TextClip, vfx
)
from moviepy.video.fx import crop

from utils import print_status, print_error, print_warning, sanitize_filename
import config

def fetch_from_pexels(query: str | None, section_id: str, api_key: str, temp_dir: str) -> dict | None:
    if not query:
        print_warning(f"Pexels query missing for section '{section_id}'. Using fallback.")
        query = "abstract technology background"
    if not api_key:
        print_warning("PEXELS_API_KEY not provided. Skipping Pexels search.")
        return None

    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": 10, "orientation": "portrait", "size": "medium"}
    
    print_status(f"Pexels: Searching for video '{query[:70]}...'")
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        videos = r.json().get("videos", [])
        if not videos:
            print_warning(f"Pexels: No videos found for '{query}'.")
            return None

        video = random.choice(videos)
        video_files = sorted([vf for vf in video.get('video_files', []) if '.mp4' in vf.get('link','')], key=lambda x: x.get('height', 0), reverse=True)
        if not video_files:
            print_warning(f"Pexels: No MP4 files found for video ID {video['id']}.")
            return None
        
        dl_url = video_files[0]['link']
        media_res = requests.get(dl_url, stream=True, timeout=70)
        media_res.raise_for_status()
        
        filename = f"pexels_{sanitize_filename(query)}_{video['id']}.mp4"
        fpath = os.path.join(temp_dir, filename)

        with open(fpath, "wb") as f:
            for chunk in media_res.iter_content(chunk_size=1024*1024):
                f.write(chunk)
        
        print_status(f"Pexels: Downloaded {os.path.basename(fpath)}")
        return {"path": fpath}
    except Exception as e:
        print_error(f"Pexels API error for query '{query}': {e}")
        return None

def fetch_music_from_pixabay(query: str, api_key: str, temp_dir: str) -> str | None:
    if not api_key:
        print_warning("PIXABAY_API_KEY not provided. Skipping background music.")
        return None
        
    url = "https://pixabay.com/api/"
    params = {"key": api_key, "q": query, "media_type": "music", "safesearch": "true", "per_page": 15}
    
    print_status(f"Pixabay Music: Searching for '{query}'...")
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        tracks = response.json().get("hits", [])
        if not tracks:
            print_warning(f"Pixabay Music: No results for '{query}'. Trying fallback 'background music'.")
            params["q"] = "background music"
            response = requests.get(url, params=params, timeout=30)
            tracks = response.json().get("hits", [])
        if not tracks:
            print_error("Pixabay Music: No music found."); return None

        suitable_tracks = [t for t in tracks if t.get("duration", 0) > 30 and t.get("downloadURL")]
        if not suitable_tracks:
            print_warning("No suitable tracks >30s found, using any track with a download URL.")
            suitable_tracks = [t for t in tracks if t.get("downloadURL")]
        if not suitable_tracks:
            print_error("No tracks with valid download URLs found."); return None

        track = random.choice(suitable_tracks)
        dl_url = track["downloadURL"]
        
        music_res = requests.get(dl_url, stream=True, timeout=100)
        music_res.raise_for_status()
        filename = f"pixabay_music_{track['id']}.mp3"
        fpath = os.path.join(temp_dir, filename)

        with open(fpath, "wb") as f:
            for chunk in music_res.iter_content(chunk_size=1024*1024): f.write(chunk)
        
        print_status(f"Pixabay Music: Downloaded '{track.get('tags', 'track')}'")
        return fpath
    except Exception as e:
        print_error(f"Pixabay API error for query '{query}': {e}"); return None

def create_processed_visual_clip(media_path: str, duration: float) -> VideoFileClip:
    try:
        clip = VideoFileClip(media_path, audio=False, target_resolution=(config.VIDEO_DIMS[1] + 100, None))
        start = max(0, (clip.duration - duration) / 2) if clip.duration > duration else 0
        clip = clip.subclip(start, start + min(duration, config.MAX_STOCK_VIDEO_DURATION)).set_duration(duration)
        
        resized = clip.resize(height=config.VIDEO_DIMS[1])
        if resized.w < config.VIDEO_DIMS[0]:
            resized = clip.resize(width=config.VIDEO_DIMS[0])
            
        final = crop(resized, x_center=resized.w/2, y_center=resized.h/2, width=config.VIDEO_DIMS[0], height=config.VIDEO_DIMS[1])
        return final.set_fps(config.FPS)
    except Exception as e:
        print_error(f"Failed to process visual clip '{media_path}': {e}")
        from moviepy.editor import ColorClip
        return ColorClip(size=config.VIDEO_DIMS, color=(0,0,0), duration=duration).set_fps(config.FPS)

def group_words_for_karaoke(word_timings: list[dict], max_words: int = 2) -> list[list[dict]]:
    if not word_timings: return []
    groups, current_group = [], []
    for word in word_timings:
        current_group.append(word)
        if len(current_group) >= max_words:
            groups.append(current_group); current_group = []
    if current_group: groups.append(current_group)
    return groups

def create_asr_synced_captions(asr_words: list, keywords: list) -> list[TextClip]:
    if not asr_words: return []
    all_clips = []
    normalized_keywords = [k.strip().lower() for k in keywords if k]
    video_width, video_height = config.VIDEO_DIMS
    style = config.CAPTION_STYLE
    
    word_groups = group_words_for_karaoke(asr_words, max_words=2)

    for group in word_groups:
        start_time, end_time = group[0]['start'], group[-1]['end']
        duration = end_time - start_time
        if duration <= 0.01: continue

        group_text = " ".join([w.get('word', '').strip() for w in group])
        has_keyword = any(w.get('word', '').strip(".,!?").lower() in normalized_keywords for w in group)
        text_color = style["accent_color"] if has_keyword else style["color"]

        try:
            text_clip = TextClip(
                txt=group_text, font=style["font_path"], fontsize=style["font_size"],
                color=text_color, stroke_color=style["stroke_color"],
                stroke_width=style["stroke_width"], method='caption', align='center',
                size=(video_width * 0.9, None)
            )
        except Exception as e:
            print_error(f"Could not create TextClip for '{group_text}': {e}"); continue

        pop_duration = min(0.15, duration * 0.4)
        def resize_func(t):
            if t < pop_duration: return 1 + 0.2 * (1 - (1 - t / pop_duration)**3)
            return 1.2 if has_keyword else 1.0

        final_clip = (text_clip.fx(vfx.resize, resize_func)
                               .set_start(start_time)
                               .set_duration(duration)
                               .fadein(pop_duration / 2)
                               .fadeout(pop_duration / 2)
                               .set_position(('center', video_height * 0.75 + random.uniform(-20, 20))))
        all_clips.append(final_clip)
        
    return all_clips