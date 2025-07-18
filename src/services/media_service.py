"""
This service handles fetching all media assets like stock videos and
background music from third-party APIs.
"""
# --- Standard Library Imports ---
import asyncio
import logging
import os
import random
from typing import Any, Dict, Optional, Tuple

# --- Third-Party Imports ---
import requests
from tqdm.asyncio import tqdm as asyncio_tqdm

# --- Local Application Imports ---
from config import (
    PEXELS_API_KEY, PIXABAY_API_KEY, MAX_RETRIES,
    REQUEST_TIMEOUT, TEMP_ASSETS_DIR
)
from models import VideoPlan
from utils import sanitize_filename

log = logging.getLogger(__name__)

class MediaService:
    async def get_assets_for_plan(self, plan: VideoPlan) -> Dict[str, Any]:
        video_tasks = [
            self._fetch_video_from_pexels(
                sub_scene.visual_search_query, f"{i}_{j}"
            )
            for i, section in enumerate(plan.sections)
            for j, sub_scene in enumerate(section.sub_scenes)
        ]
        if plan.call_to_action_text:
            video_tasks.append(
                self._fetch_video_from_pexels("motivational hopeful background", "cta")
            )
        
        music_task = asyncio.create_task(
            self._fetch_music_from_pixabay(plan.background_music_suggestion)
        )
        
        video_results = await asyncio_tqdm.gather(*video_tasks, desc="Downloading video assets")
        
        visuals = {res[0]: os.path.abspath(res[1]) for res in video_results if res}
        music = await music_task
        return {"music": os.path.abspath(music) if music else None, "visuals": visuals}

    async def _fetch_video_from_pexels(
        self, query: str, scene_id: str
    ) -> Optional[Tuple[str, str]]:
        if not PEXELS_API_KEY:
            return None
        url, headers = "https://api.pexels.com/videos/search", {
            "Authorization": PEXELS_API_KEY
        }
        params = {"query": query, "per_page": 15, "orientation": "portrait", "size": "medium"}
        for attempt in range(MAX_RETRIES):
            try:
                r = await asyncio.to_thread(
                    requests.get, url, headers=headers, params=params, timeout=REQUEST_TIMEOUT
                )
                r.raise_for_status()
                if not (videos := r.json().get("videos", [])):
                    return None
                video = random.choice(videos)
                video_files = sorted(
                    [
                        vf
                        for vf in video.get("video_files", [])
                        if ".mp4" in vf.get("link", "")
                    ],
                    key=lambda x: x.get("height", 0),
                    reverse=True,
                )
                if not video_files:
                    return None
                media_res = await asyncio.to_thread(
                    requests.get, video_files[0]["link"], stream=True, timeout=REQUEST_TIMEOUT * 2
                )
                media_res.raise_for_status()
                fpath = os.path.join(
                    TEMP_ASSETS_DIR, f"pexels_{sanitize_filename(query)}_{scene_id}_{video['id']}.mp4"
                )
                with open(fpath, "wb") as f:
                    for chunk in media_res.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                return (scene_id, fpath)
            except requests.RequestException as e:
                log.warning(f"Pexels API error for '{query}' on attempt {attempt + 1}: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2)
        return None

    async def _fetch_music_from_pixabay(self, query: str) -> Optional[str]:
        if not PIXABAY_API_KEY:
            return None
        url, params = "https://pixabay.com/api/", {
            "key": PIXABAY_API_KEY,
            "q": query,
            "media_type": "music",
            "safesearch": "true",
            "per_page": 20,
        }
        for attempt in range(MAX_RETRIES):
            try:
                r = await asyncio.to_thread(
                    requests.get, url, params=params, timeout=REQUEST_TIMEOUT
                )
                r.raise_for_status()
                tracks = [
                    t
                    for t in r.json().get("hits", [])
                    if t.get("duration", 0) > 30 and t.get("downloadURL")
                ]
                if not tracks:
                    log.warning(f"No suitable music tracks found for '{query}'.")
                    return None
                track = random.choice(tracks)
                music_res = await asyncio.to_thread(
                    requests.get, track["downloadURL"], stream=True, timeout=REQUEST_TIMEOUT * 2
                )
                music_res.raise_for_status()
                fpath = os.path.join(
                    TEMP_ASSETS_DIR, f"pixabay_music_{track['id']}.mp3"
                )
                with open(fpath, "wb") as f:
                    for chunk in music_res.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                return fpath
            except requests.RequestException as e:
                log.warning(f"Pixabay API error for '{query}' on attempt {attempt + 1}: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2)
        return None