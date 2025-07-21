"""
This service handles fetching all media assets like stock videos and
background music from third-party APIs.

Fixed to ensure reliable background music fetching.
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
    def __init__(self):
        """Initialize with API key validation."""
        self.has_pixabay = bool(PIXABAY_API_KEY)
        self.has_pexels = bool(PEXELS_API_KEY)
        
        log.info(f"MediaService initialized - Pexels: {'✅' if self.has_pexels else '❌'}, Pixabay: {'✅' if self.has_pixabay else '❌'}")
        
        if not self.has_pixabay:
            log.warning("⚠️ PIXABAY_API_KEY not configured - background music will be unavailable")

    async def get_assets_for_plan(self, plan: VideoPlan) -> Dict[str, Any]:
        """Fetch all media assets with parallel processing."""
        # Prepare video download tasks
        video_tasks = [
            self._fetch_video_from_pexels(
                sub_scene.visual_search_query, f"{i}_{j}"
            )
            for i, section in enumerate(plan.sections)
            for j, sub_scene in enumerate(section.sub_scenes)
        ]
        
        # Add CTA video task
        if plan.call_to_action_text:
            video_tasks.append(
                self._fetch_video_from_pexels("motivational hopeful background", "cta")
            )
        
        # Create music download task
        music_task = asyncio.create_task(
            self._fetch_background_music_reliably(plan.background_music_suggestion)
        )
        
        # Execute video downloads with progress
        log.info(f"Downloading {len(video_tasks)} videos...")
        video_results = await asyncio_tqdm.gather(*video_tasks, desc="Downloading video assets")
        
        # Process video results
        visuals = {}
        for result in video_results:
            if result:  # result is (scene_id, filepath) tuple
                scene_id, filepath = result
                visuals[scene_id] = os.path.abspath(filepath)
        
        # Get music result
        music_path = await music_task
        
        # Log final results
        log.info(f"📹 Videos: {len(visuals)}/{len(video_tasks)} successful")
        log.info(f"🎵 Music: {'✅ Available' if music_path else '❌ Failed'}")
        
        return {
            "music": os.path.abspath(music_path) if music_path else None,
            "visuals": visuals
        }

    async def _fetch_background_music_reliably(self, music_suggestion: str) -> Optional[str]:
        """
        Fetch background music with multiple fallback strategies.
        
        This method tries multiple approaches to ensure music is always found:
        1. Exact search query
        2. Simplified search terms  
        3. Generic fallback searches
        """
        if not self.has_pixabay:
            log.error("❌ Cannot fetch music - PIXABAY_API_KEY not configured")
            return None
        
        log.info(f"🎵 Searching for background music: '{music_suggestion}'")
        
        # Define search strategies from specific to generic
        search_strategies = [
            music_suggestion,  # Original suggestion
            self._simplify_music_query(music_suggestion),  # Simplified version
            "background music",  # Generic fallback
            "instrumental",  # Even more generic
            "music"  # Last resort
        ]
        
        for attempt, query in enumerate(search_strategies, 1):
            if not query or query == music_suggestion and attempt > 1:
                continue  # Skip if same as previous or empty
                
            log.debug(f"🎵 Music search attempt {attempt}: '{query}'")
            
            try:
                music_path = await self._fetch_music_from_pixabay_with_query(query)
                if music_path:
                    log.info(f"✅ Music found with query '{query}' on attempt {attempt}")
                    return music_path
                else:
                    log.debug(f"⚠️ No music found for '{query}'")
                    
            except Exception as e:
                log.warning(f"⚠️ Music search failed for '{query}': {e}")
                continue
        
        log.error("❌ All music search strategies failed - no background music available")
        return None

    def _simplify_music_query(self, original_query: str) -> str:
        """
        Simplify music search query to increase chances of finding results.
        
        Examples:
        - "Upbeat and inspiring instrumental" → "upbeat instrumental"
        - "Motivational background music" → "motivational music"
        """
        if not original_query:
            return "music"
        
        # Convert to lowercase and split
        words = original_query.lower().split()
        
        # Keep only important music-related keywords
        important_keywords = [
            'music', 'instrumental', 'background', 'upbeat', 'calm', 'relaxing',
            'motivational', 'inspiring', 'energetic', 'peaceful', 'ambient',
            'acoustic', 'electronic', 'piano', 'guitar', 'orchestral'
        ]
        
        # Filter to important words
        filtered_words = [word for word in words if any(keyword in word for keyword in important_keywords)]
        
        # If we have good keywords, use them; otherwise fallback
        if filtered_words:
            simplified = ' '.join(filtered_words[:2])  # Max 2 words
            log.debug(f"Simplified '{original_query}' → '{simplified}'")
            return simplified
        else:
            return "background music"

    async def _fetch_music_from_pixabay_with_query(self, query: str) -> Optional[str]:
        """Fetch music from Pixabay with a specific query."""
        if not self.has_pixabay or not query:
            return None
            
        url = "https://pixabay.com/api/"
        params = {
            "key": PIXABAY_API_KEY,
            "q": query,
            "media_type": "music",
            "safesearch": "true",
            "per_page": 20,  # Get more options
            "min_duration": 30,  # At least 30 seconds
            "category": "music"
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                log.debug(f"🔍 Pixabay API call for '{query}' (attempt {attempt + 1})")
                
                response = await asyncio.to_thread(
                    requests.get, url, params=params, timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()
                
                data = response.json()
                tracks = data.get("hits", [])
                
                if not tracks:
                    log.debug(f"No tracks found for '{query}'")
                    return None
                
                # Filter tracks for better quality
                suitable_tracks = []
                for track in tracks:
                    duration = track.get("duration", 0)
                    download_url = track.get("downloadURL")
                    
                    if download_url and duration >= 30:  # At least 30 seconds
                        suitable_tracks.append(track)
                
                if not suitable_tracks:
                    log.debug(f"No suitable tracks found for '{query}' (duration/download issues)")
                    return None
                
                # Select random track from suitable options
                selected_track = random.choice(suitable_tracks)
                track_id = selected_track.get("id")
                download_url = selected_track["downloadURL"]
                duration = selected_track.get("duration", 0)
                
                log.debug(f"📥 Downloading track {track_id} ({duration}s) from Pixabay...")
                
                # Download the music file
                music_response = await asyncio.to_thread(
                    requests.get, download_url, stream=True, timeout=REQUEST_TIMEOUT * 2
                )
                music_response.raise_for_status()
                
                # Save to file
                filename = f"pixabay_music_{track_id}.mp3"
                filepath = os.path.join(TEMP_ASSETS_DIR, filename)
                
                with open(filepath, "wb") as f:
                    for chunk in music_response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                
                # Verify file was downloaded
                if os.path.exists(filepath) and os.path.getsize(filepath) > 10000:  # At least 10KB
                    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    log.info(f"✅ Music downloaded: {filename} ({file_size_mb:.1f}MB, {duration}s)")
                    return filepath
                else:
                    log.warning(f"⚠️ Downloaded file is too small or missing: {filepath}")
                    return None
                    
            except requests.RequestException as e:
                log.warning(f"⚠️ Pixabay API error for '{query}' on attempt {attempt + 1}: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2)  # Wait before retry
            except Exception as e:
                log.error(f"❌ Unexpected error downloading music for '{query}': {e}")
                return None
        
        log.warning(f"⚠️ All {MAX_RETRIES} attempts failed for music query '{query}'")
        return None

    async def _fetch_video_from_pexels(
        self, query: str, scene_id: str
    ) -> Optional[Tuple[str, str]]:
        """Fetch video from Pexels - unchanged from original."""
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