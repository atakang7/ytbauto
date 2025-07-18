"""
Pydantic Data Models

This file centralizes all Pydantic data models used across the application,
preventing circular import errors.
"""
from typing import List, Optional, Tuple, Literal
from pydantic import BaseModel, Field

# --- Generative Mode Models ---
MotionType = Literal['none', 'zoom_in', 'zoom_out', 'pan_left', 'pan_right']

class SubScene(BaseModel):
    narration_text: str
    visual_search_query: str
    emotion: str
    keywords_for_highlighting: List[str] = []
    motion_type: MotionType = 'none'

class VideoSection(BaseModel):
    section_title: str
    sub_scenes: List[SubScene]

class VideoPlan(BaseModel):
    video_title: str
    sections: List[VideoSection]
    call_to_action_text: Optional[str] = None
    background_music_suggestion: str

# --- Transformative Mode Models ---
class AnalysedScene(BaseModel):
    scene_id: int
    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    tags: List[str] = Field(default=[])

class TextOverlay(BaseModel):
    scene_id: int
    text_content: str
    font_size: int = 70
    position: Tuple[str, str] = ('center', 'bottom')

class VoiceoverSegment(BaseModel):
    scene_id: int
    narration_text: str

class RemixPlan(BaseModel):
    remix_video_title: str
    source_video_path: str
    scene_ids_to_include: List[int]
    background_music_suggestion: str
    text_overlays: Optional[List[TextOverlay]] = None
    voiceover_segments: Optional[List[VoiceoverSegment]] = None