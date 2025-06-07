# src/planning.py
"""
Handles all AI-driven content and story planning.

This module is responsible for loading the brand persona and using the OpenAI API
to generate and refine the video script based on that persona.
"""
import json
import os
import time
from openai import OpenAI
from utils import print_status, print_error, print_warning

def load_brand_persona(persona_file_path: str) -> dict | None:
    """Loads the brand persona from the specified JSON file."""
    if not os.path.exists(persona_file_path):
        print_error(f"Persona file not found at '{persona_file_path}'.")
        print_warning(f"Please create this file (e.g., 'brand_persona.json') or use the --persona flag.")
        return None
    try:
        with open(persona_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print_error(f"Error loading or parsing persona file '{persona_file_path}': {e}")
        return None

def plan_video_content(user_prompt: str, persona: dict, client: OpenAI) -> dict | None:
    """Orchestrates the AI planning process, including draft and critique."""
    print_status("Starting AI video planning process...")
    
    draft_plan = _run_creator_ai(user_prompt, persona, client)
    if not draft_plan:
        print_error("Initial draft plan generation failed."); return None

    final_plan = _run_guardian_ai(json.dumps(draft_plan), persona, client)
    if not final_plan:
        print_warning("AI Brand Guardian failed to refine the plan. Using original draft.")
        return draft_plan
        
    return final_plan

def _run_creator_ai(user_prompt: str, persona: dict, client: OpenAI) -> dict | None:
    """The Persona-driven AI generates a story-focused first draft with SSML."""
    print_status("Generating script with AI Persona...")
    
    supported_emotions = list(persona.get('speech_settings', {}).get('emotion_prosody_map', {}).keys())
    default_rate = persona.get('speech_settings', {}).get('default_rate', 'medium')

    system_prompt = f"""
    You are a viral video scriptwriter and voice director. You must fully embody the brand persona to write a script that is impossible to scroll past.

    **YOUR PERSONA & VOICE:**
    - Archetype: {persona['archetype']}
    - Perspective: {persona.get('storytelling_perspective', 'An expert.')}
    - Vocabulary Level: **{persona.get('vocabulary_level', 'B1 Intermediate')}**.

    **CRITICAL SSML RULES:**
    1.  **SEPARATE TAGS:** The `<prosody>` tag is ONLY for `rate`, `pitch`, and `volume`. The `<speechify:style>` tag is ONLY for `emotion`. DO NOT mix their attributes.
    2.  **CORRECT EXAMPLE:** `<speak><prosody rate="{default_rate}"><speechify:style emotion="cheerful">This is correct.</speechify:style></prosody></speak>`
    3.  **INCORRECT EXAMPLE:** `<speak><prosody rate="{default_rate}" emotion="cheerful">This is WRONG.</prosody></speak>`
    4.  **VALID EMOTIONS ONLY:** You MUST only use emotions from this exact list: **{supported_emotions}**.
    5.  **GLOBAL PROSODY:** Wrap the ENTIRE narration of every script field in a single `<speak><prosody rate="{default_rate}">...</prosody></speak>` structure. Place emotion and break tags inside it.

    **OTHER RULES:**
    - Write from the specified 'Storytelling Perspective'. Use personal anecdotes.
    - Avoid all forbidden phrases: {persona['forbidden_phrases']}.

    **TASK:**
    Generate a complete JSON video plan with rich, VALID SSML. Output ONLY the valid JSON object.
    """
    json_structure_prompt = """
    {
      "video_title": "A highly clickable, human-centric title.",
      "sections": [
        {
          "section_title": "The Hook",
          "narrative_script": "<speak><prosody rate='fast'><speechify:style emotion='fearful'>Your mind goes blank... right?</speechify:style><break time='300ms'/><speechify:style emotion='calm'>I've been there.</speechify:style></prosody></speak>",
          "stock_media_search_query": "intense frustrated student thinking hard",
          "keywords_for_highlighting": ["blank", "fear", "stuck"]
        },
        {
          "section_title": "The Core Idea",
          "narrative_script": "<speak><prosody rate='fast'><speechify:style emotion='assertive'>Here is the simple trick I learned...</speechify:style></prosody></speak>",
          "stock_media_search_query": "lightbulb moment idea clarity",
          "keywords_for_highlighting": ["idea", "solution", "method"]
        }
      ],
      "call_to_action_script": "<speak><prosody rate='fast'><speechify:style emotion='warm'>Give it a try. You might surprise yourself.</speechify:style></prosody></speak>",
      "background_music_suggestion": "lofi hip hop"
    }
    """
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User's video idea: '{user_prompt}'. Generate the JSON plan. The JSON structure MUST be: {json_structure_prompt}"}
                ],
                temperature=0.85, response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print_error(f"OpenAI Creator attempt {attempt + 1} FAILED: {e}")
            if attempt < 2: time.sleep(4)
            else: break
    return None

def _run_guardian_ai(draft_plan_json: str, persona: dict, client: OpenAI) -> dict | None:
    """The 'Brand Guardian' AI refines the draft."""
    print_status("Refining draft plan with AI Brand Guardian...")
    system_prompt = f"""
    You are a ruthless Brand Guardian and SSML expert. Your only loyalty is to the authenticity of the persona.

    **PERSONA TO ENFORCE:**
    - Valid Emotions: {list(persona.get('speech_settings', {}).get('emotion_prosody_map', {}).keys())}
    - Forbidden Phrases: {persona['forbidden_phrases']}

    **YOUR TASK:**
    Review the draft JSON. Rewrite the ENTIRE JSON to be 10x more human and technically correct.
    1.  **FIX THE SSML:** THIS IS YOUR TOP PRIORITY. Ensure that `emotion` attributes are ONLY in `<speechify:style>` tags and that `rate` is ONLY in `<prosody>` tags. The structure MUST be `<speak><prosody rate="..."><speechify:style emotion="...">...</speechify:style></prosody></speak>`. Correct any and all SSML errors.
    2.  **STRENGTHEN STORY:** Make the 'I've been there' perspective more powerful.
    3.  **SIMPLIFY:** Ensure vocabulary is simple and all search queries are short keywords.

    Output the complete, rewritten, and superior JSON object with perfect SSML.
    """
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Critique and rewrite this draft plan:\n\n{draft_plan_json}"}
                ],
                temperature=0.7, response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print_error(f"OpenAI Guardian attempt {attempt + 1} FAILED: {e}")
            if attempt < 2: time.sleep(5)
            else: break
    return None