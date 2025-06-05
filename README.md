# 🔥 Fire Solid - AI Short Video Generator 🔥

## 1. Project Overview

This project is a Python-based application designed to automate the creation of short-form videos for platforms like YouTube Shorts, TikTok, and Instagram Reels. It leverages multiple AI services to transform a single text prompt into a complete video, including AI-planned content, narration, stock footage, background music, and dynamic captions.

The core philosophy is to move beyond simple automation and build a system that embodies the principles of creative direction, narrative structure, and emotional resonance to produce videos that are not just functional, but genuinely "fire solid" and attractive to viewers.

## 2. Implemented Features

The current version of the script includes the following core features:

* **AI-Powered Content Planning:**
    * Utilizes an OpenAI model (e.g., GPT-4o-mini) to generate a comprehensive JSON-based video plan from a user prompt.
    * The plan includes a detailed hook strategy, multiple content sections with scripts, and a call-to-action.
    * A "Creator & Critic" two-step AI process automatically refines the initial plan for improved narrative structure, depth, and engagement.

* **Empathic Text-to-Speech (TTS) with Fallback:**
    * Integrated to use **Hume AI's TTS API** for generating emotionally resonant voiceovers with controllable acting instructions, speed, and pauses.
    * Supports a consistent "brand voice" by allowing a default Hume Voice ID to be set in the environment configuration.
    * Includes a robust fallback to **OpenAI's HD TTS** service if Hume AI is not configured or encounters an error.

* **Automated Media Sourcing:**
    * **Visuals:** Fetches high-resolution stock videos and images from **Pexels** based on AI-generated search queries for each scene.
    * **Music:** Fetches royalty-free background music from **Freesound's** extensive library of Creative Commons 0 tracks, ensuring videos are safe for any use.

* **Dynamic Caption Generation:**
    * Generates animated, "karaoke-style" captions where words appear sequentially.
    * Timings are estimated based on the audio duration of each narration segment.
    * Supports highlighting of AI-identified keywords with different colors and a subtle animation effect.
    * Caption styling (font, size, color, stroke) is configurable.

* **Automated Video Assembly:**
    * Uses the **MoviePy** library to composite all generated assets—visuals, multi-segment narration, background music, and animated captions—into a single, cohesive video file in 9:16 aspect ratio.

* **Modular Architecture:**
    * The codebase is organized into a `main.py` orchestrator and a `utils.py` module containing stable, reusable functions for media fetching, clip processing, and caption rendering.

## 3. Setup and Usage

### 3.1. Prerequisites

* Python 3.10+
* `pip` and `venv` (or your preferred package manager like Poetry)
* `ffmpeg` (required by MoviePy for video processing)

### 3.2. Installation

1.  Clone the repository:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  Create and activate a virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  Install the required Python packages:
    ```bash
    pip install "hume[microphone]" openai python-dotenv requests Pillow moviepy numpy aiofiles
    ```

### 3.3. Configuration

1.  Create a `.env` file in the root of the project directory.
2.  Add your API keys and configuration to the `.env` file. Use the following template:

    ```env
    # REQUIRED: For AI Planning & Fallback TTS
    OPENAI_API_KEY="your_openai_api_key"

    # REQUIRED: For Empathic Voice Generation
    HUME_API_KEY="your_hume_api_key"
    # Find a voice on Hume's platform and set its name/ID here for brand consistency
    HUME_DEFAULT_VOICE_ID="your-chosen-hume-voice-name-or-id" 

    # REQUIRED: For Stock Visuals
    PEXELS_API_KEY="your_pexels_api_key"

    # REQUIRED: For Background Music
    FREESOUND_API_KEY="your_freesound_api_key"

    # REQUIRED: For Captions (path to a .ttf or .otf font file on your system)
    CAPTIONS_FONT_PATH="/path/to/your/font.ttf"
    ```

### 3.4. Activating Hume AI TTS

The script defaults to OpenAI TTS for safety. To enable the higher-quality Hume AI voice:
1.  Open `main.py`.
2.  Navigate to the `generate_tts_audio_segment` function.
3.  Find the multi-line comment block (`"""..."""`) containing the Hume API calls. **Uncomment this entire block.**
4.  Find and **delete or comment out** the line below it that says: `raise NotImplementedError(...)`.

### 3.5. Running the Script

1.  Ensure your virtual environment is activated.
2.  Run the main script from your terminal:
    ```bash
    python3 main.py
    ```
3.  Enter a video idea when prompted.
4.  The script will generate and save a refined JSON plan. You have the option to review and edit this plan before pressing Enter to proceed with the fully automated video generation.

## 4. Future Development Roadmap

This section outlines planned features and improvements to enhance the system's capabilities and output quality.

* **Tier 1: Advanced Captions & Sound**
    * **Perfectly Synced Karaoke Captions:** Integrate an ASR (Automatic Speech Recognition) model (e.g., a Whisper variant) to process the generated TTS audio and extract precise word timestamps. This will enable perfectly synchronized captions and word-level animations.
    * **Dynamic Sound Design:** Implement the `sfx_cues` from the AI plan by integrating a sound effect library and adding SFX to the final audio mix, timed to on-screen events.

* **Tier 2: Deeper AI Planning & Authenticity**
    * **Autonomous Prompt Enhancement:** Introduce an AI "Ideation" layer. If a user prompt is too general, this layer will first brainstorm several specific, high-engagement video concepts. The system will then automatically select the most promising concept and build the full plan around it, making it less reliant on the user's initial prompt quality.
    * **Enhanced Acting Direction:** Refine the AI planner's ability to generate even more nuanced Hume acting prompts, potentially describing transitions in emotion *within* a single sentence for more sophisticated delivery.

* **Tier 3: The Autonomous Creator (Long-Term Vision)**
    * **Tool Integration Framework:** Build a "Pluggable Tool" engine to execute API calls suggested in the AI plan (e.g., for web search to fetch real-time data, data visualization to create charts, or AI image generation for custom visuals).
    * **Iterative Generation (The "Context Layer"):** Re-architect the main loop to operate iteratively on 3-5 second segments. This involves developing a sophisticated **AI Evaluator Module** that can analyze rendered video chunks and provide actionable creative feedback to the AI Planner for the next segment's creation, creating a true self-correcting feedback loop.