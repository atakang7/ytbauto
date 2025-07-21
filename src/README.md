README.md
Iteration: 25.5 (Planning)

1. Project Constitution & Development Philosophy
(This section is stable and remains unchanged.)

2. Iteration 25.5 Plan: User Experience & Production Readiness
With the core rendering pipeline now stable, this iteration focuses on transforming the application from a raw engine into a polished, user-friendly production tool. We will address key areas of user feedback, workflow flexibility, and system robustness.

2.1. Progress Indicators (User Feedback)
Problem: Long-running operations like AI analysis, asset downloads, and video rendering currently provide minimal feedback, making the application appear unresponsive or frozen.

Action:

Add tqdm to the project's dependencies in pyproject.toml.

Integrate tqdm to provide clear, real-time progress bars for all significant, multi-step operations. This will be applied to:

Asset Downloads: Wrap the asyncio.gather calls in MediaService to show progress as video/music files are downloaded.

AI Vision Tagging: Wrap the asyncio.gather call in VideoAnalysisService to show progress as each scene is tagged.

The moviepy logger already provides a render bar, which is sufficient for now.

2.2. Decoupled Workflow & Plan Management (Flexibility)
Problem: The current workflow is an "all-or-nothing" process. The user cannot review or modify an AI-generated plan before committing to a potentially long render.

Action:

Introduce an Interactive Prompt: After a VideoPlan or RemixPlan is generated and saved, the script will now prompt the user: "AI plan saved to '...'. Proceed with rendering? (y/n)". If the user enters 'n', the script will exit gracefully.

Create a New "Render from File" Mode: A third option will be added to the main menu: [3] Render From Plan File.

Implement the New Mode: This workflow will prompt the user for a path to a plan's JSON file. It will then load, validate, and execute only the asset-gathering and rendering services, completely bypassing the AI planning stage. This gives the user full creative control to manually edit the AI's decisions.

2.3. Pre-flight System Check (Robustness)
Problem: The application has a hard dependency on the external command-line tool FFmpeg (required by moviepy). If it's not installed, the application will crash with a cryptic error late in the rendering process.

Action:

The check_dependencies() function in main.py will be enhanced.

It will use shutil.which('ffmpeg') to verify that the FFmpeg executable is available in the system's PATH.

If FFmpeg is not found, the application will log a fatal, user-friendly error with installation instructions and exit immediately.

3. Plan for Next Iteration (26)
The next coding iteration will be the Implementation of the User Experience & Production Readiness Features. It will involve adding tqdm, implementing the decoupled workflow with the new "Render from File" mode, and enhancing the pre-flight dependency check to include FFmpeg.