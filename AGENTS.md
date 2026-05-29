# AGENTS.md

## Project
ComfyUI-Gradio: Gradio web interface for ComfyUI image generation. Two model tabs (ZIT and Flux2-Klein-9B) with LoRA support.

## Rules
- **No comments in code**
- **Do not run Python files** - owner reviews and runs manually
- Use `uv` as package manager (Python 3.13)
- Follow existing code conventions - no new frameworks or libraries

## Commands
- Install: `uv sync`
- Run: `uv run python main.py` (requires ComfyUI server running)

## Architecture
- `main.py` - Entry point, Gradio Blocks setup, global CSS styling
- `views/zit_view.py` - ZIT tab UI renderer + event handlers
- `views/flux2_klein_9b_view.py` - Flux2 tab UI renderer + event handlers
- `workers/zit_worker.py` - ZIT ComfyUI workflow execution
- `workers/flux2_klein_9b_worker.py` - Flux2 ComfyUI workflow execution
- `documents/` - Full implementation docs (re-implementation guide)
- `markdowns/` - Original implementation specs

## Key Constraints
- ComfyUI server must be running before starting
- Workers import ComfyUI modules at runtime via `sys.path` manipulation
- `COMFYUI_PATH` env var or parent directory search locates ComfyUI
- `extra_model_paths.yaml` configures additional model locations
- Images > threshold get height normalized to 444px
- LoRA tags (`<name:weight>`) merge into prompt invisibly
- Generate button toggles to Stop during generation
- Browser refresh preserves session state
- Each generation bootstraps full ComfyUI runtime (heavy init)

## Patterns
- Views export `create_<model>_tab()` functions
- Workers export `generate()` functions
- LoRA state: `gr.State([])` with `@gr.render` for dynamic rows
- Button chaining: `toggle_generation → generate_image → stop_generation`
- Output discovery: glob for most recent `<prefix>_*.png` in output directory
