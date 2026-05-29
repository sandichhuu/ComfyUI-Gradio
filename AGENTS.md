# AGENTS.md

## Project Overview
ComfyUI-Gradio: Gradio web interface for ComfyUI image generation. Two model tabs (ZIT and Flux2-Klein-9B) with LoRA support.

## Development Rules
- **No comments in code** (explicit project rule)
- **Do not run Python files** - owner reviews and runs manually
- Use `uv` as package manager (Python 3.13)

## Commands
- Install: `uv sync`
- Run: `uv run python main.py` (requires ComfyUI server running)

## Architecture
- `main.py` - Entry point, Gradio Blocks setup
- `views/` - Tab UI renderers (empty; implement `create_zit_tab()`, `create_flux2_klein_9b_tab()`)
- `workers/` - ComfyUI workflow execution (bootstraps runtime, builds prompts)
- `markdowns/` - Implementation specs and plans

## Key Constraints
- ComfyUI server must be running before starting
- Images > threshold get height normalized to 444px
- LoRA tags (`<name:weight>`) merge into prompt invisibly
- Browser refresh should preserve session state
- Generate button toggles to Stop during generation

## Workflow Notes
- Workers import ComfyUI modules at runtime via `sys.path` manipulation
- `COMFYUI_PATH` env var or parent directory search locates ComfyUI
- `extra_model_paths.yaml` configures additional model locations
