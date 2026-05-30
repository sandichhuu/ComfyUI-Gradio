# ComfyUI-Gradio

A Gradio-based web interface for ComfyUI image generation with two model tabs (ZIT and Flux2-Klein-9B) and LoRA support.

## Features

- **Two Model Tabs**
  - **ZIT (Z-Image-Turbo)**: Fast text-to-image generation (8 steps, simple scheduler)
  - **Flux2-Klein-9B**: Text-to-image with reference image support (4 steps, Flux2 scheduler, SageAttention)
- **LoRA Management**: Collapsible list with enable/disable, name, weight, add/remove (max 6)
- **Generate/Stop Toggle**: Button switches between Generate and Stop during generation
- **Random Seed**: Seed=0 generates a random seed and displays it after generation
- **Reference Images (Flux2)**: Input Image + optional Ref Image for guided generation
- **Session Persistence**: Web interface preserves state after browser refresh

## Project Structure

```
comfyui-gradio/
├── main.py                              # Entry point, Gradio Blocks, global CSS
├── views/
│   ├── zit_view.py                      # ZIT tab UI + event handlers
│   └── flux2_klein_9b_view.py           # Flux2 tab UI + event handlers
├── workers/
│   ├── zit_worker.py                    # ZIT ComfyUI workflow execution
│   └── flux2_klein_9b_worker.py         # Flux2 ComfyUI workflow execution
├── documents/                           # Full implementation docs (re-implementation guide)
├── markdowns/                           # Original implementation specs
├── pyproject.toml                       # Project config
└── README.md
```

## Requirements

- Python >= 3.13
- uv (Python package manager)
- Gradio >= 6.15.1
- ComfyUI server (must be running)
- ComfyUI model files (see `documents/08-dependencies-setup.md`)

## Installation

```bash
git clone <repository-url>
cd comfyui-gradio
uv sync
```

## Usage

1. Start the ComfyUI server (ensure it's running in the background)
2. Run the application:
   ```bash
   uv run python main.py
   ```
3. Open browser at `http://localhost:7860`

## LoRA Format

LoRAs use the format `<name:weight>` and are merged into the prompt invisibly:
- Example: `<lora:76N0PGDVMCA64NA75C2NW7V600:1>`
- Tags are appended to the prompt text before sending to ComfyUI

## Key Technical Details

- Workers dynamically import ComfyUI modules at runtime via `sys.path` manipulation
- `COMFYUI_PATH` env var or parent directory search locates ComfyUI
- `extra_model_paths.yaml` configures additional model locations
- Images with height > 444px are displayed at 444px in the UI
- Each generation bootstraps the full ComfyUI runtime (heavy initialization)
- ZIT outputs are saved as `zit_*.png`, Flux2 as `Flux2-Klein_*.png`

## Documentation

See the `documents/` folder for full implementation details:
- `00-project-overview.md` - Project overview
- `01-architecture.md` - Architecture and data flow
- `02-main-entry-point.md` - main.py implementation
- `03-zit-view.md` - ZIT tab UI
- `04-flux2-klein-9b-view.md` - Flux2 tab UI
- `05-zit-worker.md` - ZIT worker
- `06-flux2-klein-9b-worker.md` - Flux2 worker
- `07-comfyui-integration.md` - ComfyUI integration
- `08-dependencies-setup.md` - Dependencies and model files
- `09-lora-system.md` - LoRA system
- `10-image-processing.md` - Image processing
- `11-ui-styling.md` - CSS styling

## Preview

<img width="1919" height="944" alt="image" src="https://github.com/user-attachments/assets/e3f1c979-a8bc-44dd-b102-4d1a00f42541" />
