# ComfyUI-Gradio: Project Overview

## Purpose
A Gradio-based web interface for ComfyUI that provides a simplified UI for image generation. It wraps ComfyUI's node-based workflow into a clean tabbed interface with prompt input, seed control, LoRA management, and image output.

## Tech Stack
- **Python >= 3.13** with **uv** package manager
- **Gradio >= 6.15.1** for the web UI
- **ComfyUI** running as a backend server (imported at runtime via `sys.path` manipulation)
- **PIL/Pillow** for image handling (used in Flux2 worker)

## Project Structure
```
comfyui-gradio/
├── main.py                              # Entry point, Gradio Blocks setup, global CSS
├── views/
│   ├── zit_view.py                      # ZIT tab UI renderer + event handlers
│   └── flux2_klein_9b_view.py           # Flux2-Klein-9B tab UI renderer + event handlers
├── workers/
│   ├── zit_worker.py                    # ZIT ComfyUI workflow execution
│   └── flux2_klein_9b_worker.py         # Flux2-Klein-9B ComfyUI workflow execution
├── documents/                           # This documentation folder
├── markdowns/                           # Original implementation specs
├── pyproject.toml                       # Project config (uv)
└── README.md
```

## Key Design Decisions
1. **No comments in code** - explicit project rule
2. **Workers are extracted ComfyUI scripts** - generated from ComfyUI's "Save as script" feature, then adapted to accept parameters
3. **Runtime ComfyUI import** - workers dynamically add ComfyUI to `sys.path` and import nodes at runtime
4. **LoRA tags merge invisibly** - user enters `<name:weight>` tags, they get appended to prompt text before sending to ComfyUI
5. **Generate/Stop button toggle** - UI uses chained `.click().then()` events to swap button visibility
6. **Browser refresh preservation** - Gradio's state management handles session persistence

## Two Model Tabs
- **ZIT (Z-Image-Turbo)**: Text-to-image only. Fast generation, 8 steps, simple scheduler.
- **Flux2-Klein-9B**: Text-to-image with optional reference images. Uses 2 input images, 4 steps, Flux2 scheduler, SageAttention optimization.
