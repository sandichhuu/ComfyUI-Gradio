# Dependencies & Setup

## pyproject.toml
```toml
[project]
name = "comfyui-gradio"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "gradio>=6.15.1",
]
```

## Runtime Dependencies (not in pyproject.toml)
These are expected to be available from ComfyUI:
- `torch` (PyTorch)
- `comfy` (ComfyUI core)
- `nodes` (ComfyUI nodes module)
- `folder_paths` (ComfyUI path utilities)
- `execution` (ComfyUI execution engine)
- `server` (ComfyUI server)
- `cuda_malloc` (ComfyUI CUDA allocator)
- `PIL` / `Pillow` (used in Flux2 worker)

## Model Files Required

### ZIT Models
| Model | Directory | Source |
|-------|-----------|--------|
| `ZImageTurbo/z_image_turbo_bf16.safetensors` | `diffusion_models/` | HuggingFace: Comfy-Org/z_image_turbo |
| `qwen_3_4b.safetensors` | `text_encoders/` | HuggingFace: Comfy-Org/z_image_turbo |
| `ae.safetensors` | `vae/` | HuggingFace: Comfy-Org/z_image_turbo |

### Flux2-Klein-9B Models
| Model | Directory | Source |
|-------|-----------|--------|
| `Flux.2 Klein 9B/flux-2-klein-9b-fp8.safetensors` | `diffusion_models/` | HuggingFace: black-forest-labs/FLUX.2-klein-9b-fp8 |
| `qwen_3_8b_fp8mixed.safetensors` | `text_encoders/` | HuggingFace: Comfy-Org/flux2-klein-9B |
| `full_encoder_small_decoder.safetensors` | `vae/` | HuggingFace: black-forest-labs/FLUX.2-small-decoder |
| `Flux.2 Klein 9B-base/KLEIN-Unchained-V2.safetensors` | (LoRA) | Base LoRA for Flux2 |

### User-Added LoRAs
Users can add custom LoRAs via the UI. Format: `<lora_name:weight>` (e.g., `<lora:76N0PGDVMCA64NA75C2NW7V600:1>`)

## Installation Steps

1. Ensure ComfyUI is installed and accessible
2. Clone this repository
3. Run `uv sync` to install dependencies
4. Ensure ComfyUI server is running
5. Run `uv run python main.py`

## Environment Variables
- `COMFYUI_PATH`: Override ComfyUI location (optional, auto-detected)
- `COMFYUI_TOPYTHON_UNLOAD_MODELS`: Set to "1"/"true"/"yes"/"on" to unload models after each generation
