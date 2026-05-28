# ComfyUI-Gradio

A Gradio-based web interface for ComfyUI, providing an intuitive UI for image generation with support for multiple models including Z-Image-Turbo (ZIT) and Flux2-Klein-9B.

## Features

- **Two Model Tabs**:
  - **ZIT Tab**: Z-Image-Turbo model for fast image generation
  - **Flux2-Klein-9B Tab**: Advanced model with reference image support
- **LoRA Support**: Dynamic LoRA management with enable/disable toggles, name, and strength controls
- **Interactive UI**:
  - Prompt input with Generate/Stop functionality
  - Configurable parameters: seed (with random toggle), width, height
  - Real-time button state changes (Generate ↔ Stop)
  - Collapsible LoRA list with add/remove functionality
- **Image Processing**: Automatic height normalization (fix-height = 444px) for large images
- **Session Persistence**: Web interface remains functional after browser refresh

## Project Structure

```
comfyui-gradio/
├── main.py                 # Application entry point
├── views/
│   ├── zit_view.py         # ZIT tab UI renderer
│   └── flux2_klein_9b_view.py  # Flux2-Klein-9B tab UI renderer
├── workers/
│   ├── zit_worker.py       # ZIT model worker (extracted from ComfyUI)
│   └── flux2_klein_9b_worker.py  # Flux2-Klein-9B model worker
├── markdowns/              # Documentation files
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Requirements

- Python >= 3.13
- uv (Python package manager)
- Gradio >= 6.15.1
- ComfyUI server

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd comfyui-gradio
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

## Usage

1. Start the ComfyUI server (ensure it's running in the background)

2. Run the application:
   ```bash
   uv run python main.py
   ```

3. Open your browser and navigate to the URL displayed in the terminal (typically `http://localhost:7860`)

## Configuration

### ZIT Tab
- **Prompt**: Text description for image generation
- **Seed**: Random seed value with toggle for randomization
- **Width/Height**: Output image dimensions
- **LoRA List**: Enable/disable LoRAs with adjustable strength

### Flux2-Klein-9B Tab
- **Prompt**: Text description for image generation
- **Seed**: Random seed value with toggle for randomization
- **Width/Height**: Output image dimensions
- **Toggle Ref**: Enable/disable reference image input
- **Input Image(s)**: One or two reference images (when Toggle Ref is enabled)
- **LoRA List**: Enable/disable LoRAs with adjustable strength

## LoRA Integration

LoRAs are automatically merged into prompts during generation using the format `<name:weight>`. The LoRA tags are applied internally without appearing in the prompt field.

## License

See the [LICENSE](LICENSE) file for details.