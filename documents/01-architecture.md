# Architecture: Data Flow & Component Design

## Overall Data Flow
```
User Input (Gradio UI)
    ↓
View Layer (views/*.py)
    ↓ (validates, merges LoRA tags, converts images to temp files)
Worker Layer (workers/*.py)
    ↓ (bootstraps ComfyUI runtime, builds workflow dict, executes nodes)
ComfyUI Runtime
    ↓ (returns image filepath)
View Layer (returns image to Gradio Output component)
```

## View Layer Pattern (views/*.py)

Each view module exports a single function: `create_<model>_tab()` that returns a `gr.Tab`.

### Shared Pattern in Both Views
1. Define `MAX_LORAS = 6` and `LORA_PATTERN = re.compile(r"^<[^:]+:[\d.]+>$")`
2. Helper functions: `update_chk()`, `update_txt()`, `remove_lora_row()`, `add_lora_row()`, `merge_loras_into_prompt()`
3. `generate_image()` - validates LoRA tags, merges into prompt, calls worker, returns image
4. `toggle_generation()` / `stop_generation()` - button visibility toggling
5. `create_*_tab()` - builds the UI layout with `gr.render` for dynamic LoRA list

### LoRA State Management
- Uses `gr.State([])` to hold a list of dicts: `[{"enabled": True, "tag": "<name:weight>"}, ...]`
- `@gr.render(inputs=lora_state)` dynamically renders LoRA rows
- Each row: Checkbox (enable) + Textbox (tag) + hidden Number (index) + Delete button
- Changes update the state list in-place via `update_chk` / `update_txt`

### Button Chaining Pattern
```python
generate_btn.click(
    fn=toggle_generation,          # Hide Generate, show Stop
    outputs=[generate_btn, stop_btn],
).then(
    fn=generate_image,             # Run actual generation
    inputs=[...],
    outputs=[output_image, seed],
).then(
    fn=stop_generation,            # Restore Generate, hide Stop
    outputs=[generate_btn, stop_btn],
)
```

## Worker Layer Pattern (workers/*.py)

Each worker module contains:

### 1. ComfyUI Bootstrap Functions
- `get_comfyui_path()` - checks `COMFYUI_PATH` env var, falls back to recursive parent directory search
- `add_comfyui_directory_to_sys_path()` - adds ComfyUI to Python path
- `add_extra_model_paths()` - loads `extra_model_paths.yaml` config
- `bootstrap_comfyui_runtime()` - sets up CUDA devices, mimalloc, allocator settings
- `cleanup_comfyui_runtime()` - unloads models, empties cache, runs gc
- `import_custom_nodes()` - initializes ComfyUI's event loop and custom nodes

### 2. Workflow Definition
- `build_workflow()` - returns a dict defining the ComfyUI node graph (JSON-serializable)
- `build_extra_pnginfo()` - returns workflow metadata for SaveImage node

### 3. Generate Function
```python
def generate(prompt_text, seed, width, height, ...):
    bootstrap_comfyui_runtime()
    add_extra_model_paths()
    import_custom_nodes()
    
    # Import ComfyUI nodes
    from nodes import CLIPLoader, CLIPTextEncode, KSampler, ...
    import folder_paths
    import torch
    
    try:
        with torch.inference_mode():
            # Instantiate nodes
            # Wire them together
            # Execute
            # Find output file
            return output_filepath
    finally:
        cleanup_comfyui_runtime()
```

### 4. Output File Discovery
Workers use `glob` to find the most recently created image file matching the prefix pattern:
```python
output_dir = folder_paths.get_output_directory()
pattern = os.path.join(output_dir, "zit_*.png")
files = sorted(glob.glob(pattern), key=os.path.getmtime)
return files[-1] if files else None
```

## Key Constraints
- ComfyUI server must be running before Gradio app starts
- Each generation call bootstraps the full ComfyUI runtime (heavy initialization)
- Workers use `torch.inference_mode()` context manager
- Images > threshold get height normalized to 444px (Gradio component setting)
- LoRA format: `<name:weight>` (e.g., `<lora:76N0PGDVMCA64NA75C2NW7V600:1>`)
