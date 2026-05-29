# workers/zit_worker.py - ZIT Worker Implementation

## Purpose
Extracted from ComfyUI's "Save as script" feature. Executes the Z-Image-Turbo workflow programmatically.

## ComfyUI Bootstrap Functions

### get_comfyui_path()
1. Check `COMFYUI_PATH` environment variable
2. If not set, call `find_path("ComfyUI")` to search parent directories recursively

### find_path(name, path=None)
- Starts from `os.getcwd()` if path is None
- Checks if `name` exists in `os.listdir(path)`
- Recurses into parent directory until found or root reached
- Returns full path or None

### add_comfyui_directory_to_sys_path()
- Gets ComfyUI path, adds to `sys.path[0]`
- Removes first if already present (to ensure fresh position)

### add_extra_model_paths()
- Tries to import `load_extra_path_config` from ComfyUI's `main.py` or `utils.extra_config`
- Finds and loads `extra_model_paths.yaml` via recursive search

### bootstrap_comfyui_runtime()
1. Add ComfyUI to sys.path
2. Import and enable `comfy.options` args parsing
3. On Windows: set `MIMALLOC_PURGE_DELAY=0`
4. Handle `--default-device`, `--cuda-device`, `--oneapi-device-selector` args
5. Handle `--deterministic` CUBLAS workspace config
6. Import `cuda_malloc` and handle ROCm-specific env vars

### cleanup_comfyui_runtime(unload_models=None)
1. Check `COMFYUI_TOPYTHON_UNLOAD_MODELS` env var if unload_models is None
2. Run cleanup hooks: `cleanup_models_gc`, `unload_all_models` (if enabled), `soft_empty_cache`
3. Run `gc.collect()`

### import_custom_nodes()
1. Ensure ComfyUI is in sys.path
2. Import `asyncio`, `execution`, `nodes.init_extra_nodes`
3. Create new event loop, set it
4. Create `PromptServer` instance
5. Create `PromptQueue`
6. Run `init_extra_nodes()` via asyncio

## Workflow Definition: build_workflow()

Returns a dict of ComfyUI nodes:

| Node ID | Class Type | Purpose |
|---------|-----------|---------|
| 9 | SaveImage | Save output as "zit" prefix |
| 57:29 | VAELoader | Load "ae.safetensors" |
| 57:33 | ConditioningZeroOut | Create negative conditioning |
| 57:8 | VAEDecode | Decode latent to image |
| 57:11 | ModelSamplingAuraFlow | Patch model with shift=3 |
| 57:3 | KSampler | Main sampler (8 steps, cfg=1, res_multistep, simple) |
| 57:13 | EmptySD3LatentImage | Create empty latent |
| 57:28 | UNETLoader | Load "ZImageTurbo/z_image_turbo_bf16.safetensors" |
| 57:30 | CLIPLoader | Load "qwen_3_4b.safetensors" (lumina2) |
| 57:66 | LoraTagLoader | Process prompt + LoRA tags |
| 57:27 | CLIPTextEncode | Encode text to conditioning |

### Node Connections (simplified)
```
UNETLoader → LoraTagLoader → ModelSamplingAuraFlow → KSampler
CLIPLoader → LoraTagLoader → CLIPTextEncode → KSampler (+ ConditioningZeroOut → KSampler)
EmptySD3LatentImage → KSampler → VAEDecode (with VAELoader) → SaveImage
```

## generate() Function

### Signature
```python
def generate(
    prompt_text: str,
    seed: int,
    width: int,
    height: int,
    unload_models: bool | None = None,
) -> str | None:
```

### Execution Flow
1. Bootstrap ComfyUI runtime
2. Add extra model paths
3. Import custom nodes
4. Import ComfyUI node classes
5. Enter `torch.inference_mode()` context
6. Instantiate and execute nodes in order:
   - VAELoader → load VAE
   - EmptySD3LatentImage → create latent with width/height
   - UNETLoader → load diffusion model
   - CLIPLoader → load text encoder
   - LoraTagLoader → process prompt with LoRA tags
   - CLIPTextEncode → encode text
   - ModelSamplingAuraFlow → patch model
   - ConditioningZeroOut → create negative conditioning
   - KSampler → sample (8 steps, cfg=1)
   - VAEDecode → decode to image
   - SaveImage → save to output directory
7. Find most recent "zit_*.png" in output directory
8. Return filepath
9. Finally: cleanup ComfyUI runtime

### Key Parameters in Workflow
- Sampler: `res_multistep`
- Scheduler: `simple`
- Steps: 8
- CFG: 1
- Denoise: 1
- ModelSamplingAuraFlow shift: 3
- UNET: `ZImageTurbo/z_image_turbo_bf16.safetensors`
- CLIP: `qwen_3_4b.safetensors` (type: lumina2)
- VAE: `ae.safetensors`
