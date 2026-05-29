# ComfyUI Integration Details

## How Workers Import ComfyUI

Workers do NOT import ComfyUI at module level. Instead, ComfyUI is bootstrapped at runtime inside the `generate()` function:

```python
def generate(prompt_text, seed, width, height):
    bootstrap_comfyui_runtime()  # Adds ComfyUI to sys.path, sets up env vars
    add_extra_model_paths()       # Loads extra_model_paths.yaml
    import_custom_nodes()         # Initializes event loop, server, custom nodes
    
    from nodes import CLIPLoader, KSampler, ...  # NOW ComfyUI is importable
    import folder_paths
    import torch
```

## ComfyUI Path Resolution

1. Check `COMFYUI_PATH` environment variable
2. If not set, recursively search parent directories from `os.getcwd()` upward
3. Look for a directory named "ComfyUI"

## extra_model_paths.yaml

This file configures additional model directories. The worker searches for it recursively and loads it via ComfyUI's `load_extra_path_config()`.

## Runtime Bootstrap Details

### CUDA Device Configuration
```python
if args.default_device is not None:
    default_dev = args.default_device
    devices = list(range(32))
    devices.remove(default_dev)
    devices.insert(0, default_dev)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)
```

### Windows-Specific
```python
if os.name == "nt":
    os.environ["MIMALLOC_PURGE_DELAY"] = "0"
```

### ROCm-Specific
```python
import cuda_malloc
if "rocm" in cuda_malloc.get_torch_version_noimport():
    os.environ["OCL_SET_SVM_SIZE"] = "262144"
```

## Custom Node Initialization

```python
import asyncio
import execution
from nodes import init_extra_nodes
import server

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
server_instance = server.PromptServer(loop)
execution.PromptQueue(server_instance)
asyncio.run(init_extra_nodes())
```

## Cleanup Protocol

```python
import comfy.model_management as model_management

model_management.cleanup_models_gc()
model_management.unload_all_models()  # if COMFYUI_TOPYTHON_UNLOAD_MODELS enabled
model_management.soft_empty_cache()
gc.collect()
```

## Workflow Execution Pattern

The workflow is defined as a Python dict (extracted from ComfyUI's JSON format), then nodes are instantiated and executed programmatically:

```python
with torch.inference_mode():
    # 1. Load models
    vaeloader = VAELoader()
    vae = vaeloader.load_vae(vae_name="ae.safetensors")
    
    # 2. Create inputs
    emptysd3 = EmptySD3LatentImage()
    latent = emptysd3.EXECUTE_NORMALIZED(width=1024, height=1024, batch_size=1)
    
    # 3. Process prompt
    loratagloader = LoraTagLoader()
    result = loratagloader.load_lora(text=prompt_text, model=model, clip=clip)
    
    # 4. Sample
    ksampler = KSampler()
    sampled = ksampler.sample(seed=seed, steps=8, ...)
    
    # 5. Decode
    vaedecode = VAEDecode()
    decoded = vaedecode.decode(samples=sampled, vae=vae)
    
    # 6. Save
    saveimage = SaveImage()
    saveimage.save_images(filename_prefix="zit", images=decoded)
```

## Output File Discovery

Both workers use the same pattern:
```python
output_dir = folder_paths.get_output_directory()
pattern = os.path.join(output_dir, "<prefix>_*.png")
files = sorted(glob.glob(pattern), key=os.path.getmtime)
return files[-1] if files else None
```

- ZIT prefix: `"zit"`
- Flux2 prefix: `"Flux2-Klein"`
