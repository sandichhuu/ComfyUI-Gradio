# workers/flux2_klein_9b_worker.py - Flux2-Klein-9B Worker Implementation

## Purpose
Extracted from ComfyUI. Executes the Flux2-Klein-9B workflow with reference image support.

## Bootstrap Functions
Identical to ZIT worker: `get_comfyui_path()`, `find_path()`, `add_comfyui_directory_to_sys_path()`, `add_extra_model_paths()`, `bootstrap_comfyui_runtime()`, `cleanup_comfyui_runtime()`, `import_custom_nodes()`

## Workflow Definition: build_workflow()

Returns a dict of ComfyUI nodes:

| Node ID | Class Type | Purpose |
|---------|-----------|---------|
| 9 | SaveImage | Save output as "Flux2-Klein" prefix |
| 159 | LoadImage | Load input image 1 ("full.png") |
| 163 | LoadImage | Load input image 2 ("upper__gfpgan_restore.png") |
| 179 | KSamplerSelect | Select "euler" sampler |
| 180 | VAEDecode | Decode final latent |
| 181 | RandomNoise | Set noise seed |
| 182 | EmptyFlux2LatentImage | Create empty latent |
| 183 | CFGGuider | CFG guider (cfg=1) |
| 184 | Flux2Scheduler | Scheduler (4 steps) |
| 185 | GetImageSize | Get dimensions from input |
| 186 | VAELoader | Load "full_encoder_small_decoder.safetensors" |
| 187 | CLIPLoader | Load "qwen_3_8b_fp8mixed.safetensors" (flux2) |
| 189 | PathchSageAttentionKJ | SageAttention optimization |
| 190 | LoraLoader | Load base LoRA "KLEIN-Unchained-V2.safetensors" |
| 191 | UNETLoader | Load "flux-2-klein-9b-fp8.safetensors" |
| 192 | SamplerCustomAdvanced | Main sampler |
| 193 | ReferenceLatent | Negative conditioning reference |
| 194 | ReferenceLatent | Positive conditioning reference (img1) |
| 195 | VAEEncode | Encode input image 1 |
| 196 | ImageScaleToTotalPixels | Scale input image 1 to 1MP |
| 198 | ImageScaleToTotalPixels | Scale input image 2 to 1MP |
| 199 | VAEEncode | Encode input image 2 |
| 200 | ReferenceLatent | Positive conditioning reference (img2, conditional) |
| 201 | CLIPTextEncode | Encode prompt text |
| 212 | ConditioningZeroOut | Zero out conditioning for negative |
| 219 | LoraTagLoader | Process prompt + user LoRA tags |
| 220 | PrimitiveBoolean | Toggle ref (on/off) |
| 221 | ComfySwitchNode | Switch between ref on/off paths |

### Node Connections (simplified)
```
UNETLoader → LoraLoader → LoraTagLoader → PathchSageAttentionKJ → CFGGuider
CLIPLoader → LoraLoader → LoraTagLoader → CLIPTextEncode → ReferenceLatent (img1)
                                                ↓
LoadImage(1) → ImageScaleToTotalPixels → VAEEncode → ReferenceLatent → ComfySwitchNode
LoadImage(2) → ImageScaleToTotalPixels → VAEEncode → ReferenceLatent (conditional on toggle_ref)
                                                        ↓
ReferenceLatent (negative, with ConditioningZeroOut) → CFGGuider
                                                        ↓
RandomNoise + CFGGuider + KSamplerSelect + Flux2Scheduler + EmptyFlux2LatentImage → SamplerCustomAdvanced
                                                        ↓
VAEDecode (with VAELoader) → SaveImage
```

### Reference Image Logic
- `PrimitiveBoolean` (node 220) controls whether ref images are used
- `ComfySwitchNode` (node 221) switches between:
  - `on_false`: only img1 reference (node 194 output)
  - `on_true`: both img1 and img2 references (node 200 output)

## generate() Function

### Signature
```python
def generate(
    prompt_text: str,
    seed: int,
    width: int,
    height: int,
    toggle_ref: bool,
    input_img1_path: str,
    input_img2_path: str | None = None,
    unload_models: bool | None = None,
) -> str | None:
```

### Execution Flow
1. Bootstrap ComfyUI runtime
2. Add extra model paths
3. Import custom nodes
4. **Copy input images** to ComfyUI input directory:
   ```python
   input_dir = folder_paths.get_input_directory()
   shutil.copy2(input_img1_path, os.path.join(input_dir, img1_basename))
   ```
5. **Handle dimensions**:
   - If width=0 and height=0: use original image dimensions
   - Resize both images to (width, height) using LANCZOS
6. **Calculate megapixels**: `(width * height) / 1000000.0`
7. Import ComfyUI node classes
8. Instantiate and execute nodes in order:
   - LoadImage for both inputs
   - Scale images to total pixels
   - VAEEncode both images
   - Load models (VAE, CLIP, UNET, base LoRA)
   - Process prompt with LoraTagLoader
   - CLIPTextEncode
   - PathchSageAttentionKJ (sageattn_qk_int8_pv_fp16_cuda)
   - ReferenceLatent chains
   - ComfySwitchNode (toggle_ref controls path)
   - ConditioningZeroOut for negative
   - CFGGuider (cfg=1)
   - GetImageSize → Flux2Scheduler → EmptyFlux2LatentImage
   - SamplerCustomAdvanced
   - VAEDecode
   - SaveImage
9. Find most recent "Flux2-Klein_*.png" in output directory
10. Return filepath
11. Finally: cleanup ComfyUI runtime

### Key Parameters
- Sampler: euler
- Steps: 4
- CFG: 1
- Base LoRA: `Flux.2 Klein 9B-base/KLEIN-Unchained-V2.safetensors` (strength 0.6/1.0)
- UNET: `Flux.2 Klein 9B/flux-2-klein-9b-fp8.safetensors`
- CLIP: `qwen_3_8b_fp8mixed.safetensors` (type: flux2)
- VAE: `full_encoder_small_decoder.safetensors`
- SageAttention: `sageattn_qk_int8_pv_fp16_cuda`

### Image Handling
- Input images are copied to ComfyUI's input directory
- Both images are resized to target dimensions using LANCZOS resampling
- If width=0 and height=0, original dimensions are preserved
- Megapixels parameter dynamically calculated for ImageScaleToTotalPixels nodes
