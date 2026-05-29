# views/flux2_klein_9b_view.py - Flux2-Klein-9B Tab Implementation

## Purpose
Renders the Flux2-Klein-9B tab UI with reference image support and handles user interactions.

## Constants
```python
MAX_LORAS = 6
LORA_PATTERN = re.compile(r"^<[^:]+:[\d.]+>$")
```

## Imports (differs from ZIT view)
```python
import tempfile
import os
import random
from PIL import Image as PILImage
```

## LoRA Helper Functions
Identical to ZIT view: `update_chk()`, `update_txt()`, `remove_lora_row()`, `add_lora_row()`, `merge_loras_into_prompt()`

## generate_image(prompt_text, seed_val, w, h, img1, img2, current_loras)

1. If seed is 0, generate random seed
2. Validate LoRA tags
3. Merge LoRA tags into prompt
4. Convert input images (numpy arrays from Gradio) to temp PNG files:
   ```python
   if img1 is not None:
       tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
       PILImage.fromarray(img1).save(tmp.name)
       img1_path = tmp.name
   ```
5. Determine `ref_enabled = img2 is not None`
6. Call `flux2_generate()` with all parameters
7. Return image + seed label

## UI Layout: create_flux2_klein_9b_tab()

```
Row 1: [Prompt (scale=4)] [Generate (scale=1)] [Stop (scale=1, hidden)]
Row 2: [Seed] [Width (0=same as input)] [Height (0=same as input)]
Row 3: [LoRA Column (scale=1)] [Image Column (scale=1)] [Output Column (scale=1)]
```

### Row 1
- Same as ZIT tab

### Row 2
- `gr.Number`: label="Seed", value=0, precision=0
- `gr.Number`: label="Width (0 = same as input)", value=0, precision=0
- `gr.Number`: label="Height (0 = same as input)", value=0, precision=0
- Note: Width/Height default to 0, meaning "use input image dimensions"

### Row 3 - Left Column (LoRAs)
- Identical to ZIT tab LoRA section

### Row 3 - Middle Column (Input Images)
- `gr.Row(elem_classes="image-flex-row")` containing:
  - `gr.Image(label="Input Image", height=444)` - always visible
  - `gr.Image(label="Ref Image", height=444)` - always visible (but only used if img2 is not None)

### Row 3 - Right Column (Output)
- `gr.Image(label="Output", height=444)`

## Key Differences from ZIT View
1. Has 3 columns instead of 2 (LoRAs | Input Images | Output)
2. `generate_image` accepts `img1` and `img2` parameters
3. Images are converted from numpy arrays to temp files before passing to worker
4. `ref_enabled` is determined by whether `img2` is not None
5. Width/Height default to 0 (auto-detect from input image)

## Event Wiring
Same pattern as ZIT but with additional inputs:
```python
generate_btn.click(
    fn=toggle_generation,
    outputs=[generate_btn, stop_btn],
).then(
    fn=generate_image,
    inputs=[prompt, seed, width, height, input_image1, input_image2, lora_state],
    outputs=[output_image, seed],
).then(
    fn=stop_generation,
    outputs=[generate_btn, stop_btn],
)
```
