# Image Processing Details

## Output Image Display
Both tabs use `gr.Image(label="Output", height=444)` to display results with a fixed height of 444px.

## Input Image Handling (Flux2-Klein-9B Only)

### Gradio to File Conversion
```python
import tempfile
from PIL import Image as PILImage

if img1 is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    PILImage.fromarray(img1).save(tmp.name)
    img1_path = tmp.name
```

Gradio delivers images as numpy arrays. The view converts them to temp PNG files before passing to the worker.

### Worker Image Processing
```python
input_dir = folder_paths.get_input_directory()

# Copy to ComfyUI input directory
shutil.copy2(input_img1_path, os.path.join(input_dir, img1_basename))

# Open and get original dimensions
img1 = PILImage.open(img1_dest)
orig_w, orig_h = img1.size

# Auto-detect dimensions if set to 0
if width == 0 and height == 0:
    width = orig_w
    height = orig_h

# Resize to target dimensions
img1_resized = img1.resize((width, height), PILImage.LANCZOS)
img1_resized.save(img1_dest)

# Same for image 2 if present
```

### Megapixels Calculation
```python
megapixels = (width * height) / 1000000.0
```

Used in `ImageScaleToTotalPixels` nodes to scale input images to the target resolution.

## ZIT Tab (No Input Images)
- Uses `EmptySD3LatentImage` to create a blank latent at specified width/height
- Default dimensions: 1024x1024

## Flux2-Klein-9B Tab
- Uses input images as reference
- Images are scaled to 1 megapixel total using LANCZOS resampling
- Images are encoded via VAEEncode for conditioning
- `ReferenceLatent` nodes inject image features into the sampling process

## Seed Handling
```python
input_seed = int(seed_val)
final_seed = input_seed
if final_seed == 0:
    import random
    final_seed = random.randint(1, 2**64)
```

- Seed 0 = random (generates a seed and displays it)
- Non-zero seed = used as-is

## Image Output Discovery
```python
output_dir = folder_paths.get_output_directory()
pattern = os.path.join(output_dir, "zit_*.png")  # or "Flux2-Klein_*.png"
files = sorted(glob.glob(pattern), key=os.path.getmtime)
return files[-1] if files else None
```

Most recent matching file is returned to the Gradio Image component.
