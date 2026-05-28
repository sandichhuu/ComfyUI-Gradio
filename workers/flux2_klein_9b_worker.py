"""Flux2 Klein 9B Worker - Image generation worker for Flux2-Klein-9B"""

import threading
from typing import Optional


def generate(
    prompt: str,
    seed: int,
    width: int,
    height: int,
    loras: list[dict],
    input_img1: Optional = None,
    input_img2: Optional = None,
    use_ref: bool = False,
    stop_event: Optional[threading.Event] = None,
):
    """
    Generate image using Flux2-Klein-9B model.
    
    Args:
        prompt: The text prompt for image generation
        seed: Random seed for reproducibility
        width: Output image width
        height: Output image height
        loras: List of LoRA configs with 'name', 'weight', 'enabled'
        input_img1: First input/reference image (optional)
        input_img2: Second input/reference image (optional, used when use_ref is True)
        use_ref: Whether to use reference images
        stop_event: Threading event to check for stop signal
    
    Returns:
        Generated image (numpy array or PIL Image)
    """
    # Merge LoRA tags into prompt (not visible in UI)
    full_prompt = prompt
    for lora in loras:
        if lora.get('enabled', True):
            name = lora.get('name', '')
            weight = lora.get('weight', 1.0)
            if name:
                full_prompt += f" <{name}:{weight}>"
    
    # Simulate image generation (replace with actual ComfyUI integration)
    from PIL import Image
    import numpy as np
    
    # Check for stop signal periodically
    if stop_event and stop_event.is_set():
        return None
    
    # Generate a placeholder image (replace with actual generation)
    img = Image.new('RGB', (width, height), color=(100, 149, 237))  # Cornflower blue for Flux
    
    return np.array(img)
