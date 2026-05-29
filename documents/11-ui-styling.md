# UI Styling Reference

## Global CSS (main.py)

### Container
```css
.gradio-container {
    min-height: 100vh;
}
```

### Backgrounds
```css
.dark .gradio-container {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
}
.light .gradio-container {
    background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 50%, #bbdefb 100%);
}
```

### Block Transparency
```css
.gr-block {
    background: transparent !important;
}
```

### Buttons
```css
.gr-button {
    border: none;
    color: white;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: bold;
    transition: all 0.3s ease;
}

button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}
button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
}

button.stop {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
}
button.stop:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(245, 87, 108, 0.6);
}
```

### Input Fields
```css
.dark .gr-textbox, .dark .gr-number, .dark .gr-checkbox {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
}
.light .gr-textbox, .light .gr-number, .light .gr-checkbox {
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 8px;
}
```

### Accordions
```css
.dark .gr-accordion {
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.03);
}
.light .gr-accordion {
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.5);
}
```

### Images
```css
.dark .gr-image {
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
}
.light .gr-image {
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 8px;
}
```

### Image Flex Row (Flux2 input images)
```css
.image-flex-row {
    display: flex;
    gap: 10px;
}
.image-flex-row > div {
    flex: 1;
    min-width: 0;
}
```

### LoRA Row Styling
```css
.lora-row {
    align-items: center;
    padding: 4px 0;
    gap: 6px;
}
.lora-row * {
    outline: none !important;
    box-shadow: none !important;
    border-color: transparent !important;
}
.lora-row input,
.lora-row textarea,
.lora-row button {
    outline: none !important;
    box-shadow: none !important;
}
.lora-row > div {
    padding-top: 0 !important;
    min-height: 0 !important;
    align-items: center;
}
.lora-remove-btn {
    background: #e53935 !important;
    min-width: 30px;
    max-width: 30px;
    min-height: 30px;
    max-height: 30px;
    font-size: 14px;
    font-weight: bold;
    justify-content: center;
    align-self: center;
}
```

## Layout Scale System
Gradio's `scale` parameter controls relative sizing:
- Prompt textbox: `scale=4` (takes 4x space)
- Generate/Stop buttons: `scale=1`
- LoRA columns: `scale=1` each
- Image columns: `scale=1` each
