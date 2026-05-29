# main.py - Entry Point Implementation

## Full Source Code Context
```python
import gradio as gr
from views.zit_view import create_zit_tab
from views.flux2_klein_9b_view import create_flux2_klein_9b_tab
```

## CSS Styling
The `css` variable contains custom CSS for the entire app:

### Background Gradients
- Dark mode: `linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%)`
- Light mode: `linear-gradient(135deg, #e8eaf6 0%, #c5cae9 50%, #bbdefb 100%)`

### Button Styles
- Primary (Generate): `linear-gradient(135deg, #667eea 0%, #764ba2 100%)` with purple glow shadow
- Stop: `linear-gradient(135deg, #f093fb 0%, #f5576c 100%)` with pink glow shadow
- Both have hover effects with `translateY(-2px)` and enhanced shadows

### Input Field Styling
- Dark mode: semi-transparent white backgrounds with subtle borders
- Light mode: semi-transparent white backgrounds with subtle borders
- All have `border-radius: 8px`

### Accordion Styling
- Dark: `rgba(255, 255, 255, 0.03)` background with subtle border
- Light: `rgba(255, 255, 255, 0.5)` background with subtle border

### Image Styling
- Both modes: `border-radius: 8px` with subtle border

### LoRA Table Styling
- `.lora-row`: flex layout with aligned items
- `.lora-row *`: removes outline/shadow/border on all child elements
- `.lora-remove-btn`: red background, 30x30px size, centered content

### Image Flex Row
- `.image-flex-row`: flex container with 10px gap, children flex:1

## Blocks Setup
```python
if __name__ == "__main__":
    with gr.Blocks(title="🍥 Ramen", analytics_enabled=False) as demo:
        gr.Markdown("# 🍥 Ramen")
        
        with gr.Tabs() as tabs:
            create_zit_tab()
            create_flux2_klein_9b_tab()
    
    demo.launch(css=css, pwa=True)
```

### Key Details
- Title: "🍥 Ramen"
- Analytics disabled
- PWA mode enabled
- Two tabs created via imported view functions
- Global CSS passed to `demo.launch()`
