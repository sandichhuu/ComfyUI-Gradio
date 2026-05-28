"""Flux2 Klein 9B View - Gradio UI for Flux2-Klein-9B"""

import gradio as gr
import random
import threading
from workers.flux2_klein_9b_worker import generate


def create_flux2_klein_9b_tab():
    """Create the Flux2-Klein-9B tab with all UI components."""
    
    # Global state for generation control (persists across browser refresh)
    generation_state = {"stop_event": None, "thread": None, "is_generating": False}
    
    with gr.Tab("Flux2-Klein-9B"):
        # State for LoRA items
        lora_state = gr.State([])
        
        # Row 1: Prompt and Generate/Stop button
        with gr.Row():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="helloworld",
                scale=4
            )
            generate_btn = gr.Button("Generate", variant="primary", scale=1)
        
        # Row 2: Config (seed with random toggle, width, height, toggle_ref)
        with gr.Row():
            randomize_seed = gr.Checkbox(label="Random Seed", value=True)
            seed_input = gr.Number(label="Seed", value=None, interactive=False, precision=0)
            width_input = gr.Number(label="Width", value=512, precision=0)
            height_input = gr.Number(label="Height", value=512, precision=0)
            use_ref_toggle = gr.Checkbox(label="Use Reference Images", value=False)
        
        # Row 3: Input images (conditional based on toggle_ref)
        with gr.Row() as input_images_row:
            with gr.Column(scale=1):
                input_img1 = gr.Image(label="Input Image 1", type="numpy")
            with gr.Column(scale=1, visible=False) as input_img2_col:
                input_img2 = gr.Image(label="Input Image 2", type="numpy")
        
        def toggle_input_images(use_ref):
            """Toggle visibility of second input image based on use_ref."""
            return gr.update(visible=use_ref)
        
        # Row 4: LoRA list (collapsible) and output image
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    with gr.Accordion("LoRA List", open=True):
                        add_lora_btn = gr.Button("+ Add LoRA", size="sm")
                        lora_display = gr.HTML()
            
            with gr.Column(scale=2):
                output_image = gr.Image(label="Output Image", type="numpy")
        
        def toggle_random_seed(randomize):
            """Toggle seed input based on randomize checkbox."""
            new_seed = None if randomize else random.randint(0, 2**31)
            return gr.update(interactive=not randomize, value=new_seed)
        
        def get_lora_html(loras):
            """Generate HTML for LoRA list."""
            if not loras:
                return "<p>No LoRA items added.</p>"
            
            html = '<div style="display: flex; flex-direction: column; gap: 8px;">'
            for i, lora in enumerate(loras):
                enabled = 'checked' if lora.get('enabled', True) else ''
                name = lora.get('name', '')
                weight = lora.get('weight', 1.0)
                html += f'''
                <div style="display: flex; align-items: center; gap: 8px;" data-lora-index="{i}">
                    <input type="checkbox" class="lora-enabled" {enabled} data-index="{i}" style="flex: 0 0 auto;">
                    <input type="text" class="lora-name" value="{name}" placeholder="LoRA name" data-index="{i}" style="flex: 2;">
                    <input type="number" class="lora-weight" value="{weight}" step="0.1" min="-2" max="2" data-index="{i}" style="flex: 1;">
                    <button class="lora-remove" data-index="{i}" style="flex: 0 0 auto;">🗑️</button>
                </div>
                '''
            html += '</div>'
            return html
        
        def add_lora(loras):
            """Add a new LoRA item."""
            return loras + [{"name": "", "weight": 1.0, "enabled": True}]
        
        def update_lora_display(loras):
            """Update the LoRA display HTML."""
            return get_lora_html(loras)
        
        def start_generation(prompt, seed, randomize, width, height, use_ref, img1, img2, loras):
            """Start image generation."""
            # Apply fix-height if needed
            if height > 444:
                scale = 444 / height
                width = int(width * scale)
                height = 444
            
            # Prepare LoRA list
            processed_loras = []
            for lora in loras:
                processed_loras.append({
                    'name': lora.get('name', ''),
                    'weight': lora.get('weight', 1.0),
                    'enabled': lora.get('enabled', True)
                })
            
            # Use random seed if enabled
            if randomize:
                seed = random.randint(0, 2**31)
            
            # Create stop event
            stop_event = threading.Event()
            generation_state["stop_event"] = stop_event
            generation_state["is_generating"] = True
            
            result = [None]
            
            def run_generation():
                try:
                    result[0] = generate(
                        prompt=prompt,
                        seed=seed,
                        width=int(width),
                        height=int(height),
                        loras=processed_loras,
                        input_img1=img1,
                        input_img2=img2 if use_ref else None,
                        use_ref=use_ref,
                        stop_event=stop_event
                    )
                except Exception as e:
                    print(f"Generation error: {e}")
                    result[0] = None
                finally:
                    generation_state["is_generating"] = False
            
            generation_thread = threading.Thread(target=run_generation)
            generation_state["thread"] = generation_thread
            generation_thread.start()
            
            # Wait for completion
            while generation_thread.is_alive():
                generation_thread.join(timeout=0.1)
            
            if result[0] is None:
                # Return placeholder if generation failed or was stopped
                import numpy as np
                result[0] = np.zeros((int(height), int(width), 3), dtype=np.uint8)
            
            return result[0], seed
        
        def handle_generate_click(prompt, seed, randomize, width, height, use_ref, img1, img2, loras, current_btn_label):
            """Handle generate/stop button click."""
            if generation_state.get("is_generating", False):
                # Stop generation
                if generation_state.get("stop_event"):
                    generation_state["stop_event"].set()
                return None, seed, "Generate", gr.update(variant="primary")
            else:
                # Start generation
                result, new_seed = start_generation(prompt, seed, randomize, width, height, use_ref, img1, img2, loras)
                return result, new_seed, "Generate", gr.update(variant="primary")
        
        # Event handlers
        randomize_seed.change(
            fn=toggle_random_seed,
            inputs=[randomize_seed],
            outputs=[seed_input]
        )
        
        use_ref_toggle.change(
            fn=toggle_input_images,
            inputs=[use_ref_toggle],
            outputs=[input_img2_col]
        )
        
        add_lora_btn.click(
            fn=add_lora,
            inputs=[lora_state],
            outputs=[lora_state]
        ).then(
            fn=update_lora_display,
            inputs=[lora_state],
            outputs=[lora_display]
        )
        
        generate_btn.click(
            fn=handle_generate_click,
            inputs=[prompt_input, seed_input, randomize_seed, width_input, height_input, use_ref_toggle, input_img1, input_img2, lora_state, generate_btn],
            outputs=[output_image, seed_input, generate_btn]
        )
    
    return None
