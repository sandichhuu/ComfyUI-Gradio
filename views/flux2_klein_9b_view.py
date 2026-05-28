import gradio as gr
import random
from workers.flux2_klein_9b_worker import generate


def create_flux2_klein_9b_tab():
    """Create the Flux2-Klein-9B tab UI."""
    
    with gr.Row():
        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="helloworld",
            lines=2,
            scale=4
        )
        generate_btn = gr.Button("Generate", variant="primary", scale=1)
    
    with gr.Row():
        seed_input = gr.Number(label="Seed", value=random.randint(1, 2**32), precision=0)
        random_seed_btn = gr.Button("🎲", size="sm")
        width_input = gr.Slider(label="Width", minimum=256, maximum=2048, step=64, value=1024)
        height_input = gr.Slider(label="Height", minimum=256, maximum=2048, step=64, value=1024)
        steps_input = gr.Slider(label="Steps", minimum=1, maximum=50, step=1, value=4)
        toggle_ref = gr.Checkbox(label="Use Reference Images", value=False)
    
    with gr.Row() as input_images_row:
        input_img1 = gr.Image(label="Input Image 1", type="numpy", height=444)
        input_img2 = gr.Image(label="Input Image 2", type="numpy", height=444, visible=False)
    
    # LoRA section with collapse
    with gr.Accordion("LoRA Settings", open=False):
        lora_container = gr.HTML()
        add_lora_btn = gr.Button("Add LoRA", size="sm")
        
        # Hidden state for LoRA list
        lora_state = gr.State([])
        
        def render_lora_list(loras):
            html = '<div style="display: flex; flex-direction: column; gap: 8px;">'
            for i, (name, weight, enabled) in enumerate(loras):
                html += f'''
                <div style="display: flex; align-items: center; gap: 8px; padding: 8px; background: #f5f5f5; border-radius: 4px;">
                    <input type="checkbox" id="lora_enabled_{i}" {"checked" if enabled else ""} style="width: 16px; height: 16px;">
                    <input type="text" id="lora_name_{i}" value="{name}" placeholder="LoRA name" style="flex: 1; padding: 4px;">
                    <input type="number" id="lora_weight_{i}" value="{weight}" step="0.1" min="0" max="2" style="width: 80px; padding: 4px;">
                    <button onclick="removeLora({i})" style="padding: 4px 8px; background: #ff4444; color: white; border: none; border-radius: 4px; cursor: pointer;">Remove</button>
                </div>
                '''
            html += '</div>'
            return html
        
        def add_lora(loras):
            loras.append(("", 1.0, True))
            return loras, render_lora_list(loras)
        
        def remove_lora(index, loras):
            if 0 <= index < len(loras):
                loras.pop(index)
            return loras, render_lora_list(loras)
        
        add_lora_btn.click(
            fn=add_lora,
            inputs=[lora_state],
            outputs=[lora_state, lora_container]
        )
    
    output_image = gr.Image(label="Output Image", height=444)
    
    # Toggle reference images visibility
    def toggle_ref_visibility(use_ref):
        return gr.update(visible=use_ref), gr.update(visible=use_ref)
    
    toggle_ref.change(
        fn=toggle_ref_visibility,
        inputs=[toggle_ref],
        outputs=[input_img1, input_img2]
    )
    
    # Event handlers
    def toggle_random_seed():
        return random.randint(1, 2**32)
    
    random_seed_btn.click(
        fn=toggle_random_seed,
        outputs=[seed_input]
    )
    
    # Generate function
    def do_generate(prompt, seed, width, height, steps, loras, use_ref, img1, img2):
        # Filter enabled LoRAs
        enabled_loras = [(name, weight) for name, weight, enabled in loras if enabled]
        
        # Convert width/height to int
        width = int(width)
        height = int(height)
        steps = int(steps)
        seed = int(seed)
        
        try:
            result = generate(
                prompt_text=prompt,
                seed=seed,
                width=width,
                height=height,
                steps=steps,
                loras=enabled_loras,
                toggle_ref=use_ref,
                input_img1=img1,
                input_img2=img2 if use_ref else None
            )
            return result
        except Exception as e:
            print(f"Generation error: {e}")
            return None
    
    generate_btn.click(
        fn=do_generate,
        inputs=[prompt_input, seed_input, width_input, height_input, steps_input, lora_state, toggle_ref, input_img1, input_img2],
        outputs=[output_image]
    )
    
    return gr.TabItem("Flux2-Klein-9B")
