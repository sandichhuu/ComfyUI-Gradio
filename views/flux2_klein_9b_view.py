import re
import gradio as gr
from workers.flux2_klein_9b_worker import generate as flux2_generate
import tempfile
import os
import random
from PIL import Image as PILImage

MAX_LORAS = 6
LORA_PATTERN = re.compile(r"^<[^:]+:[\d.]+>$")

def update_chk(state_list, idx, val):
    idx = int(idx)
    if 0 <= idx < len(state_list):
        state_list[idx]["enabled"] = val
    return state_list

def update_txt(state_list, idx, val):
    idx = int(idx)
    if 0 <= idx < len(state_list):
        state_list[idx]["tag"] = val.strip()
    return state_list

def remove_lora_row(state_list, idx_to_remove):
    idx = int(idx_to_remove)
    if 0 <= idx < len(state_list):
        return state_list[:idx] + state_list[idx+1:]
    return state_list

def add_lora_row(current_state):
    if len(current_state) >= MAX_LORAS:
        gr.Warning(f"Max allowance {MAX_LORAS} LoRAs!")
        return current_state
    return current_state + [{"enabled": True, "tag": ""}]

def merge_loras_into_prompt(prompt_text, lora_tags):
    if lora_tags:
        return prompt_text + "\n" + "\n".join(lora_tags)
    return prompt_text

def generate_image(prompt_text, seed_val, w, h, img1, img2, current_loras):
    input_seed = int(seed_val)
    final_seed = input_seed
    if final_seed == 0:
        final_seed = random.randint(1, 2**64)

    lora_tags = []
    for item in current_loras:
        enabled = item.get("enabled", False)
        tag = item.get("tag", "").strip()
        if enabled and tag:
            if not LORA_PATTERN.match(tag):
                gr.Warning(f"Invalid LoRA format: '{tag}'. Expected format: <name:weight>")
                return gr.update(), gr.update(label="Seed")
            lora_tags.append(tag)

    merged_prompt = merge_loras_into_prompt(prompt_text, lora_tags)

    img1_path = None
    img2_path = None

    if img1 is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        PILImage.fromarray(img1).save(tmp.name)
        img1_path = tmp.name

    if img2 is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        PILImage.fromarray(img2).save(tmp.name)
        img2_path = tmp.name

    ref_enabled = img2 is not None

    result = flux2_generate(
        prompt_text=merged_prompt,
        seed=final_seed,
        width=int(w),
        height=int(h),
        toggle_ref=ref_enabled,
        input_img1_path=img1_path,
        input_img2_path=img2_path,
    )

    seed_label = f"Seed ({final_seed})" if input_seed == 0 else "Seed"
    return result, gr.update(label=seed_label)

def toggle_generation():
    return gr.update(visible=False), gr.update(visible=True)

def stop_generation():
    return gr.update(visible=True), gr.update(visible=False)

def create_flux2_klein_9b_tab():
    with gr.Tab("Flux2-Klein-9B"):
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value="helloworld", scale=4, lines=1)
            generate_btn = gr.Button("Generate", variant="primary", scale=1)
            stop_btn = gr.Button("Stop", variant="stop", scale=1, visible=False)

        with gr.Row():
            seed = gr.Number(label="Seed", value=0, precision=0)
            width = gr.Number(label="Width (0 = same as input)", value=0, precision=0)
            height = gr.Number(label="Height (0 = same as input)", value=0, precision=0)

        with gr.Row():
            with gr.Column(scale=1):
                lora_state = gr.State([])
                with gr.Accordion("LoRAs", open=True):
                    add_lora_btn = gr.Button("➕ Add LoRA", variant="secondary", size="sm")

                    @gr.render(inputs=lora_state)
                    def render_loras(state_list):
                        if not state_list:
                            gr.Markdown("*No LoRA loaded*")
                            return

                        for idx, item in enumerate(state_list):
                            with gr.Row(elem_id=f"lora-row-{idx}"):
                                chk = gr.Checkbox(value=item["enabled"], show_label=False, scale=0, min_width=40)
                                txt = gr.Textbox(
                                    value=item["tag"],
                                    placeholder="<lora_name:1.0>",
                                    show_label=False,
                                    scale=6,
                                    container=False
                                )
                                idx_holder = gr.Number(value=idx, visible=False)
                                del_btn = gr.Button("🗑️", variant="stop", scale=0, min_width=80)

                                chk.change(fn=update_chk, inputs=[lora_state, idx_holder, chk], outputs=[lora_state])
                                txt.change(fn=update_txt, inputs=[lora_state, idx_holder, txt], outputs=[lora_state])
                                del_btn.click(fn=remove_lora_row, inputs=[lora_state, idx_holder], outputs=[lora_state])

            with gr.Column(scale=1):
                with gr.Row(elem_classes="image-flex-row"):
                    input_image1 = gr.Image(label="Input Image", height=444)
                    input_image2 = gr.Image(label="Ref Image", height=444)

            with gr.Column(scale=1):
                output_image = gr.Image(label="Output", height=444)

            add_lora_btn.click(fn=add_lora_row, inputs=[lora_state], outputs=[lora_state])

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

        stop_btn.click(
            fn=stop_generation,
            outputs=[generate_btn, stop_btn],
        )