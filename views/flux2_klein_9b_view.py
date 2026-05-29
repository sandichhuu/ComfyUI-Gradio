import re
import gradio as gr
from workers.flux2_klein_9b_worker import generate as flux2_generate

MAX_LORAS = 10
LORA_PATTERN = re.compile(r"^<[^:]+:[\d.]+>$")


def create_flux2_klein_9b_tab():
    with gr.Tab("Flux2-Klein-9B"):
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                value="helloworld",
                scale=4,
                lines=1,
            )
            generate_btn = gr.Button("Generate", variant="primary", scale=1)
            stop_btn = gr.Button("Stop", variant="stop", scale=1, visible=False)

        with gr.Row():
            seed = gr.Number(label="Seed", value=0, precision=0)
            width = gr.Number(label="Width (0 = same as input)", value=0, precision=0)
            height = gr.Number(label="Height (0 = same as input)", value=0, precision=0)

        with gr.Row():
            with gr.Column(scale=1):
                lora_count = gr.State(0)
                with gr.Accordion("LoRAs", open=False):
                    add_lora_btn = gr.Button("Add LoRA", size="sm")
                    lora_rows = []
                    for i in range(MAX_LORAS):
                        with gr.Row(visible=False, elem_classes="lora-row") as row:
                            lora_toggle = gr.Checkbox(
                                value=True,
                                show_label=False,
                                elem_classes="lora-toggle",
                            )
                            lora_input = gr.Textbox(
                                placeholder="<name:weight>",
                                show_label=False,
                                scale=4,
                            )
                            remove_btn = gr.Button(
                                "X",
                                scale=0,
                                elem_classes="lora-remove-btn",
                            )
                        lora_rows.append((row, lora_toggle, lora_input, remove_btn))

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row(elem_classes="image-flex-row"):
                    input_image1 = gr.Image(label="Input Image", height=444)
                    input_image2 = gr.Image(label="Ref Image", height=444)
            with gr.Column(scale=1):
                output_image = gr.Image(label="Output", height=444)

        for i, (row, lora_toggle, lora_input, remove_btn) in enumerate(lora_rows):

            def make_remove_handler(idx):
                def handler(count, *values):
                    new_count = count - 1
                    row_updates = []
                    for r_idx in range(MAX_LORAS):
                        if r_idx < idx:
                            row_updates.append(gr.update())
                        elif r_idx == idx:
                            row_updates.append(gr.update(visible=False))
                        elif r_idx < count - 1:
                            row_updates.append(gr.update(visible=True))
                        elif r_idx < count:
                            row_updates.append(gr.update(visible=False))
                        else:
                            row_updates.append(gr.update())
                    return (new_count,) + tuple(row_updates)

                return handler

            remove_btn.click(
                fn=make_remove_handler(i),
                inputs=[lora_count] + [row_data[2] for row_data in lora_rows],
                outputs=[lora_count] + [row_data[0] for row_data in lora_rows],
            )

        def add_lora_click(count, *values):
            if count >= MAX_LORAS:
                return (count,) + tuple(gr.update() for _ in range(MAX_LORAS))
            row_updates = []
            for i in range(MAX_LORAS):
                if i == count:
                    row_updates.append(gr.update(visible=True))
                else:
                    row_updates.append(gr.update())
            return (count + 1,) + tuple(row_updates)

        add_lora_btn.click(
            fn=add_lora_click,
            inputs=[lora_count] + [row_data[2] for row_data in lora_rows],
            outputs=[lora_count] + [row_data[0] for row_data in lora_rows],
        )

        def toggle_generation():
            return gr.update(visible=False), gr.update(visible=True)

        def stop_generation():
            return gr.update(visible=True), gr.update(visible=False)

        def merge_loras_into_prompt(prompt_text, lora_tags):
            if lora_tags:
                return prompt_text + "\n" + "\n".join(lora_tags)
            return prompt_text

        def generate_image(
            prompt_text,
            seed_val,
            w,
            h,
            img1,
            img2,
            count,
            *values,
        ):
            input_seed = int(seed_val)
            final_seed = input_seed
            if final_seed == 0:
                import random

                final_seed = random.randint(1, 2**64)

            lora_tags = []
            for i in range(count):
                enabled = values[i * 2]
                tag = values[i * 2 + 1].strip()
                if tag and not LORA_PATTERN.match(tag):
                    gr.Warning(
                        f"LoRA #{i + 1} wrong format: '{tag}'. "
                        f"Expected format: <name:weight>"
                    )
                    return gr.update(), gr.update(label="Seed")
                if enabled and tag:
                    lora_tags.append(tag)

            merged_prompt = merge_loras_into_prompt(prompt_text, lora_tags)

            img1_path = None
            img2_path = None

            if img1 is not None:
                import tempfile
                import os
                from PIL import Image as PILImage

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                PILImage.fromarray(img1).save(tmp.name)
                img1_path = tmp.name

            if img2 is not None:
                import tempfile
                import os
                from PIL import Image as PILImage

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

        all_lora_inputs = [lora_count]
        for row_data in lora_rows:
            all_lora_inputs.append(row_data[1])
            all_lora_inputs.append(row_data[2])

        generate_btn.click(
            fn=toggle_generation,
            outputs=[generate_btn, stop_btn],
        ).then(
            fn=generate_image,
            inputs=[prompt, seed, width, height, input_image1, input_image2]
            + all_lora_inputs,
            outputs=[output_image, seed],
        ).then(
            fn=stop_generation,
            outputs=[generate_btn, stop_btn],
        )

        stop_btn.click(
            fn=stop_generation,
            outputs=[generate_btn, stop_btn],
        )
