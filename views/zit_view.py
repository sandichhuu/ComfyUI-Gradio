import random
import gradio as gr
from workers.zit_worker import generate as zit_generate


def generate_image(prompt_text, w, h):
    final_seed = random.randint(1, 2**64)

    result = zit_generate(
        prompt_text=prompt_text,
        seed=final_seed,
        width=int(w),
        height=int(h),
    )

    return result


def toggle_generation():
    return gr.update(visible=False), gr.update(visible=True)


def stop_generation():
    return gr.update(visible=True), gr.update(visible=False)


def create_zit_tab():
    with gr.Tab("ZIT"):
        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                value="helloworld",
                scale=4,
                lines=1,
            )
            width = gr.Number(label="Width", value=1024, precision=0)
            height = gr.Number(label="Height", value=1024, precision=0)
            generate_btn = gr.Button("Generate", variant="primary", scale=1)
            stop_btn = gr.Button("Stop", variant="stop", scale=1, visible=False)

        with gr.Row():
            output_image = gr.Image(label="Output", height=444)

        generate_btn.click(
            fn=toggle_generation,
            outputs=[generate_btn, stop_btn],
        ).then(
            fn=generate_image,
            inputs=[prompt, width, height],
            outputs=[output_image],
        ).then(
            fn=stop_generation,
            outputs=[generate_btn, stop_btn],
        )

        stop_btn.click(
            fn=stop_generation,
            outputs=[generate_btn, stop_btn],
        )
