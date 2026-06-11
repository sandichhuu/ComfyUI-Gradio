import gradio as gr
from workers.ltx23_worker import generate as ltx23_generate
import tempfile
from PIL import Image as PILImage


def generate_image(prompt_text, img1, duration, fps, image_compression):
    img1_path = None

    if img1 is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        PILImage.fromarray(img1).save(tmp.name)
        img1_path = tmp.name

    result = ltx23_generate(
        prompt_text=prompt_text,
        image_path=img1_path,
        duration=float(duration),
        fps=int(fps),
        image_compression=int(image_compression)
    )

    return result

def toggle_generation():
    return gr.update(visible=False), gr.update(visible=True)

def stop_generation():
    return gr.update(visible=True), gr.update(visible=False)

def create_ltx23_tab():
    with gr.Tab("LTX23"):
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value="helloworld", scale=4, lines=1)
            duration = gr.Number(label="Duration (seconds)", value=5)
            fps = gr.Number(label="FPS", value=9, precision=0)
            image_compression = gr.Number(label="Image Compression", value=18, precision=0, minimum=0, maximum=100)
            generate_btn = gr.Button("Generate", variant="primary", scale=1)
            stop_btn = gr.Button("Stop", variant="stop", scale=1, visible=False)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input Image", height=444)

            with gr.Column(scale=1):
                output_video = gr.Video(label="Output Video", height=444)

        generate_btn.click(
            fn=toggle_generation,
            outputs=[generate_btn, stop_btn],
        ).then(
            fn=generate_image,
            inputs=[prompt, input_image, duration, fps, image_compression],
            outputs=[output_video],
        ).then(
            fn=stop_generation,
            outputs=[generate_btn, stop_btn],
        )

        stop_btn.click(
            fn=stop_generation,
            outputs=[generate_btn, stop_btn],
        )
