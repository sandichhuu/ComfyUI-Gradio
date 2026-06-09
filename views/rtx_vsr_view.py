import gradio as gr
from workers.rtx_vsr_worker import generate as rtx_vsr_generate
from views.utilities.video_info import get_video_info
import tempfile
import shutil
import os


def generate_video(video):
    if video is None:
        return None, ""

    tmp_dir = tempfile.mkdtemp()
    ext = os.path.splitext(video)[-1] if isinstance(video, str) else ".mp4"
    tmp = os.path.join(tmp_dir, f"input{ext}")

    if isinstance(video, str):
        shutil.copy(video, tmp)
    else:
        shutil.move(video, tmp)

    result = rtx_vsr_generate(
        video_path=tmp,
    )

    shutil.rmtree(tmp_dir, ignore_errors=True)
    output_info = get_video_info(result) if result else ""
    return result, output_info


def on_input_video(video):
    if video is None:
        return ""
    path = video if isinstance(video, str) else video
    return get_video_info(path)


def toggle_generation():
    return gr.update(visible=False), gr.update(visible=True)


def stop_generation():
    return gr.update(visible=True), gr.update(visible=False)


def create_rtx_vsr_tab():
    with gr.Tab("RTX-Video Super Resolution"):
        with gr.Row():
            generate_btn = gr.Button("Start", variant="primary", scale=0, min_width=80)
            stop_btn = gr.Button(
                "Stop", variant="stop", scale=0, min_width=80, visible=False
            )

        with gr.Row():
            with gr.Column(scale=1):
                input_info = gr.Textbox(
                    label="Video Info", interactive=False, max_lines=1
                )
                input_video = gr.Video(label="Input Video", height=444)

            with gr.Column(scale=1):
                output_info = gr.Textbox(
                    label="Video Info", interactive=False, max_lines=1
                )
                output_video = gr.Video(label="Output Video", height=444)

        input_video.change(
            fn=on_input_video,
            inputs=[input_video],
            outputs=[input_info],
        )

        generate_btn.click(
            fn=toggle_generation,
            outputs=[generate_btn, stop_btn],
        ).then(
            fn=generate_video,
            inputs=[input_video],
            outputs=[output_video, output_info],
        ).then(
            fn=stop_generation,
            outputs=[generate_btn, stop_btn],
        )

        stop_btn.click(
            fn=stop_generation,
            outputs=[generate_btn, stop_btn],
        )
