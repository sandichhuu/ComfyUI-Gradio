"""Main entry point for the ComfyUI Gradio application."""

import gradio as gr
from views.zit_view import create_zit_tab
from views.flux2_klein_9b_view import create_flux2_klein_9b_tab


def main():
    """Launch the Gradio application with ZIT and Flux2-Klein-9B tabs."""
    # Create the Gradio interface with tabs
    with gr.Blocks(title="ComfyUI Gradio") as demo:
        gr.Markdown("# ComfyUI Image Generation")
        
        # Create tabs
        create_zit_tab()
        create_flux2_klein_9b_tab()
    
    # Launch the application
    demo.launch()


if __name__ == "__main__":
    main()
