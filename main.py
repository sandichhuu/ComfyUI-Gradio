import gradio as gr
from views.zit_view import create_zit_tab
from views.flux2_klein_9b_view import create_flux2_klein_9b_tab


def main():
    # Create Gradio interface with tabs
    with gr.Blocks(title="ComfyUI Image Generator") as demo:
        gr.Markdown("# ComfyUI Image Generator")
        
        with gr.Tabs() as tabs:
            zit_tab = create_zit_tab()
            flux2_tab = create_flux2_klein_9b_tab()
    
    demo.launch()


if __name__ == "__main__":
    main()
