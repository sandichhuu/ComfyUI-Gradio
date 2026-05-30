import gradio as gr
from views.zit_view import create_zit_tab
from views.flux2_klein_9b_view import create_flux2_klein_9b_tab
from views.ltx23_view import create_ltx23_tab

css = """
.gradio-container {
    min-height: 100vh;
}
.dark .gradio-container {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
}
.light .gradio-container {
    background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 50%, #bbdefb 100%);
}
.gr-block {
    background: transparent !important;
}
.gr-button {
    border: none;
    color: white;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: bold;
    transition: all 0.3s ease;
}
button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}
button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
}
button.stop {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
}
button.stop:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(245, 87, 108, 0.6);
}
.dark .gr-textbox, .dark .gr-number, .dark .gr-checkbox {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
}
.light .gr-textbox, .light .gr-number, .light .gr-checkbox {
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 8px;
}
.dark .gr-accordion {
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.03);
}
.light .gr-accordion {
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.5);
}
.dark .gr-image {
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
}
.light .gr-image {
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 8px;
}
.image-flex-row {
    display: flex;
    gap: 10px;
}
.image-flex-row > div {
    flex: 1;
    min-width: 0;
}
"""

if __name__ == "__main__":
    with gr.Blocks(title="🍥 Ramen", analytics_enabled=False) as demo:
        gr.Markdown("# 🍥 Ramen")

        with gr.Tabs() as tabs:
            create_zit_tab()
            create_flux2_klein_9b_tab()
            create_ltx23_tab()

    demo.launch(css=css, pwa=True)
