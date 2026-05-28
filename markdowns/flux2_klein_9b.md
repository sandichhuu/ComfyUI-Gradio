This document describe the Flux module (Flux2-Klein-9B) working.

File structure:
1. views/flux2_klein_9b_view.py
2. workers/flux2_klein_9b_worker.py

flux2_klein_9b_view: The gradio package support UI renderer.
flux2_klein_9b_worker: The code extract from comfyui.

# flux2_klein_9b_view layout:

| prompt: helloworld | Generate / Stop button |
| config: seed (with toggle random value), width, height, toggle_ref |
| loRA list (each with enable toggle, name, strength) |
| input_img | output image |

if toggle_ref enabled -> we have 2 input_img.

# flux2_klein_9b_worker:

need modify, remove main block.
rename main function to generate and add config parameters into.