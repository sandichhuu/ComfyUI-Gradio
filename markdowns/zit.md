This document describe the ZIT module (Z-Image-Turbo) working.

File structure:
1. views/zit_view.py
2. workers/zit_worker.py

zit_view: The gradio package support UI renderer.
zit_worker: The code extract from comfyui.

# zit_view layout:

| prompt: helloworld | Generate / Stop button |
| config: seed (with toggle random value), width, height |
| loRA list (each with enable toggle, name, strength) | output image |

# zit_worker:

need modify, remove main block.
rename main function to generate and add config parameters into.