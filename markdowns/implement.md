We will use gradio to draw 2 tab on main.py
tab1: ZIT. tab2: Flux2-Klein-9B

to implement:
tab1: zit.md.
tab2: flux2_klein_9b.md

This project is using Python (uv), using gradio with comfyui server to generate images.
If the input / output image too big, set fix-height = 444px

Make sure the web still working if refresh browser.
When running, the Generate button have to changed to Stop button.

loRA list have collapse function.
this one can add more loRA item.
each loRA has field name, weight and remove button.
if press genegrate button, loRA tag like <name:1.0> will be merge into prompt then send. (but without see on prompt field).

see implement_plan.md to more detail.