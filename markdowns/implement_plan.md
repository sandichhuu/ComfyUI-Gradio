Implementation Plan: ComfyUI Gradio Interface Integration
We will build the Gradio interface with two tabs (ZIT and Flux2-Klein-9B) to control ComfyUI-driven image generation as described in the requirements.

Proposed Changes
Component 1: View Layer (Gradio Interface)
[NEW] 
zit_view.py
Create the view module for the ZIT tab:

Implement create_zit_tab() returning a gr.Tab component.
Layout:
Row 1: Prompt Textbox (default: "helloworld", scale=4) + Generate Button / Stop Button (scale=1).
Row 2: Config fields: Seed (number input), Randomize Seed (checkbox), Width (slider, 256-2048, default 1024), Height (slider, 256-2048, default 1024).
Row 3: Two columns:
Column 1: Accordion "LoRAs" (collapsed by default). Inside: dynamic list of LoRAs using @gr.render and gr.State. Each LoRA has an enable checkbox, name textbox, weight number, and remove button. Plus an "Add LoRA" button.
Column 2: Output Image component (fixed height = 444px).
Event handling:
Generate button click hides Generate button, shows Stop button, triggers the generation function, and then restores buttons.
Stop button cancels the generation event and restores buttons.
Prompt text and active LoRA tags (e.g. <name:weight>) are merged before sending to the worker.
[NEW] 
flux2_klein_9b_view.py
Create the view module for the Flux2-Klein-9B tab:

Implement create_flux2_klein_9b_tab() returning a gr.Tab component.
Layout:
Row 1: Prompt Textbox (default: "helloworld", scale=4) + Generate Button / Stop Button (scale=1).
Row 2: Config fields: Seed (number input), Randomize Seed (checkbox), Width (slider, 256-2048, default 1024), Height (slider, 256-2048, default 1024), Toggle Ref (checkbox).
Row 3: Accordion "LoRAs" (collapsed by default) containing dynamic list of LoRAs.
Row 4: Two columns:
Column 1: Input Images. Image 1 is always visible. Image 2 is visible only if "Toggle Ref" is checked. Both have a fixed height of 444px.
Column 2: Output Image (fixed height = 444px).
Event handling:
Similar Generate/Stop toggling and cancellation.
Validates that input images are provided. If Toggle Ref is enabled, both input images are required; otherwise, only the first is required.
Prompt text and active LoRA tags are merged.
[MODIFY] 
main.py
Add premium design styling using custom CSS passed to gr.Blocks(css=css):

Dark glassmorphic background styling, neon glowing button hover states, rounded corners, custom padding.
Ensure the views are imported and mounted correctly.
Component 2: Worker Layer (ComfyUI Integration)
[MODIFY] 
zit_worker.py
Rename main(unload_models: bool | None = None) to generate(prompt_text: str, seed: int, width: int, height: int, unload_models: bool | None = None).
Replace the hardcoded prompt, seed, width, and height with parameters:
Set prompt text in loratagloader.load_lora to prompt_text.
Set width and height in emptysd3latentimage.EXECUTE_NORMALIZED to width and height.
Set seed in KSampler to seed.
Return the generated output image filepath. Use folder_paths.get_output_directory() to find the absolute path of the generated image.
Remove the if __name__ == "__main__": main() block.
[MODIFY] 
flux2_klein_9b_worker.py
Rename main(unload_models: bool | None = None) to generate(prompt_text: str, seed: int, width: int, height: int, toggle_ref: bool, input_img1_path: str, input_img2_path: str | None = None, unload_models: bool | None = None).
Handle copying input images:
Import PIL to resize the input image(s) to (width, height) and save to ComfyUI input directory.
Pass the copied basenames to loadimage.load_image.
Update nodes:
Dynamically set the megapixels in ImageScaleToTotalPixels nodes to (width * height) / 1000000.0.
Update primitiveboolean_220 with toggle_ref.
Set prompt text in loratagloader.load_lora to prompt_text.
Set seed in RandomNoise to seed.
Return the generated output image filepath.
Remove the if __name__ == "__main__": main() block.
Verification Plan
Automated/Manual Tests
Launch the application: uv run main.py.
Test Tab 1 (ZIT):
Add/remove LoRAs, change weights, collapse/expand the accordion.
Verify "Generate" changes to "Stop" during run.
Verify clicking "Stop" cancels generation.
Verify refreshing the browser resets state correctly.
Test Tab 2 (Flux2-Klein-9B):
Toggle "Toggle Ref" and verify Image 2 visibility toggles.
Upload image(s), input prompt, generate, verify result.
Verify image components keep height at 444px.