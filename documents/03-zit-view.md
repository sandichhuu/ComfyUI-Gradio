# views/zit_view.py - ZIT Tab Implementation

## Purpose
Renders the ZIT (Z-Image-Turbo) tab UI and handles all user interactions for text-to-image generation.

## Constants
```python
MAX_LORAS = 6
LORA_PATTERN = re.compile(r"^<[^:]+:[\d.]+>$")
```

## LoRA Helper Functions

### update_chk(state_list, idx, val)
Updates the `enabled` field of a LoRA item at index `idx` in the state list.

### update_txt(state_list, idx, val)
Updates the `tag` field (stripped) of a LoRA item at index `idx`.

### remove_lora_row(state_list, idx_to_remove)
Returns a new list with the item at `idx_to_remove` excluded.

### add_lora_row(current_state)
Appends `{"enabled": True, "tag": ""}` if under `MAX_LORAS`, otherwise shows warning.

### merge_loras_into_prompt(prompt_text, lora_tags)
Joins prompt text and LoRA tags with newlines: `prompt_text + "\n" + "\n".join(lora_tags)`

## generate_image(prompt_text, seed_val, w, h, current_loras)

1. If seed is 0, generate random seed: `random.randint(1, 2**64)`
2. Collect enabled LoRA tags, validate each against `LORA_PATTERN`
3. Invalid format shows warning and returns early
4. Merge LoRA tags into prompt text
5. Call `zit_generate()` from worker
6. Return image result + updated seed label

## toggle_generation() / stop_generation()
- `toggle_generation()`: hides generate button, shows stop button
- `stop_generation()`: shows generate button, hides stop button

## UI Layout: create_zit_tab()

```
Row 1: [Prompt (scale=4)] [Generate (scale=1)] [Stop (scale=1, hidden)]
Row 2: [Seed (Number)] [Width (Number)] [Height (Number)]
Row 3: [LoRA Column (scale=1)] [Output Image Column (scale=1)]
```

### Row 1
- `gr.Textbox`: label="Prompt", value="helloworld", scale=4, lines=1
- `gr.Button`: "Generate", variant="primary", scale=1
- `gr.Button`: "Stop", variant="stop", scale=1, visible=False

### Row 2
- `gr.Number`: label="Seed", value=0, precision=0
- `gr.Number`: label="Width", value=1024, precision=0
- `gr.Number`: label="Height", value=1024, precision=0

### Row 3 - Left Column (LoRAs)
- `gr.State([])` for LoRA state
- `gr.Accordion("LoRAs", open=True)`
  - `gr.Button("➕ Add LoRA", variant="secondary", size="sm")`
  - `@gr.render(inputs=lora_state)` renders dynamic LoRA rows:
    - If empty: shows "*No LoRA loaded*"
    - Each row: Checkbox (scale=0, min_width=40) + Textbox (scale=6, container=False) + hidden Number + Delete Button (scale=0, min_width=80)
    - Event handlers on each: `chk.change`, `txt.change`, `del_btn.click`

### Row 3 - Right Column (Output)
- `gr.Image(label="Output", height=444)`

## Event Wiring
```python
add_lora_btn.click(fn=add_lora_row, inputs=[lora_state], outputs=[lora_state])

generate_btn.click(
    fn=toggle_generation,
    outputs=[generate_btn, stop_btn],
).then(
    fn=generate_image,
    inputs=[prompt, seed, width, height, lora_state],
    outputs=[output_image, seed],
).then(
    fn=stop_generation,
    outputs=[generate_btn, stop_btn],
)

stop_btn.click(
    fn=stop_generation,
    outputs=[generate_btn, stop_btn],
)
```
