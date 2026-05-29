# LoRA System Implementation

## Overview
LoRAs (Low-Rank Adaptations) are managed dynamically in the UI and merged into prompts before sending to ComfyUI.

## UI Representation
Each LoRA is a dict in the state list:
```python
{"enabled": True, "tag": "<lora_name:weight>"}
```

## LoRA Format
- Pattern: `^<[^:]+:[\d.]+>$`
- Example: `<lora:76N0PGDVMCA64NA75C2NW7V600:1>`
- Components: `<lora:` + name + `:` + weight + `>`

## Validation
```python
LORA_PATTERN = re.compile(r"^<[^:]+:[\d.]+>$")

for item in current_loras:
    enabled = item.get("enabled", False)
    tag = item.get("tag", "").strip()
    
    if enabled and tag:
        if not LORA_PATTERN.match(tag):
            gr.Warning(f"Invalid LoRA format: '{tag}'. Expected format: <name:weight>")
            return gr.update(), gr.update(label="Seed")
        lora_tags.append(tag)
```

## Merging into Prompt
```python
def merge_loras_into_prompt(prompt_text, lora_tags):
    if lora_tags:
        return prompt_text + "\n" + "\n".join(lora_tags)
    return prompt_text
```

LoRA tags are appended on new lines after the prompt text. The LoRA tags are NOT visible in the prompt textbox - they're merged internally.

## ComfyUI Processing
In the worker, the merged prompt (with LoRA tags) is passed to `LoraTagLoader`:
```python
loratagloader = NODE_CLASS_MAPPINGS["LoraTagLoader"]()
result = loratagloader.load_lora(
    text=prompt_text,  # Contains prompt + LoRA tags
    model=model,
    clip=clip,
)
```

`LoraTagLoader` parses the `<name:weight>` tags, loads the LoRA files, and modifies the model and CLIP accordingly. It outputs:
- Modified MODEL
- Modified CLIP
- Cleaned text (tags removed)

## Dynamic UI with @gr.render
```python
@gr.render(inputs=lora_state)
def render_loras(state_list):
    if not state_list:
        gr.Markdown("*No LoRA loaded*")
        return
    
    for idx, item in enumerate(state_list):
        with gr.Row(elem_id=f"lora-row-{idx}"):
            chk = gr.Checkbox(value=item["enabled"], show_label=False, scale=0, min_width=40)
            txt = gr.Textbox(value=item["tag"], placeholder="<lora_name:1.0>", show_label=False, scale=6, container=False)
            idx_holder = gr.Number(value=idx, visible=False)
            del_btn = gr.Button("🗑️", variant="stop", scale=0, min_width=80)
```

## State Updates
Each UI interaction updates the state list:
- Checkbox change → `update_chk(state_list, idx, val)` → updates `enabled`
- Text change → `update_txt(state_list, idx, val)` → updates `tag`
- Delete click → `remove_lora_row(state_list, idx)` → removes item
- Add click → `add_lora_row(state_list)` → appends new item

## Maximum LoRAs
Both tabs enforce `MAX_LORAS = 6`.
