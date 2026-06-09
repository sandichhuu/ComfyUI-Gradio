import json
import os
import random
from typing import Any

from workers.comfy_worker import (
    get_value_at_index,
    bootstrap_comfyui_runtime,
    add_extra_model_paths,
    import_custom_nodes,
    cleanup_comfyui_runtime,
)


# Workflow data
def build_workflow() -> dict[str, Any]:
    return {
        "9": {
            "inputs": {"filename_prefix": "Flux2-Klein", "images": ["180", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
        },
        "159": {
            "inputs": {"image": "example.png"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"},
        },
        "163": {
            "inputs": {"image": "example.png"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"},
        },
        "179": {
            "inputs": {"sampler_name": "euler"},
            "class_type": "KSamplerSelect",
            "_meta": {"title": "KSamplerSelect"},
        },
        "180": {
            "inputs": {"samples": ["192", 0], "vae": ["186", 0]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
        },
        "181": {
            "inputs": {"noise_seed": 153320490984662},
            "class_type": "RandomNoise",
            "_meta": {"title": "RandomNoise"},
        },
        "182": {
            "inputs": {"width": ["223", 1], "height": ["223", 2], "batch_size": 1},
            "class_type": "EmptyFlux2LatentImage",
            "_meta": {"title": "Empty Flux 2 Latent"},
        },
        "183": {
            "inputs": {
                "cfg": 1,
                "model": ["189", 0],
                "positive": ["221", 0],
                "negative": ["193", 0],
            },
            "class_type": "CFGGuider",
            "_meta": {"title": "CFG Guider"},
        },
        "184": {
            "inputs": {"steps": 4, "width": ["223", 1], "height": ["223", 2]},
            "class_type": "Flux2Scheduler",
            "_meta": {"title": "Flux2Scheduler"},
        },
        "186": {
            "inputs": {"vae_name": "full_encoder_small_decoder.safetensors"},
            "class_type": "VAELoader",
            "_meta": {"title": "Load VAE"},
        },
        "187": {
            "inputs": {
                "clip_name": "qwen_3_8b_fp8mixed.safetensors",
                "type": "flux2",
                "device": "default",
            },
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load CLIP"},
        },
        "189": {
            "inputs": {
                "sage_attention": "sageattn_qk_int8_pv_fp16_cuda",
                "allow_compile": True,
                "model": ["219", 0],
            },
            "class_type": "PathchSageAttentionKJ",
            "_meta": {"title": "Patch Sage Attention KJ"},
        },
        "190": {
            "inputs": {
                "lora_name": "Flux.2 Klein 9B-base/KLEIN-Unchained-V2.safetensors",
                "strength_model": 0.6,
                "strength_clip": 1,
                "model": ["191", 0],
                "clip": ["187", 0],
            },
            "class_type": "LoraLoader",
            "_meta": {"title": "Load LoRA (Model and CLIP)"},
        },
        "191": {
            "inputs": {
                "unet_name": "Flux.2 Klein 9B/flux-2-klein-9b-fp8.safetensors",
                "weight_dtype": "default",
            },
            "class_type": "UNETLoader",
            "_meta": {"title": "Load Diffusion Model"},
        },
        "192": {
            "inputs": {
                "noise": ["181", 0],
                "guider": ["183", 0],
                "sampler": ["179", 0],
                "sigmas": ["184", 0],
                "latent_image": ["182", 0],
            },
            "class_type": "SamplerCustomAdvanced",
            "_meta": {"title": "SamplerCustomAdvanced"},
        },
        "193": {
            "inputs": {"conditioning": ["212", 0], "latent": ["195", 0]},
            "class_type": "ReferenceLatent",
            "_meta": {"title": "ReferenceLatent"},
        },
        "194": {
            "inputs": {"conditioning": ["201", 0], "latent": ["195", 0]},
            "class_type": "ReferenceLatent",
            "_meta": {"title": "ReferenceLatent"},
        },
        "195": {
            "inputs": {"pixels": ["196", 0], "vae": ["186", 0]},
            "class_type": "VAEEncode",
            "_meta": {"title": "VAE Encode"},
        },
        "196": {
            "inputs": {
                "upscale_method": "lanczos",
                "megapixels": 1,
                "resolution_steps": 1,
                "image": ["159", 0],
            },
            "class_type": "ImageScaleToTotalPixels",
            "_meta": {"title": "Scale Image to Total Pixels"},
        },
        "198": {
            "inputs": {
                "upscale_method": "lanczos",
                "megapixels": 1,
                "resolution_steps": 1,
                "image": ["163", 0],
            },
            "class_type": "ImageScaleToTotalPixels",
            "_meta": {"title": "Scale Image to Total Pixels"},
        },
        "199": {
            "inputs": {"pixels": ["198", 0], "vae": ["186", 0]},
            "class_type": "VAEEncode",
            "_meta": {"title": "VAE Encode"},
        },
        "200": {
            "inputs": {"conditioning": ["194", 0], "latent": ["199", 0]},
            "class_type": "ReferenceLatent",
            "_meta": {"title": "ReferenceLatent"},
        },
        "201": {
            "inputs": {"text": ["219", 2], "clip": ["219", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
        },
        "212": {
            "inputs": {"conditioning": ["194", 0]},
            "class_type": "ConditioningZeroOut",
            "_meta": {"title": "ConditioningZeroOut"},
        },
        "219": {
            "inputs": {
                "text": "remove bicycle.\n<lora:76N0PGDVMCA64NA75C2NW7V600:1>",
                "model": ["190", 0],
                "clip": ["190", 1],
            },
            "class_type": "LoraTagLoader",
            "_meta": {"title": "Load LoRA Tag"},
        },
        "220": {
            "inputs": {"value": False},
            "class_type": "PrimitiveBoolean",
            "_meta": {"title": "Use Ref"},
        },
        "221": {
            "inputs": {
                "switch": ["220", 0],
                "on_false": ["194", 0],
                "on_true": ["200", 0],
            },
            "class_type": "ComfySwitchNode",
            "_meta": {"title": "Switch"},
        },
        "223": {
            "inputs": {"image": ["196", 0]},
            "class_type": "GetImageSizeAndCount",
            "_meta": {"title": "Get Image Size & Count"},
        },
    }


def build_extra_pnginfo() -> dict[str, Any] | None:
    return {
        "workflow": {
            "id": "92112d97-bb64-4b44-86f2-ea5691ef8f6e",
            "revision": 0,
            "last_node_id": 223,
            "last_link_id": 376,
            "nodes": [
                {
                    "id": 179,
                    "type": "KSamplerSelect",
                    "pos": [338.10647798790785, 913.388807325185],
                    "size": [270, 110],
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "SAMPLER", "type": "SAMPLER", "links": [298]}],
                    "properties": {
                        "Node name for S&R": "KSamplerSelect",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.7",
                            "input_ue_unconnectable": {},
                        },
                    },
                    "widgets_values": ["euler"],
                },
                {
                    "id": 180,
                    "type": "VAEDecode",
                    "pos": [1018.1064779879082, 563.3888073251848],
                    "size": [230, 100],
                    "flags": {},
                    "order": 26,
                    "mode": 0,
                    "inputs": [
                        {"name": "samples", "type": "LATENT", "link": 301},
                        {"name": "vae", "type": "VAE", "link": 302},
                    ],
                    "outputs": [
                        {
                            "name": "IMAGE",
                            "type": "IMAGE",
                            "slot_index": 0,
                            "links": [303],
                        }
                    ],
                    "properties": {
                        "Node name for S&R": "VAEDecode",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.7",
                            "input_ue_unconnectable": {},
                        },
                    },
                    "widgets_values": [],
                },
                {
                    "id": 181,
                    "type": "RandomNoise",
                    "pos": [338.10647798790785, 553.3888073251848],
                    "size": [270, 110],
                    "flags": {},
                    "order": 1,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "NOISE", "type": "NOISE", "links": [296]}],
                    "properties": {
                        "Node name for S&R": "RandomNoise",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.7",
                            "input_ue_unconnectable": {},
                        },
                    },
                    "widgets_values": [153320490984662, "randomize"],
                },
                {
                    "id": 182,
                    "type": "EmptyFlux2LatentImage",
                    "pos": [980.1608924591122, 1333.3888073251846],
                    "size": [270, 170],
                    "flags": {},
                    "order": 18,
                    "mode": 0,
                    "inputs": [
                        {
                            "name": "width",
                            "type": "INT",
                            "widget": {"name": "width"},
                            "link": 375,
                        },
                        {
                            "name": "height",
                            "type": "INT",
                            "widget": {"name": "height"},
                            "link": 376,
                        },
                    ],
                    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [300]}],
                    "properties": {
                        "Node name for S&R": "EmptyFlux2LatentImage",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.7",
                            "input_ue_unconnectable": {},
                        },
                    },
                    "widgets_values": [1024, 1024, 1],
                },
                {
                    "id": 184,
                    "type": "Flux2Scheduler",
                    "pos": [338.10647798790785, 1083.3888073251844],
                    "size": [270, 170],
                    "flags": {},
                    "order": 17,
                    "mode": 0,
                    "inputs": [
                        {
                            "name": "width",
                            "type": "INT",
                            "widget": {"name": "width"},
                            "link": 373,
                        },
                        {
                            "name": "height",
                            "type": "INT",
                            "widget": {"name": "height"},
                            "link": 374,
                        },
                    ],
                    "outputs": [{"name": "SIGMAS", "type": "SIGMAS", "links": [299]}],
                    "properties": {
                        "Node name for S&R": "Flux2Scheduler",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.7",
                            "input_ue_unconnectable": {},
                        },
                    },
                    "widgets_values": [4, 1024, 1024],
                },
                {
                    "id": 187,
                    "type": "CLIPLoader",
                    "pos": [-863.1744346951235, 788.4861402285036],
                    "size": [370, 150],
                    "flags": {},
                    "order": 2,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "CLIP", "type": "CLIP", "links": [308]}],
                    "properties": {
                        "Node name for S&R": "CLIPLoader",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "models": [
                            {
                                "name": "qwen_3_8b_fp8mixed.safetensors",
                                "url": "https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
                                "directory": "text_encoders",
                            }
                        ],
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.7",
                            "input_ue_unconnectable": {},
                        },
                    },
                    "widgets_values": [
                        "qwen_3_8b_fp8mixed.safetensors",
                        "flux2",
                        "default",
                    ],
                },
                {
                    "id": 191,
                    "type": "UNETLoader",
                    "pos": [-863.1744346951235, 548.4861402285032],
                    "size": [370, 110],
                    "flags": {},
                    "order": 3,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "MODEL", "type": "MODEL", "links": [311]}],
                    "properties": {
                        "Node name for S&R": "UNETLoader",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "models": [
                            {
                                "name": "flux-2-klein-9b-fp8.safetensors",
                                "url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/resolve/main/flux-2-klein-9b-fp8.safetensors",
                                "directory": "diffusion_models",
                            }
                        ],
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.7",
                            "input_ue_unconnectable": {},
                        },
                    },
                    "widgets_values": [
                        "Flux.2 Klein 9B/flux-2-klein-9b-fp8.safetensors",
                        "default",
                    ],
                },
                {
                    "id": 192,
                    "type": "SamplerCustomAdvanced",
                    "pos": [678.1064779879074, 553.3888073251848],
                    "size": [230, 172],
                    "flags": {},
                    "order": 25,
                    "mode": 0,
                    "inputs": [
                        {"name": "noise", "type": "NOISE", "link": 296},
                        {"name": "guider", "type": "GUIDER", "link": 297},
                        {"name": "sampler", "type": "SAMPLER", "link": 298},
                        {"name": "sigmas", "type": "SIGMAS", "link": 299},
                        {"name": "latent_image", "type": "LATENT", "link": 300},
                    ],
                    "outputs": [
                        {"name": "output", "type": "LATENT", "links": [301]},
                        {"name": "denoised_output", "type": "LATENT", "links": []},
                    ],
                    "properties": {
                        "Node name for S&R": "SamplerCustomAdvanced",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.7",
                            "input_ue_unconnectable": {},
                        },
                    },
                    "widgets_values": [],
                },
                {
                    "id": 199,
                    "type": "VAEEncode",
                    "pos": [-262.42519181120747, 1759.1256407867652],
                    "size": [230, 100],
                    "flags": {"collapsed": False},
                    "order": 14,
                    "mode": 0,
                    "inputs": [
                        {"name": "pixels", "type": "IMAGE", "link": 331},
                        {"name": "vae", "type": "VAE", "link": 332},
                    ],
                    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [327]}],
                    "properties": {
                        "Node name for S&R": "VAEEncode",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.7",
                            "input_ue_unconnectable": {},
                        },
                    },
                    "widgets_values": [],
                },
                {
                    "id": 9,
                    "type": "SaveImage",
                    "pos": [1462.2500310255027, 723.6631559788559],
                    "size": [380, 410],
                    "flags": {},
                    "order": 27,
                    "mode": 0,
                    "inputs": [{"name": "images", "type": "IMAGE", "link": 303}],
                    "outputs": [],
                    "properties": {
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "Node name for S&R": "SaveImage",
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.7",
                            "input_ue_unconnectable": {},
                        },
                    },
                    "widgets_values": ["Flux2-Klein"],
                },
                {
                    "id": 195,
                    "type": "VAEEncode",
                    "pos": [-237.5954803629689, 1375.9874282048181],
                    "size": [230, 100],
                    "flags": {"collapsed": False},
                    "order": 12,
                    "mode": 0,
                    "inputs": [
                        {"name": "pixels", "type": "IMAGE", "link": 319},
                        {"name": "vae", "type": "VAE", "link": 321},
                    ],
                    "outputs": [
                        {"name": "LATENT", "type": "LATENT", "links": [317, 318]}
                    ],
                    "properties": {
                        "Node name for S&R": "VAEEncode",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.7",
                            "input_ue_unconnectable": {},
                        },
                    },
                    "widgets_values": [],
                },
                {
                    "id": 183,
                    "type": "CFGGuider",
                    "pos": [338.10647798790785, 703.3888073251844],
                    "size": [270, 160],
                    "flags": {},
                    "order": 24,
                    "mode": 0,
                    "inputs": [
                        {"name": "model", "type": "MODEL", "link": 316},
                        {"name": "positive", "type": "CONDITIONING", "link": 370},
                        {"name": "negative", "type": "CONDITIONING", "link": 324},
                    ],
                    "outputs": [{"name": "GUIDER", "type": "GUIDER", "links": [297]}],
                    "properties": {
                        "Node name for S&R": "CFGGuider",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.7",
                            "input_ue_unconnectable": {},
                        },
                    },
                    "widgets_values": [1],
                },
                {
                    "id": 212,
                    "type": "ConditioningZeroOut",
                    "pos": [72.07201978401042, 1482.405335123961],
                    "size": [204.134765625, 26],
                    "flags": {},
                    "order": 21,
                    "mode": 0,
                    "inputs": [
                        {"name": "conditioning", "type": "CONDITIONING", "link": 351}
                    ],
                    "outputs": [
                        {"name": "CONDITIONING", "type": "CONDITIONING", "links": [350]}
                    ],
                    "properties": {"Node name for S&R": "ConditioningZeroOut"},
                    "widgets_values": [],
                },
                {
                    "id": 193,
                    "type": "ReferenceLatent",
                    "pos": [64.94028635437203, 1562.713571586732],
                    "size": [230, 100],
                    "flags": {"collapsed": False},
                    "order": 23,
                    "mode": 0,
                    "inputs": [
                        {"name": "conditioning", "type": "CONDITIONING", "link": 350},
                        {"name": "latent", "shape": 7, "type": "LATENT", "link": 317},
                    ],
                    "outputs": [
                        {"name": "CONDITIONING", "type": "CONDITIONING", "links": [324]}
                    ],
                    "properties": {
                        "Node name for S&R": "ReferenceLatent",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.7",
                            "input_ue_unconnectable": {},
                        },
                    },
                    "widgets_values": [],
                },
                {
                    "id": 186,
                    "type": "VAELoader",
                    "pos": [-863.1744346951235, 1058.4861402285028],
                    "size": [370, 110],
                    "flags": {},
                    "order": 4,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [
                        {"name": "VAE", "type": "VAE", "links": [302, 321, 332]}
                    ],
                    "properties": {
                        "Node name for S&R": "VAELoader",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "models": [
                            {
                                "name": "full_encoder_small_decoder.safetensors",
                                "url": "https://huggingface.co/black-forest-labs/FLUX.2-small-decoder/resolve/main/full_encoder_small_decoder.safetensors",
                                "directory": "vae",
                            }
                        ],
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.7",
                            "input_ue_unconnectable": {},
                        },
                    },
                    "widgets_values": ["full_encoder_small_decoder.safetensors"],
                },
                {
                    "id": 189,
                    "type": "PathchSageAttentionKJ",
                    "pos": [326.4006995823687, 280.61277213638186],
                    "size": [270, 132],
                    "flags": {},
                    "order": 15,
                    "mode": 0,
                    "inputs": [{"name": "model", "type": "MODEL", "link": 363}],
                    "outputs": [{"name": "MODEL", "type": "MODEL", "links": [316]}],
                    "properties": {"Node name for S&R": "PathchSageAttentionKJ"},
                    "widgets_values": ["sageattn_qk_int8_pv_fp16_cuda", True],
                },
                {
                    "id": 190,
                    "type": "LoraLoader",
                    "pos": [-861.3194618932883, 286.4410898764934],
                    "size": [290.4333190917969, 168],
                    "flags": {},
                    "order": 8,
                    "mode": 0,
                    "inputs": [
                        {"name": "model", "type": "MODEL", "link": 311},
                        {"name": "clip", "type": "CLIP", "link": 308},
                    ],
                    "outputs": [
                        {"name": "MODEL", "type": "MODEL", "links": [361]},
                        {"name": "CLIP", "type": "CLIP", "links": [362]},
                    ],
                    "properties": {"Node name for S&R": "LoraLoader"},
                    "widgets_values": [
                        "Flux.2 Klein 9B-base/KLEIN-Unchained-V2.safetensors",
                        0.6,
                        1,
                    ],
                },
                {
                    "id": 201,
                    "type": "CLIPTextEncode",
                    "pos": [-171.89352201209232, 553.3888073251848],
                    "size": [368.63135836515335, 88],
                    "flags": {},
                    "order": 16,
                    "mode": 0,
                    "inputs": [
                        {"name": "clip", "type": "CLIP", "link": 364},
                        {
                            "name": "text",
                            "type": "STRING",
                            "widget": {"name": "text"},
                            "link": 365,
                        },
                    ],
                    "outputs": [
                        {
                            "name": "CONDITIONING",
                            "type": "CONDITIONING",
                            "slot_index": 0,
                            "links": [322],
                        }
                    ],
                    "title": "CLIP Text Encode (Positive Prompt)",
                    "properties": {
                        "Node name for S&R": "CLIPTextEncode",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.7",
                            "input_ue_unconnectable": {},
                        },
                    },
                    "widgets_values": [""],
                    "color": "#232",
                    "bgcolor": "#353",
                },
                {
                    "id": 219,
                    "type": "LoraTagLoader",
                    "pos": [-184.58934439892158, 255.80204862283267],
                    "size": [400, 200],
                    "flags": {},
                    "order": 11,
                    "mode": 0,
                    "inputs": [
                        {"name": "model", "type": "MODEL", "link": 361},
                        {"name": "clip", "type": "CLIP", "link": 362},
                    ],
                    "outputs": [
                        {"name": "MODEL", "type": "MODEL", "links": [363]},
                        {"name": "CLIP", "type": "CLIP", "links": [364]},
                        {"name": "STRING", "type": "STRING", "links": [365]},
                    ],
                    "properties": {"Node name for S&R": "LoraTagLoader"},
                    "widgets_values": [
                        "remove bicycle.\n<lora:76N0PGDVMCA64NA75C2NW7V600:1>"
                    ],
                },
                {
                    "id": 220,
                    "type": "PrimitiveBoolean",
                    "pos": [-1652.2780198342068, 1756.7481584975178],
                    "size": [270, 58],
                    "flags": {},
                    "order": 5,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "BOOLEAN", "type": "BOOLEAN", "links": [367]}],
                    "title": "Use Ref",
                    "properties": {"Node name for S&R": "PrimitiveBoolean"},
                    "widgets_values": [False],
                },
                {
                    "id": 200,
                    "type": "ReferenceLatent",
                    "pos": [65.64271755946478, 1754.8244340948752],
                    "size": [230, 100],
                    "flags": {"collapsed": False},
                    "order": 20,
                    "mode": 0,
                    "inputs": [
                        {"name": "conditioning", "type": "CONDITIONING", "link": 328},
                        {"name": "latent", "shape": 7, "type": "LATENT", "link": 327},
                    ],
                    "outputs": [
                        {"name": "CONDITIONING", "type": "CONDITIONING", "links": [368]}
                    ],
                    "properties": {
                        "Node name for S&R": "ReferenceLatent",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.7",
                            "input_ue_unconnectable": {},
                        },
                    },
                    "widgets_values": [],
                },
                {
                    "id": 194,
                    "type": "ReferenceLatent",
                    "pos": [61.667786303697774, 1341.829778204818],
                    "size": [230, 100],
                    "flags": {"collapsed": False},
                    "order": 19,
                    "mode": 0,
                    "inputs": [
                        {"name": "conditioning", "type": "CONDITIONING", "link": 322},
                        {"name": "latent", "shape": 7, "type": "LATENT", "link": 318},
                    ],
                    "outputs": [
                        {
                            "name": "CONDITIONING",
                            "type": "CONDITIONING",
                            "links": [328, 351, 369],
                        }
                    ],
                    "properties": {
                        "Node name for S&R": "ReferenceLatent",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.7",
                            "input_ue_unconnectable": {},
                        },
                    },
                    "widgets_values": [],
                },
                {
                    "id": 221,
                    "type": "ComfySwitchNode",
                    "pos": [420.8501008348952, 1632.6184969248613],
                    "size": [270, 78],
                    "flags": {},
                    "order": 22,
                    "mode": 0,
                    "inputs": [
                        {"name": "on_false", "type": "CONDITIONING", "link": 369},
                        {"name": "on_true", "type": "CONDITIONING", "link": 368},
                        {
                            "name": "switch",
                            "type": "BOOLEAN",
                            "widget": {"name": "switch"},
                            "link": 367,
                        },
                    ],
                    "outputs": [
                        {"name": "output", "type": "CONDITIONING", "links": [370]}
                    ],
                    "properties": {"Node name for S&R": "ComfySwitchNode"},
                    "widgets_values": [False],
                },
                {
                    "id": 159,
                    "type": "LoadImage",
                    "pos": [-1641.4014134062184, 595.9626804093201],
                    "size": [345.5666809082031, 446.1499938964844],
                    "flags": {},
                    "order": 6,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [
                        {"name": "IMAGE", "type": "IMAGE", "links": [320]},
                        {"name": "MASK", "type": "MASK", "links": None},
                    ],
                    "properties": {"Node name for S&R": "LoadImage"},
                    "widgets_values": ["example.png", "image"],
                },
                {
                    "id": 163,
                    "type": "LoadImage",
                    "pos": [-1646.102658744966, 1145.3039933595937],
                    "size": [342.5, 371.5],
                    "flags": {},
                    "order": 7,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [
                        {"name": "IMAGE", "type": "IMAGE", "links": [330]},
                        {"name": "MASK", "type": "MASK", "links": None},
                    ],
                    "properties": {"Node name for S&R": "LoadImage"},
                    "widgets_values": ["example.png", "image"],
                },
                {
                    "id": 198,
                    "type": "ImageScaleToTotalPixels",
                    "pos": [-751.5489666646375, 1519.735574503712],
                    "size": [270, 140],
                    "flags": {},
                    "order": 10,
                    "mode": 0,
                    "inputs": [{"name": "image", "type": "IMAGE", "link": 330}],
                    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [331]}],
                    "properties": {
                        "Node name for S&R": "ImageScaleToTotalPixels",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.7",
                            "input_ue_unconnectable": {},
                        },
                    },
                    "widgets_values": ["lanczos", 1, 1],
                },
                {
                    "id": 196,
                    "type": "ImageScaleToTotalPixels",
                    "pos": [-746.8545028900982, 1314.0629294088326],
                    "size": [270, 140],
                    "flags": {},
                    "order": 9,
                    "mode": 0,
                    "inputs": [{"name": "image", "type": "IMAGE", "link": 320}],
                    "outputs": [
                        {"name": "IMAGE", "type": "IMAGE", "links": [319, 372]}
                    ],
                    "properties": {
                        "Node name for S&R": "ImageScaleToTotalPixels",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.7",
                            "input_ue_unconnectable": {},
                        },
                    },
                    "widgets_values": ["lanczos", 1, 1],
                },
                {
                    "id": 223,
                    "type": "GetImageSizeAndCount",
                    "pos": [-279.6064154727612, 1562.6693783498815],
                    "size": [252.34876407790352, 86],
                    "flags": {},
                    "order": 13,
                    "mode": 0,
                    "inputs": [{"name": "image", "type": "IMAGE", "link": 372}],
                    "outputs": [
                        {"name": "image", "type": "IMAGE", "links": None},
                        {
                            "label": "width",
                            "name": "width",
                            "type": "INT",
                            "links": [373, 375],
                        },
                        {
                            "label": "height",
                            "name": "height",
                            "type": "INT",
                            "links": [374, 376],
                        },
                        {
                            "label": "count",
                            "name": "count",
                            "type": "INT",
                            "links": None,
                        },
                    ],
                    "properties": {"Node name for S&R": "GetImageSizeAndCount"},
                    "widgets_values": [],
                },
            ],
            "links": [
                [296, 181, 0, 192, 0, "NOISE"],
                [297, 183, 0, 192, 1, "GUIDER"],
                [298, 179, 0, 192, 2, "SAMPLER"],
                [299, 184, 0, 192, 3, "SIGMAS"],
                [300, 182, 0, 192, 4, "LATENT"],
                [301, 192, 0, 180, 0, "LATENT"],
                [302, 186, 0, 180, 1, "VAE"],
                [303, 180, 0, 9, 0, "IMAGE"],
                [308, 187, 0, 190, 1, "CLIP"],
                [311, 191, 0, 190, 0, "MODEL"],
                [316, 189, 0, 183, 0, "MODEL"],
                [317, 195, 0, 193, 1, "LATENT"],
                [318, 195, 0, 194, 1, "LATENT"],
                [319, 196, 0, 195, 0, "IMAGE"],
                [320, 159, 0, 196, 0, "IMAGE"],
                [321, 186, 0, 195, 1, "VAE"],
                [322, 201, 0, 194, 0, "CONDITIONING"],
                [324, 193, 0, 183, 2, "CONDITIONING"],
                [327, 199, 0, 200, 1, "LATENT"],
                [328, 194, 0, 200, 0, "CONDITIONING"],
                [330, 163, 0, 198, 0, "IMAGE"],
                [331, 198, 0, 199, 0, "IMAGE"],
                [332, 186, 0, 199, 1, "VAE"],
                [350, 212, 0, 193, 0, "CONDITIONING"],
                [351, 194, 0, 212, 0, "CONDITIONING"],
                [361, 190, 0, 219, 0, "MODEL"],
                [362, 190, 1, 219, 1, "CLIP"],
                [363, 219, 0, 189, 0, "MODEL"],
                [364, 219, 1, 201, 0, "CLIP"],
                [365, 219, 2, 201, 1, "STRING"],
                [367, 220, 0, 221, 2, "BOOLEAN"],
                [368, 200, 0, 221, 1, "CONDITIONING"],
                [369, 194, 0, 221, 0, "CONDITIONING"],
                [370, 221, 0, 183, 1, "CONDITIONING"],
                [372, 196, 0, 223, 0, "IMAGE"],
                [373, 223, 1, 184, 0, "INT"],
                [374, 223, 2, 184, 1, "INT"],
                [375, 223, 1, 182, 0, "INT"],
                [376, 223, 2, 182, 1, "INT"],
            ],
            "groups": [],
            "config": {},
            "extra": {
                "frontendVersion": "1.46.6",
                "workflowRendererVersion": "LG",
                "VHS_latentpreview": False,
                "VHS_latentpreviewrate": 0,
                "VHS_MetadataImage": True,
                "VHS_KeepIntermediate": True,
                "ue_links": [],
                "ds": {
                    "scale": 0.6303940863128502,
                    "offset": [1465.7619011302713, -387.14064672449206],
                },
            },
            "version": 0.4,
        }
    }


workflow = build_workflow()
prompt = json.loads(json.dumps(workflow))
extra_pnginfo = build_extra_pnginfo()


def generate(
    prompt_text: str,
    toggle_ref: bool,
    input_img1_path: str,
    input_img2_path: str | None = None,
    unload_models: bool | None = None,
):
    bootstrap_comfyui_runtime()
    add_extra_model_paths()
    import_custom_nodes()

    from nodes import (
        CLIPLoader,
        CLIPTextEncode,
        ConditioningZeroOut,
        LoadImage,
        LoraLoader,
        NODE_CLASS_MAPPINGS,
        SaveImage,
        UNETLoader,
        VAEDecode,
        VAEEncode,
        VAELoader,
    )
    import folder_paths
    import torch

    try:
        with torch.inference_mode():
            loadimage = LoadImage()
            loadimage_159 = loadimage.load_image(image=input_img1_path)
            loadimage_163 = loadimage.load_image(
                image=input_img2_path if input_img2_path else input_img1_path
            )
            ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
            ksamplerselect_179 = ksamplerselect.EXECUTE_NORMALIZED(sampler_name="euler")
            randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
            node_181_noise_seed = prompt["181"]["inputs"]["noise_seed"] = (
                random.randint(1, 2**64)
            )
            randomnoise_181 = randomnoise.EXECUTE_NORMALIZED(
                noise_seed=node_181_noise_seed
            )
            vaeloader = VAELoader()
            vaeloader_186 = vaeloader.load_vae(
                vae_name="full_encoder_small_decoder.safetensors"
            )
            cliploader = CLIPLoader()
            cliploader_187 = cliploader.load_clip(
                clip_name="qwen_3_8b_fp8mixed.safetensors",
                type="flux2",
                device="default",
            )
            unetloader = UNETLoader()
            unetloader_191 = unetloader.load_unet(
                unet_name="Flux.2 Klein 9B/flux-2-klein-9b-fp8.safetensors",
                weight_dtype="default",
            )
            imagescaletototalpixels = NODE_CLASS_MAPPINGS["ImageScaleToTotalPixels"]()
            imagescaletototalpixels_196 = imagescaletototalpixels.EXECUTE_NORMALIZED(
                upscale_method="lanczos",
                megapixels=1,
                resolution_steps=1,
                image=get_value_at_index(loadimage_159, 0),
            )
            vaeencode = VAEEncode()
            vaeencode_195 = vaeencode.encode(
                pixels=get_value_at_index(imagescaletototalpixels_196, 0),
                vae=get_value_at_index(vaeloader_186, 0),
            )
            imagescaletototalpixels_198 = imagescaletototalpixels.EXECUTE_NORMALIZED(
                upscale_method="lanczos",
                megapixels=1,
                resolution_steps=1,
                image=get_value_at_index(loadimage_163, 0),
            )
            vaeencode_199 = vaeencode.encode(
                pixels=get_value_at_index(imagescaletototalpixels_198, 0),
                vae=get_value_at_index(vaeloader_186, 0),
            )
            loraloader = LoraLoader()
            loraloader_190 = loraloader.load_lora(
                lora_name="Flux.2 Klein 9B-base/KLEIN-Unchained-V2.safetensors",
                strength_model=0.6,
                strength_clip=1,
                model=get_value_at_index(unetloader_191, 0),
                clip=get_value_at_index(cliploader_187, 0),
            )
            loratagloader = NODE_CLASS_MAPPINGS["LoraTagLoader"]()
            loratagloader_219 = loratagloader.load_lora(
                text=prompt_text,
                model=get_value_at_index(loraloader_190, 0),
                clip=get_value_at_index(loraloader_190, 1),
            )
            cliptextencode = CLIPTextEncode()
            cliptextencode_201 = cliptextencode.encode(
                text=get_value_at_index(loratagloader_219, 2),
                clip=get_value_at_index(loratagloader_219, 1),
            )
            primitiveboolean = NODE_CLASS_MAPPINGS["PrimitiveBoolean"]()
            primitiveboolean_220 = primitiveboolean.EXECUTE_NORMALIZED(value=toggle_ref)
            pathchsageattentionkj = NODE_CLASS_MAPPINGS["PathchSageAttentionKJ"]()
            referencelatent = NODE_CLASS_MAPPINGS["ReferenceLatent"]()
            comfyswitchnode = NODE_CLASS_MAPPINGS["ComfySwitchNode"]()
            conditioningzeroout = ConditioningZeroOut()
            cfgguider = NODE_CLASS_MAPPINGS["CFGGuider"]()
            getimagesizeandcount = NODE_CLASS_MAPPINGS["GetImageSizeAndCount"]()
            flux2scheduler = NODE_CLASS_MAPPINGS["Flux2Scheduler"]()
            emptyflux2latentimage = NODE_CLASS_MAPPINGS["EmptyFlux2LatentImage"]()
            samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
            vaedecode = VAEDecode()
            saveimage = SaveImage()
            for q in range(1):
                pathchsageattentionkj_189 = pathchsageattentionkj.patch(
                    sage_attention="sageattn_qk_int8_pv_fp16_cuda",
                    allow_compile=True,
                    model=get_value_at_index(loratagloader_219, 0),
                )
                referencelatent_194 = referencelatent.EXECUTE_NORMALIZED(
                    conditioning=get_value_at_index(cliptextencode_201, 0),
                    latent=get_value_at_index(vaeencode_195, 0),
                )
                referencelatent_200 = referencelatent.EXECUTE_NORMALIZED(
                    conditioning=get_value_at_index(referencelatent_194, 0),
                    latent=get_value_at_index(vaeencode_199, 0),
                )
                comfyswitchnode_221 = comfyswitchnode.EXECUTE_NORMALIZED(
                    switch=get_value_at_index(primitiveboolean_220, 0),
                    on_false=get_value_at_index(referencelatent_194, 0),
                    on_true=get_value_at_index(referencelatent_200, 0),
                )
                conditioningzeroout_212 = conditioningzeroout.zero_out(
                    conditioning=get_value_at_index(referencelatent_194, 0)
                )
                referencelatent_193 = referencelatent.EXECUTE_NORMALIZED(
                    conditioning=get_value_at_index(conditioningzeroout_212, 0),
                    latent=get_value_at_index(vaeencode_195, 0),
                )
                cfgguider_183 = cfgguider.EXECUTE_NORMALIZED(
                    cfg=1,
                    model=get_value_at_index(pathchsageattentionkj_189, 0),
                    positive=get_value_at_index(comfyswitchnode_221, 0),
                    negative=get_value_at_index(referencelatent_193, 0),
                )
                getimagesizeandcount_223 = getimagesizeandcount.getsize(
                    image=get_value_at_index(imagescaletototalpixels_196, 0)
                )
                flux2scheduler_184 = flux2scheduler.EXECUTE_NORMALIZED(
                    steps=4,
                    width=get_value_at_index(getimagesizeandcount_223, 1),
                    height=get_value_at_index(getimagesizeandcount_223, 2),
                )
                emptyflux2latentimage_182 = emptyflux2latentimage.EXECUTE_NORMALIZED(
                    width=get_value_at_index(getimagesizeandcount_223, 1),
                    height=get_value_at_index(getimagesizeandcount_223, 2),
                    batch_size=1,
                )
                samplercustomadvanced_192 = samplercustomadvanced.EXECUTE_NORMALIZED(
                    noise=get_value_at_index(randomnoise_181, 0),
                    guider=get_value_at_index(cfgguider_183, 0),
                    sampler=get_value_at_index(ksamplerselect_179, 0),
                    sigmas=get_value_at_index(flux2scheduler_184, 0),
                    latent_image=get_value_at_index(emptyflux2latentimage_182, 0),
                )
                vaedecode_180 = vaedecode.decode(
                    samples=get_value_at_index(samplercustomadvanced_192, 0),
                    vae=get_value_at_index(vaeloader_186, 0),
                )
                saveimage_9 = saveimage.save_images(
                    filename_prefix="Flux2-Klein",
                    images=get_value_at_index(vaedecode_180, 0),
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                )
                output_dir = folder_paths.get_output_directory()
                import glob as glob_module

                pattern = os.path.join(output_dir, "Flux2-Klein_*.png")
                files = sorted(glob_module.glob(pattern), key=os.path.getmtime)
                if files:
                    return files[-1]
                return None
    finally:
        cleanup_comfyui_runtime(unload_models=unload_models)
