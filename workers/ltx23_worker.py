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
        "246": {
            "inputs": {"image": "hanfu.jpg"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"},
        },
        "250": {
            "inputs": {
                "frame_rate": ["252:93", 0],
                "loop_count": 0,
                "filename_prefix": "ltx23",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
                "images": ["249:117", 0],
                "audio": ["249:92", 0],
            },
            "class_type": "VHS_VideoCombine",
            "_meta": {"title": "Video Combine 🎥🅥🅗🅢"},
        },
        "254:218": {
            "inputs": {
                "vae_name": "LTX23_video_vae_bf16.safetensors",
                "device": "main_device",
                "weight_dtype": "bf16",
            },
            "class_type": "VAELoaderKJ",
            "_meta": {"title": "VAELoader KJ"},
        },
        "254:219": {
            "inputs": {
                "vae_name": "LTX23_audio_vae_bf16.safetensors",
                "device": "main_device",
                "weight_dtype": "bf16",
            },
            "class_type": "VAELoaderKJ",
            "_meta": {"title": "VAELoader KJ"},
        },
        "254:239": {
            "inputs": {
                "clip_name1": "gemma-3-12b-it-qat-UD-Q2_K_XL.gguf",
                "clip_name2": "ltx-2.3_text_projection_bf16.safetensors",
                "type": "ltxv",
            },
            "class_type": "DualCLIPLoaderGGUF",
            "_meta": {"title": "DualCLIPLoader (GGUF)"},
        },
        "254:240": {
            "inputs": {"unet_name": "Unknown/ltx-2.3-22b-distilled-1.1-UD-Q2_K.gguf"},
            "class_type": "UnetLoaderGGUF",
            "_meta": {"title": "Unet Loader (GGUF)"},
        },
        "252:91": {
            "inputs": {"value": 25},
            "class_type": "INTConstant",
            "_meta": {"title": "FPS"},
        },
        "252:95": {
            "inputs": {"value": 5},
            "class_type": "PrimitiveFloat",
            "_meta": {"title": "Duration (seconds)"},
        },
        "252:76": {
            "inputs": {
                "width": ["252:187:186", 1],
                "height": ["252:190:199", 0],
                "length": 97,
                "batch_size": 1,
            },
            "class_type": "EmptyLTXVLatentVideo",
            "_meta": {"title": "EmptyLTXVLatentVideo"},
        },
        "252:89": {
            "inputs": {
                "frames_number": ["252:184:194", 0],
                "frame_rate": ["252:93", 1],
                "batch_size": 1,
                "audio_vae": ["254:219", 0],
            },
            "class_type": "LTXVEmptyLatentAudio",
            "_meta": {"title": "LTXV Empty Latent Audio"},
        },
        "252:93": {
            "inputs": {"value": ["252:91", 0]},
            "class_type": "ComfyNumberConvert",
            "_meta": {"title": "Convert Number"},
        },
        "252:87": {
            "inputs": {"image": ["246", 0]},
            "class_type": "GetImageSizeAndCount",
            "_meta": {"title": "Get Image Size & Count"},
        },
        "252:75": {
            "inputs": {"img_compression": 18, "image": ["252:87", 0]},
            "class_type": "LTXVPreprocess",
            "_meta": {"title": "LTXV Preprocess"},
        },
        "252:184:96": {
            "inputs": {
                "expression": "a * b",
                "values.a": ["252:95", 0],
                "values.b": ["252:91", 0],
            },
            "class_type": "ComfyMathExpression",
            "_meta": {"title": "frames"},
        },
        "252:184:183": {
            "inputs": {"expression": "a * 8 + 1", "values.a": ["252:184:201", 1]},
            "class_type": "ComfyMathExpression",
            "_meta": {"title": "correction"},
        },
        "252:184:193": {
            "inputs": {
                "expression": "a < b",
                "values.a": ["252:184:183", 1],
                "values.b": ["252:184:196", 0],
            },
            "class_type": "ComfyMathExpression",
            "_meta": {"title": "Math Expression"},
        },
        "252:184:194": {
            "inputs": {
                "switch": ["252:184:193", 2],
                "on_false": ["252:184:183", 1],
                "on_true": ["252:184:196", 0],
            },
            "class_type": "ComfySwitchNode",
            "_meta": {"title": "Switch"},
        },
        "252:184:196": {
            "inputs": {"value": 9},
            "class_type": "PrimitiveInt",
            "_meta": {"title": "Int"},
        },
        "252:184:201": {
            "inputs": {"expression": "a / 8 + 1", "values.a": ["252:184:96", 1]},
            "class_type": "ComfyMathExpression",
            "_meta": {"title": "correction"},
        },
        "252:187:185": {
            "inputs": {"expression": "a * 0.015625", "values.a": ["252:87", 1]},
            "class_type": "ComfyMathExpression",
            "_meta": {"title": "Math Expression"},
        },
        "252:187:186": {
            "inputs": {"expression": "a * 64", "values.a": ["252:187:185", 1]},
            "class_type": "ComfyMathExpression",
            "_meta": {"title": "Math Expression"},
        },
        "252:190:188": {
            "inputs": {"expression": "a * 0.015625", "values.a": ["252:87", 2]},
            "class_type": "ComfyMathExpression",
            "_meta": {"title": "Math Expression"},
        },
        "252:190:189": {
            "inputs": {"expression": "a * 64", "values.a": ["252:190:188", 1]},
            "class_type": "ComfyMathExpression",
            "_meta": {"title": "Math Expression"},
        },
        "252:190:197": {
            "inputs": {
                "expression": "a < b",
                "variables.a": ["252:190:189", 1],
                "variables.b": ["252:190:198", 0],
            },
            "class_type": "SimpleCalculatorKJ",
            "_meta": {"title": "SimpleCalculatorKJ"},
        },
        "252:190:198": {
            "inputs": {"value": 64},
            "class_type": "PrimitiveInt",
            "_meta": {"title": "Int"},
        },
        "252:190:199": {
            "inputs": {
                "switch": ["252:190:197", 2],
                "on_false": ["252:190:189", 1],
                "on_true": ["252:190:198", 0],
            },
            "class_type": "ComfySwitchNode",
            "_meta": {"title": "Switch"},
        },
        "252:203": {
            "inputs": {"source": ["252:190:199", 0]},
            "class_type": "PreviewAny",
            "_meta": {"title": "[preview] height"},
        },
        "252:204": {
            "inputs": {"source": ["252:184:194", 0]},
            "class_type": "PreviewAny",
            "_meta": {"title": "[preview] frame_count"},
        },
        "252:202": {
            "inputs": {"source": ["252:187:186", 1]},
            "class_type": "PreviewAny",
            "_meta": {"title": "[preview] width"},
        },
        "247:86": {
            "inputs": {"text": "Dancing", "clip": ["254:239", 0]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"},
        },
        "247:84": {
            "inputs": {"conditioning": ["247:86", 0]},
            "class_type": "ConditioningZeroOut",
            "_meta": {"title": "ConditioningZeroOut"},
        },
        "247:90": {
            "inputs": {
                "frame_rate": ["252:93", 0],
                "positive": ["247:86", 0],
                "negative": ["247:84", 0],
            },
            "class_type": "LTXVConditioning",
            "_meta": {"title": "LTXVConditioning"},
        },
        "248:78": {
            "inputs": {"av_latent": ["248:79", 0]},
            "class_type": "LTXVSeparateAVLatent",
            "_meta": {"title": "LTXVSeparateAVLatent"},
        },
        "248:85": {
            "inputs": {"noise_seed": 10},
            "class_type": "RandomNoise",
            "_meta": {"title": "RandomNoise"},
        },
        "248:79": {
            "inputs": {
                "noise": ["248:85", 0],
                "guider": ["248:80", 0],
                "sampler": ["248:81", 0],
                "sigmas": ["248:255", 0],
                "latent_image": ["248:144", 0],
            },
            "class_type": "SamplerCustomAdvanced",
            "_meta": {"title": "SamplerCustomAdvanced"},
        },
        "248:145": {
            "inputs": {
                "strength": 1,
                "bypass": False,
                "vae": ["254:218", 0],
                "image": ["252:75", 0],
                "latent": ["252:76", 0],
            },
            "class_type": "LTXVImgToVideoInplace",
            "_meta": {"title": "LTXVImgToVideoInplace"},
        },
        "248:144": {
            "inputs": {"video_latent": ["248:145", 0], "audio_latent": ["252:89", 0]},
            "class_type": "LTXVConcatAVLatent",
            "_meta": {"title": "LTXVConcatAVLatent"},
        },
        "248:80": {
            "inputs": {
                "cfg": 1,
                "model": ["254:240", 0],
                "positive": ["247:90", 0],
                "negative": ["247:90", 1],
            },
            "class_type": "CFGGuider",
            "_meta": {"title": "CFG Guider"},
        },
        "248:81": {
            "inputs": {"sampler_name": "euler_ancestral"},
            "class_type": "KSamplerSelect",
            "_meta": {"title": "KSamplerSelect"},
        },
        "248:255": {
            "inputs": {
                "steps": 8,
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1,
                "latent": ["248:144", 0],
            },
            "class_type": "LTXVScheduler",
            "_meta": {"title": "LTXVScheduler"},
        },
        "249:117": {
            "inputs": {
                "spatial_tiles": 4,
                "spatial_overlap": 8,
                "temporal_tile_length": 64,
                "temporal_overlap": 8,
                "last_frame_fix": False,
                "working_device": "auto",
                "working_dtype": "auto",
                "vae": ["254:218", 0],
                "latents": ["248:78", 0],
            },
            "class_type": "LTXVSpatioTemporalTiledVAEDecode",
            "_meta": {"title": "🅛🅣🅧 LTXV Spatio Temporal Tiled VAE Decode"},
        },
        "249:92": {
            "inputs": {"samples": ["248:78", 1], "audio_vae": ["254:219", 0]},
            "class_type": "LTXVAudioVAEDecode",
            "_meta": {"title": "LTXV Audio VAE Decode"},
        },
    }


def build_extra_pnginfo() -> dict[str, Any] | None:
    return {
        "workflow": {
            "id": "481b8718-95c3-4e07-bc93-8e48caea1e72",
            "revision": 0,
            "last_node_id": 255,
            "last_link_id": 584,
            "nodes": [
                {
                    "id": 241,
                    "type": "SetNode",
                    "pos": [2149.937066132752, 639.8562534850689],
                    "size": [225, 34],
                    "flags": {"collapsed": True},
                    "order": 9,
                    "mode": 0,
                    "inputs": [
                        {"name": "CONDITIONING", "type": "CONDITIONING", "link": 560}
                    ],
                    "outputs": [
                        {"name": "CONDITIONING", "type": "CONDITIONING", "links": None}
                    ],
                    "title": "Set_POSITIVE",
                    "properties": {
                        "Node name for S&R": "SetNode",
                        "aux_id": "kijai/ComfyUI-KJNodes",
                        "previousName": "POSITIVE",
                    },
                    "widgets_values": ["POSITIVE"],
                    "color": "#332922",
                    "bgcolor": "#593930",
                },
                {
                    "id": 242,
                    "type": "SetNode",
                    "pos": [1644.6913187755354, 955.5378269586176],
                    "size": [225, 34],
                    "flags": {"collapsed": True},
                    "order": 4,
                    "mode": 0,
                    "inputs": [{"name": "VAE", "type": "VAE", "link": 561}],
                    "outputs": [{"name": "VAE", "type": "VAE", "links": None}],
                    "title": "Set_VIDEO_VAE",
                    "properties": {
                        "Node name for S&R": "SetNode",
                        "aux_id": "kijai/ComfyUI-KJNodes",
                        "previousName": "VIDEO_VAE",
                    },
                    "widgets_values": ["VIDEO_VAE"],
                    "color": "#322",
                    "bgcolor": "#533",
                },
                {
                    "id": 243,
                    "type": "SetNode",
                    "pos": [1644.6913187755358, 1006.814729693363],
                    "size": [225, 34],
                    "flags": {"collapsed": True},
                    "order": 5,
                    "mode": 0,
                    "inputs": [{"name": "VAE", "type": "VAE", "link": 562}],
                    "outputs": [{"name": "VAE", "type": "VAE", "links": None}],
                    "title": "Set_AUDIO_VAE",
                    "properties": {
                        "Node name for S&R": "SetNode",
                        "aux_id": "kijai/ComfyUI-KJNodes",
                        "previousName": "AUDIO_VAE",
                    },
                    "widgets_values": ["AUDIO_VAE"],
                    "color": "#322",
                    "bgcolor": "#533",
                },
                {
                    "id": 244,
                    "type": "SetNode",
                    "pos": [1968.8619908568148, 1342.203280177082],
                    "size": [225, 34],
                    "flags": {"collapsed": True},
                    "order": 8,
                    "mode": 0,
                    "inputs": [{"name": "IMAGE", "type": "IMAGE", "link": 563}],
                    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": None}],
                    "title": "Set_IMG",
                    "properties": {
                        "Node name for S&R": "SetNode",
                        "aux_id": "kijai/ComfyUI-KJNodes",
                        "previousName": "IMG",
                    },
                    "widgets_values": ["IMG"],
                    "color": "#2a363b",
                    "bgcolor": "#3f5159",
                },
                {
                    "id": 245,
                    "type": "SetNode",
                    "pos": [2149.9854570685884, 718.4193416328444],
                    "size": [225, 34],
                    "flags": {"collapsed": True},
                    "order": 10,
                    "mode": 0,
                    "inputs": [
                        {"name": "CONDITIONING", "type": "CONDITIONING", "link": 564}
                    ],
                    "outputs": [
                        {"name": "CONDITIONING", "type": "CONDITIONING", "links": None}
                    ],
                    "title": "Set_NEGATIVE",
                    "properties": {
                        "Node name for S&R": "SetNode",
                        "aux_id": "kijai/ComfyUI-KJNodes",
                        "previousName": "NEGATIVE",
                    },
                    "widgets_values": ["NEGATIVE"],
                    "color": "#332922",
                    "bgcolor": "#593930",
                },
                {
                    "id": 246,
                    "type": "LoadImage",
                    "pos": [1064.4322461138763, 1060.5300262366554],
                    "size": [550.6333618164062, 368.01666259765625],
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [
                        {"name": "IMAGE", "type": "IMAGE", "links": [575]},
                        {"name": "MASK", "type": "MASK", "links": None},
                    ],
                    "properties": {"Node name for S&R": "LoadImage"},
                    "widgets_values": ["hanfu.jpg", "image"],
                    "color": "#432",
                    "bgcolor": "#653",
                },
                {
                    "id": 247,
                    "type": "6fa7b9b7-78da-46d0-ab9a-fb1a59174854",
                    "pos": [1641.7432259204288, 619.7276237403507],
                    "size": [453.2833251953125, 279.75],
                    "flags": {},
                    "order": 3,
                    "mode": 0,
                    "inputs": [{"name": "clip", "type": "CLIP", "link": 565}],
                    "outputs": [
                        {"name": "positive", "type": "CONDITIONING", "links": [560]},
                        {"name": "negative", "type": "CONDITIONING", "links": [564]},
                    ],
                    "properties": {"previewExposures": []},
                    "widgets_values": ["Dancing"],
                    "color": "#232",
                    "bgcolor": "#353",
                },
                {
                    "id": 248,
                    "type": "1d6012e4-9b7e-4b97-ae02-04cdcac65e2e",
                    "pos": [1970.906523477132, 997.5291549794515],
                    "size": [285.6166687011719, 224],
                    "flags": {},
                    "order": 6,
                    "mode": 0,
                    "inputs": [
                        {
                            "label": "distilled_model",
                            "name": "model",
                            "type": "MODEL",
                            "link": 566,
                        },
                        {
                            "label": "video_latent",
                            "name": "latent",
                            "type": "LATENT",
                            "link": 567,
                        },
                        {"name": "audio_latent", "type": "LATENT", "link": 568},
                    ],
                    "outputs": [
                        {"name": "video_latent", "type": "LATENT", "links": [580]},
                        {"name": "audio_latent", "type": "LATENT", "links": [581]},
                    ],
                    "properties": {"previewExposures": []},
                    "widgets_values": [1, "euler_ancestral", 8],
                    "color": "#223",
                    "bgcolor": "#335",
                },
                {
                    "id": 249,
                    "type": "9bf05148-ab95-43e1-b4c6-299bac0163f5",
                    "pos": [2308.444700880831, 998.1035328636433],
                    "size": [225, 148],
                    "flags": {},
                    "order": 11,
                    "mode": 0,
                    "inputs": [
                        {
                            "label": "video_latent",
                            "name": "latents",
                            "type": "LATENT",
                            "link": 580,
                        },
                        {
                            "label": "audio_latent",
                            "name": "samples",
                            "type": "LATENT",
                            "link": 581,
                        },
                    ],
                    "outputs": [
                        {
                            "label": "frames",
                            "name": "image",
                            "type": "IMAGE",
                            "links": [571],
                        },
                        {
                            "label": "audio",
                            "name": "Audio",
                            "type": "AUDIO",
                            "links": [572],
                        },
                        {
                            "label": "fps",
                            "name": "FLOAT",
                            "type": "FLOAT",
                            "links": [573],
                        },
                    ],
                    "properties": {"previewExposures": []},
                    "color": "#322",
                    "bgcolor": "#533",
                },
                {
                    "id": 251,
                    "type": "SetNode",
                    "pos": [1968.925765058893, 1297.3665123390983],
                    "size": [225, 34],
                    "flags": {"collapsed": True},
                    "order": 7,
                    "mode": 0,
                    "inputs": [{"name": "FLOAT", "type": "FLOAT", "link": 574}],
                    "outputs": [{"name": "FLOAT", "type": "FLOAT", "links": []}],
                    "title": "Set_FPS",
                    "properties": {
                        "Node name for S&R": "SetNode",
                        "aux_id": "kijai/ComfyUI-KJNodes",
                        "previousName": "FPS",
                    },
                    "widgets_values": ["FPS"],
                    "color": "#232",
                    "bgcolor": "#353",
                },
                {
                    "id": 252,
                    "type": "68322638-2fa0-48e4-b108-07c6bf1d2390",
                    "pos": [1636.3966951434431, 1061.5020898592747],
                    "size": [299, 212],
                    "flags": {},
                    "order": 2,
                    "mode": 0,
                    "inputs": [
                        {
                            "label": "duration (seconds)",
                            "name": "value",
                            "type": "FLOAT",
                            "widget": {"name": "value"},
                            "link": None,
                        },
                        {
                            "label": "fps",
                            "name": "value_1",
                            "type": "INT",
                            "widget": {"name": "value_1"},
                            "link": None,
                        },
                        {
                            "label": "input_image",
                            "name": "image",
                            "type": "IMAGE",
                            "link": 575,
                        },
                    ],
                    "outputs": [
                        {
                            "label": "video_latent",
                            "name": "LATENT",
                            "type": "LATENT",
                            "links": [567],
                        },
                        {
                            "label": "audio_latent",
                            "name": "Latent",
                            "type": "LATENT",
                            "links": [568],
                        },
                        {
                            "label": "fps",
                            "name": "FLOAT",
                            "type": "FLOAT",
                            "links": [574],
                        },
                        {
                            "label": "compressed_image",
                            "name": "output_image",
                            "type": "IMAGE",
                            "links": [563],
                        },
                    ],
                    "properties": {"previewExposures": []},
                    "widgets_values": [5, 25],
                    "color": "#432",
                    "bgcolor": "#653",
                },
                {
                    "id": 254,
                    "type": "30f35274-1bfd-4ef1-8a1f-83077a47ab02",
                    "pos": [1062.6818382171791, 614.1723549397643],
                    "size": [552.2166748046875, 388],
                    "flags": {},
                    "order": 1,
                    "mode": 0,
                    "inputs": [
                        {
                            "label": "distilled_model",
                            "name": "unet_name",
                            "type": "COMBO",
                            "widget": {"name": "unet_name"},
                            "link": None,
                        },
                        {
                            "label": "gemma3_clip",
                            "name": "clip_name1",
                            "type": "COMBO",
                            "widget": {"name": "clip_name1"},
                            "link": None,
                        },
                        {
                            "label": "ltx_encoder",
                            "name": "clip_name2",
                            "type": "COMBO",
                            "widget": {"name": "clip_name2"},
                            "link": None,
                        },
                        {
                            "label": "video_vae",
                            "name": "vae_name",
                            "type": "COMBO",
                            "widget": {"name": "vae_name"},
                            "link": None,
                        },
                        {
                            "label": "audio_vae",
                            "name": "vae_name_1",
                            "type": "COMBO",
                            "widget": {"name": "vae_name_1"},
                            "link": None,
                        },
                    ],
                    "outputs": [
                        {
                            "label": "model",
                            "name": "MODEL_1",
                            "type": "MODEL",
                            "links": [566],
                        },
                        {
                            "label": "clip",
                            "name": "CLIP",
                            "type": "CLIP",
                            "links": [565],
                        },
                        {
                            "label": "video_vae",
                            "name": "VAE",
                            "type": "VAE",
                            "links": [561],
                        },
                        {
                            "label": "audio_vae",
                            "name": "VAE_1",
                            "type": "VAE",
                            "links": [562],
                        },
                    ],
                    "properties": {"previewExposures": []},
                    "widgets_values": [
                        "Unknown/ltx-2.3-22b-distilled-1.1-UD-Q2_K.gguf",
                        "gemma-3-12b-it-qat-UD-Q2_K_XL.gguf",
                        "ltx-2.3_text_projection_bf16.safetensors",
                        "LTX23_video_vae_bf16.safetensors",
                        "LTX23_audio_vae_bf16.safetensors",
                    ],
                    "color": "#432",
                    "bgcolor": "#653",
                },
                {
                    "id": 250,
                    "type": "VHS_VideoCombine",
                    "pos": [2590.392731722424, 629.1446065482279],
                    "size": [459.4666748046875, 1139.0400146484376],
                    "flags": {},
                    "order": 12,
                    "mode": 0,
                    "inputs": [
                        {"name": "images", "type": "IMAGE", "link": 571},
                        {"name": "audio", "shape": 7, "type": "AUDIO", "link": 572},
                        {
                            "name": "meta_batch",
                            "shape": 7,
                            "type": "VHS_BatchManager",
                            "link": None,
                        },
                        {"name": "vae", "shape": 7, "type": "VAE", "link": None},
                        {
                            "name": "frame_rate",
                            "type": "FLOAT",
                            "widget": {"name": "frame_rate"},
                            "link": 573,
                        },
                    ],
                    "outputs": [
                        {"name": "Filenames", "type": "VHS_FILENAMES", "links": None}
                    ],
                    "properties": {"Node name for S&R": "VHS_VideoCombine"},
                    "widgets_values": {
                        "frame_rate": 24,
                        "loop_count": 0,
                        "filename_prefix": "ltx23",
                        "format": "video/h264-mp4",
                        "pix_fmt": "yuv420p",
                        "crf": 19,
                        "save_metadata": True,
                        "trim_to_audio": False,
                        "pingpong": False,
                        "save_output": True,
                        "videopreview": {
                            "hidden": False,
                            "paused": False,
                            "params": {
                                "filename": "ltx23_00010-audio.webm",
                                "subfolder": "",
                                "type": "output",
                                "format": "video/webm",
                                "frame_rate": 25,
                                "workflow": "ltx23_00010.png",
                                "fullpath": "/comfy/output/ltx23_00010-audio.webm",
                            },
                        },
                    },
                },
            ],
            "links": [
                [560, 247, 0, 241, 0, "CONDITIONING"],
                [561, 254, 2, 242, 0, "VAE"],
                [562, 254, 3, 243, 0, "VAE"],
                [563, 252, 3, 244, 0, "IMAGE"],
                [564, 247, 1, 245, 0, "CONDITIONING"],
                [565, 254, 1, 247, 0, "CLIP"],
                [566, 254, 0, 248, 0, "MODEL"],
                [567, 252, 0, 248, 1, "LATENT"],
                [568, 252, 1, 248, 2, "LATENT"],
                [571, 249, 0, 250, 0, "IMAGE"],
                [572, 249, 1, 250, 1, "AUDIO"],
                [573, 249, 2, 250, 4, "FLOAT"],
                [574, 252, 2, 251, 0, "FLOAT"],
                [575, 246, 0, 252, 2, "IMAGE"],
                [580, 248, 0, 249, 0, "LATENT"],
                [581, 248, 1, 249, 1, "LATENT"],
            ],
            "groups": [],
            "definitions": {
                "subgraphs": [
                    {
                        "id": "6fa7b9b7-78da-46d0-ab9a-fb1a59174854",
                        "version": 1,
                        "state": {
                            "lastGroupId": 0,
                            "lastNodeId": 255,
                            "lastLinkId": 584,
                            "lastRerouteId": 0,
                        },
                        "revision": 0,
                        "config": {},
                        "name": "Condition",
                        "inputNode": {
                            "id": -10,
                            "bounding": [
                                -563.7823189977273,
                                2840.768918305491,
                                128,
                                88,
                            ],
                        },
                        "outputNode": {
                            "id": -20,
                            "bounding": [1359.380792318475, 2840.768918305491, 128, 88],
                        },
                        "inputs": [
                            {
                                "id": "80966e7d-b015-4db2-bca9-0bbdf05fee87",
                                "name": "clip",
                                "type": "CLIP",
                                "linkIds": [212],
                                "localized_name": "clip",
                                "pos": [-459.7823189977273, 2864.768918305491],
                            },
                            {
                                "id": "7bbe0da3-d94b-4b46-b904-cc098db1dc88",
                                "name": "text",
                                "type": "STRING",
                                "linkIds": [346],
                                "pos": [-459.7823189977273, 2884.768918305491],
                            },
                        ],
                        "outputs": [
                            {
                                "id": "d3a86db0-c31f-48ac-bdab-53806ae18bba",
                                "name": "positive",
                                "type": "CONDITIONING",
                                "linkIds": [134, 134],
                                "localized_name": "positive",
                                "pos": [1383.380792318475, 2864.768918305491],
                            },
                            {
                                "id": "9e2c4f04-aee3-417b-a065-857c8b1f69fc",
                                "name": "negative",
                                "type": "CONDITIONING",
                                "linkIds": [135, 135],
                                "localized_name": "negative",
                                "pos": [1383.380792318475, 2884.768918305491],
                            },
                        ],
                        "widgets": [],
                        "nodes": [
                            {
                                "id": 86,
                                "type": "CLIPTextEncode",
                                "pos": [-375.7823189977273, 2632.937729903558],
                                "size": [590, 350],
                                "flags": {},
                                "order": 2,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "clip",
                                        "name": "clip",
                                        "type": "CLIP",
                                        "link": 212,
                                    },
                                    {
                                        "localized_name": "text",
                                        "name": "text",
                                        "type": "STRING",
                                        "widget": {"name": "text"},
                                        "link": 346,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "CONDITIONING",
                                        "name": "CONDITIONING",
                                        "type": "CONDITIONING",
                                        "links": [136, 143],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "CLIPTextEncode",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.3.56",
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                    "ue_properties": {
                                        "widget_ue_connectable": {},
                                        "version": "7.1",
                                        "input_ue_unconnectable": {},
                                    },
                                },
                                "widgets_values": ["Dancing"],
                                "color": "#232",
                                "bgcolor": "#353",
                            },
                            {
                                "id": 84,
                                "type": "ConditioningZeroOut",
                                "pos": [706.0610793641832, 2653.533276675919],
                                "size": [225, 48],
                                "flags": {},
                                "order": 1,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "conditioning",
                                        "name": "conditioning",
                                        "type": "CONDITIONING",
                                        "link": 136,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "CONDITIONING",
                                        "name": "CONDITIONING",
                                        "type": "CONDITIONING",
                                        "links": [144],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "ConditioningZeroOut"
                                },
                                "widgets_values": [],
                            },
                            {
                                "id": 90,
                                "type": "LTXVConditioning",
                                "pos": [706.8289225263642, 2773.0097418575724],
                                "size": [270, 104],
                                "flags": {},
                                "order": 3,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "positive",
                                        "name": "positive",
                                        "type": "CONDITIONING",
                                        "link": 143,
                                    },
                                    {
                                        "localized_name": "negative",
                                        "name": "negative",
                                        "type": "CONDITIONING",
                                        "link": 144,
                                    },
                                    {
                                        "localized_name": "frame_rate",
                                        "name": "frame_rate",
                                        "type": "FLOAT",
                                        "widget": {"name": "frame_rate"},
                                        "link": 316,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "positive",
                                        "name": "positive",
                                        "type": "CONDITIONING",
                                        "links": [134],
                                    },
                                    {
                                        "localized_name": "negative",
                                        "name": "negative",
                                        "type": "CONDITIONING",
                                        "links": [135],
                                    },
                                ],
                                "properties": {
                                    "Node name for S&R": "LTXVConditioning",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.3.56",
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                    "ue_properties": {
                                        "widget_ue_connectable": {},
                                        "version": "7.1",
                                        "input_ue_unconnectable": {},
                                    },
                                },
                                "widgets_values": [25],
                            },
                            {
                                "id": 141,
                                "type": "GetNode",
                                "pos": [433.0353682964486, 2860.0632792909064],
                                "size": [225, 34],
                                "flags": {"collapsed": True},
                                "order": 0,
                                "mode": 0,
                                "inputs": [],
                                "outputs": [
                                    {"name": "FLOAT", "type": "FLOAT", "links": [316]}
                                ],
                                "title": "Get_FPS",
                                "properties": {
                                    "Node name for S&R": "GetNode",
                                    "aux_id": "kijai/ComfyUI-KJNodes",
                                },
                                "widgets_values": ["FPS"],
                                "color": "#232",
                                "bgcolor": "#353",
                            },
                        ],
                        "groups": [],
                        "links": [
                            {
                                "id": 136,
                                "origin_id": 86,
                                "origin_slot": 0,
                                "target_id": 84,
                                "target_slot": 0,
                                "type": "CONDITIONING",
                            },
                            {
                                "id": 143,
                                "origin_id": 86,
                                "origin_slot": 0,
                                "target_id": 90,
                                "target_slot": 0,
                                "type": "CONDITIONING",
                            },
                            {
                                "id": 144,
                                "origin_id": 84,
                                "origin_slot": 0,
                                "target_id": 90,
                                "target_slot": 1,
                                "type": "CONDITIONING",
                            },
                            {
                                "id": 212,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 86,
                                "target_slot": 0,
                                "type": "CLIP",
                            },
                            {
                                "id": 134,
                                "origin_id": 90,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 0,
                                "type": "CONDITIONING",
                            },
                            {
                                "id": 135,
                                "origin_id": 90,
                                "origin_slot": 1,
                                "target_id": -20,
                                "target_slot": 1,
                                "type": "CONDITIONING",
                            },
                            {
                                "id": 316,
                                "origin_id": 141,
                                "origin_slot": 0,
                                "target_id": 90,
                                "target_slot": 2,
                                "type": "FLOAT",
                            },
                            {
                                "id": 346,
                                "origin_id": -10,
                                "origin_slot": 1,
                                "target_id": 86,
                                "target_slot": 1,
                                "type": "STRING",
                            },
                        ],
                        "extra": {},
                    },
                    {
                        "id": "1d6012e4-9b7e-4b97-ae02-04cdcac65e2e",
                        "version": 1,
                        "state": {
                            "lastGroupId": 0,
                            "lastNodeId": 255,
                            "lastLinkId": 584,
                            "lastRerouteId": 0,
                        },
                        "revision": 0,
                        "config": {},
                        "name": "Samper",
                        "inputNode": {
                            "id": -10,
                            "bounding": [
                                509.8291495725796,
                                2510.715893182679,
                                132.41666412353516,
                                168,
                            ],
                        },
                        "outputNode": {
                            "id": -20,
                            "bounding": [
                                1622.2513021621314,
                                2530.715893182679,
                                128,
                                88,
                            ],
                        },
                        "inputs": [
                            {
                                "id": "6a0dee37-f4fb-435d-b04f-c0d6995d0dcf",
                                "name": "model",
                                "type": "MODEL",
                                "linkIds": [270],
                                "label": "distilled_model",
                                "pos": [618.2458136961147, 2534.715893182679],
                            },
                            {
                                "id": "8d470d71-f74e-4b8e-996a-6d90109f453e",
                                "name": "latent",
                                "type": "LATENT",
                                "linkIds": [342],
                                "label": "video_latent",
                                "pos": [618.2458136961147, 2554.715893182679],
                            },
                            {
                                "id": "6c4fe426-2c57-4121-a134-a010be134e5d",
                                "name": "audio_latent",
                                "type": "LATENT",
                                "linkIds": [345],
                                "pos": [618.2458136961147, 2574.715893182679],
                            },
                            {
                                "id": "4b22ce3e-ecb9-4076-830e-31f5c54980ce",
                                "name": "cfg",
                                "type": "FLOAT",
                                "linkIds": [522],
                                "pos": [618.2458136961147, 2594.715893182679],
                            },
                            {
                                "id": "b09ec3ea-a7e2-4941-9f63-dfe1e05b06a7",
                                "name": "sampler_name",
                                "type": "COMBO",
                                "linkIds": [523],
                                "pos": [618.2458136961147, 2614.715893182679],
                            },
                            {
                                "id": "cc7d3d7f-9d56-4979-82b2-f1cf55ef1b2e",
                                "name": "steps",
                                "type": "INT",
                                "linkIds": [584],
                                "pos": [618.2458136961147, 2634.715893182679],
                            },
                        ],
                        "outputs": [
                            {
                                "id": "2f9f9cd5-f794-4b76-b685-6e7122821c5b",
                                "name": "video_latent",
                                "type": "LATENT",
                                "linkIds": [168, 168],
                                "localized_name": "video_latent",
                                "pos": [1646.2513021621314, 2554.715893182679],
                            },
                            {
                                "id": "a2174901-4229-4bb4-8e91-388a604d5e6a",
                                "name": "audio_latent",
                                "type": "LATENT",
                                "linkIds": [178],
                                "localized_name": "audio_latent",
                                "pos": [1646.2513021621314, 2574.715893182679],
                            },
                        ],
                        "widgets": [],
                        "nodes": [
                            {
                                "id": 78,
                                "type": "LTXVSeparateAVLatent",
                                "pos": [1337.2513021621314, 2303.6481293307934],
                                "size": [225, 72],
                                "flags": {},
                                "order": 5,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "av_latent",
                                        "name": "av_latent",
                                        "type": "LATENT",
                                        "link": 195,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "video_latent",
                                        "name": "video_latent",
                                        "type": "LATENT",
                                        "links": [168],
                                    },
                                    {
                                        "localized_name": "audio_latent",
                                        "name": "audio_latent",
                                        "type": "LATENT",
                                        "links": [178],
                                    },
                                ],
                                "properties": {
                                    "Node name for S&R": "LTXVSeparateAVLatent",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.5.1",
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                    "ue_properties": {
                                        "widget_ue_connectable": {},
                                        "version": "7.1",
                                        "input_ue_unconnectable": {},
                                    },
                                },
                                "widgets_values": [],
                            },
                            {
                                "id": 85,
                                "type": "RandomNoise",
                                "pos": [734.9390428474774, 2302.4735353079445],
                                "size": [302.8999938964844, 82],
                                "flags": {},
                                "order": 0,
                                "mode": 0,
                                "inputs": [],
                                "outputs": [
                                    {
                                        "localized_name": "NOISE",
                                        "name": "NOISE",
                                        "type": "NOISE",
                                        "links": [128],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "RandomNoise",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.3.56",
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                    "ue_properties": {
                                        "widget_ue_connectable": {},
                                        "version": "7.1",
                                        "input_ue_unconnectable": {},
                                    },
                                },
                                "widgets_values": [10, "fixed"],
                            },
                            {
                                "id": 79,
                                "type": "SamplerCustomAdvanced",
                                "pos": [1077.8423232607834, 2302.4735353079445],
                                "size": [225, 172],
                                "flags": {},
                                "order": 6,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "noise",
                                        "name": "noise",
                                        "type": "NOISE",
                                        "link": 128,
                                    },
                                    {
                                        "localized_name": "guider",
                                        "name": "guider",
                                        "type": "GUIDER",
                                        "link": 129,
                                    },
                                    {
                                        "localized_name": "sampler",
                                        "name": "sampler",
                                        "type": "SAMPLER",
                                        "link": 130,
                                    },
                                    {
                                        "localized_name": "sigmas",
                                        "name": "sigmas",
                                        "type": "SIGMAS",
                                        "link": 583,
                                    },
                                    {
                                        "localized_name": "latent_image",
                                        "name": "latent_image",
                                        "type": "LATENT",
                                        "link": 334,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "output",
                                        "name": "output",
                                        "type": "LATENT",
                                        "links": [195],
                                    },
                                    {
                                        "localized_name": "denoised_output",
                                        "name": "denoised_output",
                                        "type": "LATENT",
                                        "links": [],
                                    },
                                ],
                                "properties": {
                                    "Node name for S&R": "SamplerCustomAdvanced",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.3.60",
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                    "ue_properties": {
                                        "widget_ue_connectable": {},
                                        "version": "7.1",
                                        "input_ue_unconnectable": {},
                                    },
                                },
                                "widgets_values": [],
                            },
                            {
                                "id": 145,
                                "type": "LTXVImgToVideoInplace",
                                "pos": [740.2213251721514, 3026.4927648493035],
                                "size": [270, 156],
                                "flags": {},
                                "order": 10,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "vae",
                                        "name": "vae",
                                        "type": "VAE",
                                        "link": 368,
                                    },
                                    {
                                        "localized_name": "image",
                                        "name": "image",
                                        "type": "IMAGE",
                                        "link": 352,
                                    },
                                    {
                                        "localized_name": "latent",
                                        "name": "latent",
                                        "type": "LATENT",
                                        "link": 342,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "latent",
                                        "name": "latent",
                                        "type": "LATENT",
                                        "links": [322],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "LTXVImgToVideoInplace",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.7.0",
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                    "ue_properties": {
                                        "widget_ue_connectable": {},
                                        "version": "7.1",
                                        "input_ue_unconnectable": {},
                                    },
                                },
                                "widgets_values": [1, False],
                            },
                            {
                                "id": 144,
                                "type": "LTXVConcatAVLatent",
                                "pos": [1102.1311550477958, 2925.317122002979],
                                "size": [225, 72],
                                "flags": {},
                                "order": 9,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "video_latent",
                                        "name": "video_latent",
                                        "type": "LATENT",
                                        "link": 322,
                                    },
                                    {
                                        "localized_name": "audio_latent",
                                        "name": "audio_latent",
                                        "type": "LATENT",
                                        "link": 345,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "latent",
                                        "name": "latent",
                                        "type": "LATENT",
                                        "links": [334, 582],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "LTXVConcatAVLatent",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.7.0",
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                    "ue_properties": {
                                        "widget_ue_connectable": {},
                                        "version": "7.1",
                                        "input_ue_unconnectable": {},
                                    },
                                },
                                "widgets_values": [],
                            },
                            {
                                "id": 149,
                                "type": "GetNode",
                                "pos": [324.31506191854476, 3021.450785613783],
                                "size": [225, 104],
                                "flags": {},
                                "order": 1,
                                "mode": 0,
                                "inputs": [],
                                "outputs": [
                                    {"name": "IMAGE", "type": "IMAGE", "links": [352]}
                                ],
                                "title": "Get_IMG",
                                "properties": {
                                    "Node name for S&R": "GetNode",
                                    "aux_id": "kijai/ComfyUI-KJNodes",
                                },
                                "widgets_values": ["IMG"],
                                "color": "#2a363b",
                                "bgcolor": "#3f5159",
                            },
                            {
                                "id": 151,
                                "type": "GetNode",
                                "pos": [305.72767062711773, 2414.262670093835],
                                "size": [225, 104],
                                "flags": {},
                                "order": 2,
                                "mode": 0,
                                "inputs": [],
                                "outputs": [
                                    {
                                        "name": "CONDITIONING",
                                        "type": "CONDITIONING",
                                        "links": [354],
                                    }
                                ],
                                "title": "Get_NEGATIVE",
                                "properties": {
                                    "Node name for S&R": "GetNode",
                                    "aux_id": "kijai/ComfyUI-KJNodes",
                                },
                                "widgets_values": ["NEGATIVE"],
                                "color": "#332922",
                                "bgcolor": "#593930",
                            },
                            {
                                "id": 150,
                                "type": "GetNode",
                                "pos": [309.8582020252125, 2260.4003755148015],
                                "size": [225, 104],
                                "flags": {},
                                "order": 3,
                                "mode": 0,
                                "inputs": [],
                                "outputs": [
                                    {
                                        "name": "CONDITIONING",
                                        "type": "CONDITIONING",
                                        "links": [353],
                                    }
                                ],
                                "title": "Get_POSITIVE",
                                "properties": {
                                    "Node name for S&R": "GetNode",
                                    "aux_id": "kijai/ComfyUI-KJNodes",
                                },
                                "widgets_values": ["POSITIVE"],
                                "color": "#332922",
                                "bgcolor": "#593930",
                            },
                            {
                                "id": 158,
                                "type": "GetNode",
                                "pos": [322.2497962194974, 2859.327428238555],
                                "size": [225, 104],
                                "flags": {"collapsed": False},
                                "order": 4,
                                "mode": 0,
                                "inputs": [],
                                "outputs": [
                                    {"name": "VAE", "type": "VAE", "links": [368]}
                                ],
                                "title": "Get_VIDEO_VAE",
                                "properties": {
                                    "Node name for S&R": "GetNode",
                                    "aux_id": "kijai/ComfyUI-KJNodes",
                                },
                                "widgets_values": ["VIDEO_VAE"],
                                "color": "#322",
                                "bgcolor": "#533",
                            },
                            {
                                "id": 80,
                                "type": "CFGGuider",
                                "pos": [733.9712993059097, 2452.4735353079445],
                                "size": [303.8666687011719, 128],
                                "flags": {},
                                "order": 7,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "model",
                                        "name": "model",
                                        "type": "MODEL",
                                        "link": 270,
                                    },
                                    {
                                        "localized_name": "positive",
                                        "name": "positive",
                                        "type": "CONDITIONING",
                                        "link": 353,
                                    },
                                    {
                                        "localized_name": "negative",
                                        "name": "negative",
                                        "type": "CONDITIONING",
                                        "link": 354,
                                    },
                                    {
                                        "localized_name": "cfg",
                                        "name": "cfg",
                                        "type": "FLOAT",
                                        "widget": {"name": "cfg"},
                                        "link": 522,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "GUIDER",
                                        "name": "GUIDER",
                                        "type": "GUIDER",
                                        "links": [129],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "CFGGuider",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.3.64",
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                    "ue_properties": {
                                        "widget_ue_connectable": {},
                                        "version": "7.1",
                                        "input_ue_unconnectable": {},
                                    },
                                },
                                "widgets_values": [1],
                            },
                            {
                                "id": 81,
                                "type": "KSamplerSelect",
                                "pos": [734.9390428474774, 2622.4735353079445],
                                "size": [302.8999938964844, 80],
                                "flags": {},
                                "order": 8,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "sampler_name",
                                        "name": "sampler_name",
                                        "type": "COMBO",
                                        "widget": {"name": "sampler_name"},
                                        "link": 523,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "SAMPLER",
                                        "name": "SAMPLER",
                                        "type": "SAMPLER",
                                        "links": [130],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "KSamplerSelect",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.3.56",
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                    "ue_properties": {
                                        "widget_ue_connectable": {},
                                        "version": "7.1",
                                        "input_ue_unconnectable": {},
                                    },
                                },
                                "widgets_values": ["euler_ancestral"],
                            },
                            {
                                "id": 255,
                                "type": "LTXVScheduler",
                                "pos": [1102.5661292929572, 2704.3188977418017],
                                "size": [270, 172],
                                "flags": {},
                                "order": 11,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "latent",
                                        "name": "latent",
                                        "shape": 7,
                                        "type": "LATENT",
                                        "link": 582,
                                    },
                                    {
                                        "localized_name": "steps",
                                        "name": "steps",
                                        "type": "INT",
                                        "widget": {"name": "steps"},
                                        "link": 584,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "SIGMAS",
                                        "name": "SIGMAS",
                                        "type": "SIGMAS",
                                        "links": [583],
                                    }
                                ],
                                "properties": {"Node name for S&R": "LTXVScheduler"},
                                "widgets_values": [8, 2.05, 0.95, True, 0.1],
                            },
                        ],
                        "groups": [],
                        "links": [
                            {
                                "id": 195,
                                "origin_id": 79,
                                "origin_slot": 0,
                                "target_id": 78,
                                "target_slot": 0,
                                "type": "LATENT",
                            },
                            {
                                "id": 128,
                                "origin_id": 85,
                                "origin_slot": 0,
                                "target_id": 79,
                                "target_slot": 0,
                                "type": "NOISE",
                            },
                            {
                                "id": 129,
                                "origin_id": 80,
                                "origin_slot": 0,
                                "target_id": 79,
                                "target_slot": 1,
                                "type": "GUIDER",
                            },
                            {
                                "id": 130,
                                "origin_id": 81,
                                "origin_slot": 0,
                                "target_id": 79,
                                "target_slot": 2,
                                "type": "SAMPLER",
                            },
                            {
                                "id": 168,
                                "origin_id": 78,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 0,
                                "type": "LATENT",
                            },
                            {
                                "id": 178,
                                "origin_id": 78,
                                "origin_slot": 1,
                                "target_id": -20,
                                "target_slot": 1,
                                "type": "LATENT",
                            },
                            {
                                "id": 270,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 80,
                                "target_slot": 0,
                                "type": "MODEL",
                            },
                            {
                                "id": 322,
                                "origin_id": 145,
                                "origin_slot": 0,
                                "target_id": 144,
                                "target_slot": 0,
                                "type": "LATENT",
                            },
                            {
                                "id": 334,
                                "origin_id": 144,
                                "origin_slot": 0,
                                "target_id": 79,
                                "target_slot": 4,
                                "type": "LATENT",
                            },
                            {
                                "id": 342,
                                "origin_id": -10,
                                "origin_slot": 1,
                                "target_id": 145,
                                "target_slot": 2,
                                "type": "LATENT",
                            },
                            {
                                "id": 345,
                                "origin_id": -10,
                                "origin_slot": 2,
                                "target_id": 144,
                                "target_slot": 1,
                                "type": "LATENT",
                            },
                            {
                                "id": 352,
                                "origin_id": 149,
                                "origin_slot": 0,
                                "target_id": 145,
                                "target_slot": 1,
                                "type": "IMAGE",
                            },
                            {
                                "id": 353,
                                "origin_id": 150,
                                "origin_slot": 0,
                                "target_id": 80,
                                "target_slot": 1,
                                "type": "CONDITIONING",
                            },
                            {
                                "id": 354,
                                "origin_id": 151,
                                "origin_slot": 0,
                                "target_id": 80,
                                "target_slot": 2,
                                "type": "CONDITIONING",
                            },
                            {
                                "id": 368,
                                "origin_id": 158,
                                "origin_slot": 0,
                                "target_id": 145,
                                "target_slot": 0,
                                "type": "VAE",
                            },
                            {
                                "id": 522,
                                "origin_id": -10,
                                "origin_slot": 3,
                                "target_id": 80,
                                "target_slot": 3,
                                "type": "FLOAT",
                            },
                            {
                                "id": 523,
                                "origin_id": -10,
                                "origin_slot": 4,
                                "target_id": 81,
                                "target_slot": 0,
                                "type": "COMBO",
                            },
                            {
                                "id": 582,
                                "origin_id": 144,
                                "origin_slot": 0,
                                "target_id": 255,
                                "target_slot": 0,
                                "type": "LATENT",
                            },
                            {
                                "id": 583,
                                "origin_id": 255,
                                "origin_slot": 0,
                                "target_id": 79,
                                "target_slot": 3,
                                "type": "SIGMAS",
                            },
                            {
                                "id": 584,
                                "origin_id": -10,
                                "origin_slot": 5,
                                "target_id": 255,
                                "target_slot": 1,
                                "type": "INT",
                            },
                        ],
                        "extra": {},
                    },
                    {
                        "id": "9bf05148-ab95-43e1-b4c6-299bac0163f5",
                        "version": 1,
                        "state": {
                            "lastGroupId": 0,
                            "lastNodeId": 255,
                            "lastLinkId": 584,
                            "lastRerouteId": 0,
                        },
                        "revision": 0,
                        "config": {},
                        "name": "Decoder",
                        "inputNode": {
                            "id": -10,
                            "bounding": [
                                3323.6423077831214,
                                2854.744326499678,
                                128,
                                88,
                            ],
                        },
                        "outputNode": {
                            "id": -20,
                            "bounding": [
                                4386.265617878105,
                                2884.744326499678,
                                128,
                                108,
                            ],
                        },
                        "inputs": [
                            {
                                "id": "f74cad52-0a26-48d1-b315-d92386a74573",
                                "name": "latents",
                                "type": "LATENT",
                                "linkIds": [262],
                                "label": "video_latent",
                                "pos": [3427.6423077831214, 2878.744326499678],
                            },
                            {
                                "id": "6e189ef9-ac41-4b60-be75-1f540d58eb2c",
                                "name": "samples",
                                "type": "LATENT",
                                "linkIds": [263],
                                "label": "audio_latent",
                                "pos": [3427.6423077831214, 2898.744326499678],
                            },
                        ],
                        "outputs": [
                            {
                                "id": "eb1a056f-6bb1-467b-a266-d492db8d6dcf",
                                "name": "image",
                                "type": "IMAGE",
                                "linkIds": [468],
                                "label": "frames",
                                "pos": [4410.265617878105, 2908.744326499678],
                            },
                            {
                                "id": "db72b3b7-6639-4eb4-ba5c-58fc8837530a",
                                "name": "Audio",
                                "type": "AUDIO",
                                "linkIds": [469],
                                "label": "audio",
                                "pos": [4410.265617878105, 2928.744326499678],
                            },
                            {
                                "id": "019db135-2fe3-453e-bc53-1a3c78b2e053",
                                "name": "FLOAT",
                                "type": "FLOAT",
                                "linkIds": [470],
                                "label": "fps",
                                "pos": [4410.265617878105, 2948.744326499678],
                            },
                        ],
                        "widgets": [],
                        "nodes": [
                            {
                                "id": 159,
                                "type": "GetNode",
                                "pos": [3088.701383072742, 2624.178242961405],
                                "size": [225, 104],
                                "flags": {"collapsed": False},
                                "order": 0,
                                "mode": 0,
                                "inputs": [],
                                "outputs": [
                                    {"name": "VAE", "type": "VAE", "links": [369]}
                                ],
                                "title": "Get_VIDEO_VAE",
                                "properties": {
                                    "Node name for S&R": "GetNode",
                                    "aux_id": "kijai/ComfyUI-KJNodes",
                                },
                                "widgets_values": ["VIDEO_VAE"],
                                "color": "#322",
                                "bgcolor": "#533",
                            },
                            {
                                "id": 160,
                                "type": "GetNode",
                                "pos": [3084.1041927086344, 3017.6977381290344],
                                "size": [225, 104],
                                "flags": {"collapsed": False},
                                "order": 1,
                                "mode": 0,
                                "inputs": [],
                                "outputs": [
                                    {"name": "VAE", "type": "VAE", "links": [370]}
                                ],
                                "title": "Get_AUDIO_VAE",
                                "properties": {
                                    "Node name for S&R": "GetNode",
                                    "aux_id": "kijai/ComfyUI-KJNodes",
                                },
                                "widgets_values": ["AUDIO_VAE"],
                                "color": "#322",
                                "bgcolor": "#533",
                            },
                            {
                                "id": 117,
                                "type": "LTXVSpatioTemporalTiledVAEDecode",
                                "pos": [3511.6423077831214, 2625.676773662002],
                                "size": [409.0333251953125, 316],
                                "flags": {},
                                "order": 4,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "vae",
                                        "name": "vae",
                                        "type": "VAE",
                                        "link": 369,
                                    },
                                    {
                                        "localized_name": "latents",
                                        "name": "latents",
                                        "type": "LATENT",
                                        "link": 262,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "image",
                                        "name": "image",
                                        "type": "IMAGE",
                                        "links": [468],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "LTXVSpatioTemporalTiledVAEDecode"
                                },
                                "widgets_values": [4, 8, 64, 8, False, "auto", "auto"],
                            },
                            {
                                "id": 92,
                                "type": "LTXVAudioVAEDecode",
                                "pos": [3516.4763680372953, 3001.214378538911],
                                "size": [240, 72],
                                "flags": {},
                                "order": 3,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "samples",
                                        "name": "samples",
                                        "type": "LATENT",
                                        "link": 263,
                                    },
                                    {
                                        "label": "Audio VAE",
                                        "localized_name": "audio_vae",
                                        "name": "audio_vae",
                                        "type": "VAE",
                                        "link": 370,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "Audio",
                                        "name": "Audio",
                                        "type": "AUDIO",
                                        "links": [469],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "LTXVAudioVAEDecode",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.7.0",
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                    "ue_properties": {
                                        "widget_ue_connectable": {},
                                        "version": "7.1",
                                        "input_ue_unconnectable": {},
                                    },
                                },
                                "widgets_values": [],
                            },
                            {
                                "id": 142,
                                "type": "GetNode",
                                "pos": [3519.9406352113315, 3113.9955785990123],
                                "size": [225, 104],
                                "flags": {},
                                "order": 2,
                                "mode": 0,
                                "inputs": [],
                                "outputs": [
                                    {"name": "FLOAT", "type": "FLOAT", "links": [470]}
                                ],
                                "title": "Get_FPS",
                                "properties": {
                                    "Node name for S&R": "GetNode",
                                    "aux_id": "kijai/ComfyUI-KJNodes",
                                },
                                "widgets_values": ["FPS"],
                                "color": "#232",
                                "bgcolor": "#353",
                            },
                        ],
                        "groups": [],
                        "links": [
                            {
                                "id": 262,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 117,
                                "target_slot": 1,
                                "type": "LATENT",
                            },
                            {
                                "id": 263,
                                "origin_id": -10,
                                "origin_slot": 1,
                                "target_id": 92,
                                "target_slot": 0,
                                "type": "LATENT",
                            },
                            {
                                "id": 369,
                                "origin_id": 159,
                                "origin_slot": 0,
                                "target_id": 117,
                                "target_slot": 0,
                                "type": "VAE",
                            },
                            {
                                "id": 370,
                                "origin_id": 160,
                                "origin_slot": 0,
                                "target_id": 92,
                                "target_slot": 1,
                                "type": "VAE",
                            },
                            {
                                "id": 468,
                                "origin_id": 117,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 0,
                                "type": "IMAGE",
                            },
                            {
                                "id": 469,
                                "origin_id": 92,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 1,
                                "type": "AUDIO",
                            },
                            {
                                "id": 470,
                                "origin_id": 142,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 2,
                                "type": "FLOAT",
                            },
                        ],
                        "extra": {},
                    },
                    {
                        "id": "68322638-2fa0-48e4-b108-07c6bf1d2390",
                        "version": 1,
                        "state": {
                            "lastGroupId": 0,
                            "lastNodeId": 255,
                            "lastLinkId": 584,
                            "lastRerouteId": 0,
                        },
                        "revision": 0,
                        "config": {},
                        "name": "Image Input",
                        "inputNode": {
                            "id": -10,
                            "bounding": [
                                -886.220085613943,
                                3509.235191619284,
                                155.11666870117188,
                                108,
                            ],
                        },
                        "outputNode": {
                            "id": -20,
                            "bounding": [
                                843.1677305538424,
                                3509.235191619284,
                                158.6500015258789,
                                128,
                            ],
                        },
                        "inputs": [
                            {
                                "id": "3372b170-c2f5-48b1-9bae-f06ea059c8f7",
                                "name": "value",
                                "type": "FLOAT",
                                "linkIds": [208],
                                "label": "duration (seconds)",
                                "pos": [-755.1034169127711, 3533.235191619284],
                            },
                            {
                                "id": "a28c234a-a05a-4554-9151-6ca7953227eb",
                                "name": "value_1",
                                "type": "INT",
                                "linkIds": [209],
                                "label": "fps",
                                "pos": [-755.1034169127711, 3553.235191619284],
                            },
                            {
                                "id": "fc3e05f0-78fc-49b1-b6c0-34fee739914a",
                                "name": "image",
                                "type": "IMAGE",
                                "linkIds": [386],
                                "label": "input_image",
                                "pos": [-755.1034169127711, 3573.235191619284],
                            },
                        ],
                        "outputs": [
                            {
                                "id": "42189281-4c40-43b5-a609-16ad6b0aa6cc",
                                "name": "LATENT",
                                "type": "LATENT",
                                "linkIds": [327],
                                "label": "video_latent",
                                "pos": [867.1677305538424, 3533.235191619284],
                            },
                            {
                                "id": "6ee8d8dd-4be2-465c-9d7f-4c1853afc750",
                                "name": "Latent",
                                "type": "LATENT",
                                "linkIds": [328],
                                "label": "audio_latent",
                                "pos": [867.1677305538424, 3553.235191619284],
                            },
                            {
                                "id": "0693e711-2ff4-4996-9481-bc739f835722",
                                "name": "FLOAT",
                                "type": "FLOAT",
                                "linkIds": [486],
                                "label": "fps",
                                "pos": [867.1677305538424, 3573.235191619284],
                            },
                            {
                                "id": "2e224f7a-b792-4957-a8d0-6e3b9d66928b",
                                "name": "output_image",
                                "type": "IMAGE",
                                "linkIds": [387],
                                "label": "compressed_image",
                                "pos": [867.1677305538424, 3593.235191619284],
                            },
                        ],
                        "widgets": [],
                        "nodes": [
                            {
                                "id": 91,
                                "type": "INTConstant",
                                "pos": [-698.220085613943, 3808.123063970049],
                                "size": [270.6166687011719, 104],
                                "flags": {},
                                "order": 6,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "value",
                                        "name": "value",
                                        "type": "INT",
                                        "widget": {"name": "value"},
                                        "link": 209,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "value",
                                        "name": "value",
                                        "type": "INT",
                                        "links": [432, 485],
                                    }
                                ],
                                "title": "FPS",
                                "properties": {"Node name for S&R": "INTConstant"},
                                "widgets_values": [25],
                                "color": "#1b4669",
                                "bgcolor": "#29699c",
                            },
                            {
                                "id": 95,
                                "type": "PrimitiveFloat",
                                "pos": [-697.2996493980697, 3657.581998585889],
                                "size": [270, 80],
                                "flags": {},
                                "order": 8,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "value",
                                        "name": "value",
                                        "type": "FLOAT",
                                        "widget": {"name": "value"},
                                        "link": 208,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "FLOAT",
                                        "name": "FLOAT",
                                        "type": "FLOAT",
                                        "links": [431],
                                    }
                                ],
                                "title": "Duration (seconds)",
                                "properties": {"Node name for S&R": "PrimitiveFloat"},
                                "widgets_values": [5],
                            },
                            {
                                "id": 76,
                                "type": "EmptyLTXVLatentVideo",
                                "pos": [454.93350763053036, 3395.231349501993],
                                "size": [270, 176],
                                "flags": {},
                                "order": 3,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "width",
                                        "name": "width",
                                        "type": "INT",
                                        "widget": {"name": "width"},
                                        "link": 439,
                                    },
                                    {
                                        "localized_name": "height",
                                        "name": "height",
                                        "type": "INT",
                                        "widget": {"name": "height"},
                                        "link": 442,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "LATENT",
                                        "name": "LATENT",
                                        "type": "LATENT",
                                        "links": [327],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "EmptyLTXVLatentVideo",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.3.60",
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                    "ue_properties": {
                                        "widget_ue_connectable": {},
                                        "version": "7.1",
                                        "input_ue_unconnectable": {},
                                    },
                                },
                                "widgets_values": [768, 512, 97, 1],
                            },
                            {
                                "id": 89,
                                "type": "LTXVEmptyLatentAudio",
                                "pos": [458.569871266894, 3620.6858949565394],
                                "size": [270, 144],
                                "flags": {},
                                "order": 5,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "audio_vae",
                                        "name": "audio_vae",
                                        "type": "VAE",
                                        "link": 367,
                                    },
                                    {
                                        "localized_name": "frames_number",
                                        "name": "frames_number",
                                        "type": "INT",
                                        "widget": {"name": "frames_number"},
                                        "link": 483,
                                    },
                                    {
                                        "localized_name": "frame_rate",
                                        "name": "frame_rate",
                                        "type": "INT",
                                        "widget": {"name": "frame_rate"},
                                        "link": 142,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "Latent",
                                        "name": "Latent",
                                        "type": "LATENT",
                                        "links": [328],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "LTXVEmptyLatentAudio",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.3.68",
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                    "ue_properties": {
                                        "widget_ue_connectable": {},
                                        "version": "7.1",
                                        "input_ue_unconnectable": {},
                                    },
                                },
                                "widgets_values": [97, 10, 1],
                            },
                            {
                                "id": 93,
                                "type": "ComfyNumberConvert",
                                "pos": [-229.9956837289258, 3929.827999201051],
                                "size": [339.6499938964844, 72],
                                "flags": {"collapsed": False},
                                "order": 7,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "value",
                                        "localized_name": "value",
                                        "name": "value",
                                        "type": "INT,FLOAT,STRING,BOOLEAN",
                                        "link": 485,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "FLOAT",
                                        "name": "FLOAT",
                                        "type": "FLOAT",
                                        "links": [486],
                                    },
                                    {
                                        "localized_name": "INT",
                                        "name": "INT",
                                        "type": "INT",
                                        "links": [142],
                                    },
                                ],
                                "properties": {
                                    "Node name for S&R": "ComfyNumberConvert"
                                },
                                "widgets_values": [],
                            },
                            {
                                "id": 87,
                                "type": "GetImageSizeAndCount",
                                "pos": [-698.2565607424958, 3316.600505076189],
                                "size": [323.9333190917969, 172],
                                "flags": {},
                                "order": 4,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "image",
                                        "name": "image",
                                        "type": "IMAGE",
                                        "link": 386,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "image",
                                        "name": "image",
                                        "type": "IMAGE",
                                        "links": [120],
                                    },
                                    {
                                        "label": "350 width",
                                        "localized_name": "width",
                                        "name": "width",
                                        "type": "INT",
                                        "links": [440],
                                    },
                                    {
                                        "label": "622 height",
                                        "localized_name": "height",
                                        "name": "height",
                                        "type": "INT",
                                        "links": [441],
                                    },
                                    {
                                        "label": "1 count",
                                        "localized_name": "count",
                                        "name": "count",
                                        "type": "INT",
                                        "links": None,
                                    },
                                ],
                                "properties": {
                                    "Node name for S&R": "GetImageSizeAndCount"
                                },
                                "widgets_values": [],
                            },
                            {
                                "id": 157,
                                "type": "GetNode",
                                "pos": [-981.9480726883353, 3364.977225375118],
                                "size": [225, 104],
                                "flags": {"collapsed": False},
                                "order": 0,
                                "mode": 0,
                                "inputs": [],
                                "outputs": [
                                    {"name": "VAE", "type": "VAE", "links": [367]}
                                ],
                                "title": "Get_AUDIO_VAE",
                                "properties": {
                                    "Node name for S&R": "GetNode",
                                    "aux_id": "kijai/ComfyUI-KJNodes",
                                },
                                "widgets_values": ["AUDIO_VAE"],
                                "color": "#322",
                                "bgcolor": "#533",
                            },
                            {
                                "id": 75,
                                "type": "LTXVPreprocess",
                                "pos": [457.1899449938919, 3831.831349501993],
                                "size": [271.5333251953125, 80.71666717529297],
                                "flags": {},
                                "order": 2,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "image",
                                        "name": "image",
                                        "type": "IMAGE",
                                        "link": 120,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "output_image",
                                        "name": "output_image",
                                        "type": "IMAGE",
                                        "links": [387],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "LTXVPreprocess",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.7.0",
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                    "ue_properties": {
                                        "widget_ue_connectable": {},
                                        "version": "7.1",
                                        "input_ue_unconnectable": {},
                                    },
                                },
                                "widgets_values": [18],
                            },
                            {
                                "id": 182,
                                "type": "MarkdownNote",
                                "pos": [-1154.044607687379, 3664.429977136585],
                                "size": [393.4166564941406, 198.8000030517578],
                                "flags": {"collapsed": False},
                                "order": 1,
                                "mode": 0,
                                "inputs": [],
                                "outputs": [],
                                "title": "About Size",
                                "properties": {},
                                "widgets_values": [
                                    "Important: "
                                    "Do "
                                    "not "
                                    "change "
                                    "the "
                                    "math "
                                    "inside "
                                    "`Frame "
                                    "Correction` "
                                    "or "
                                    "`Dimensity "
                                    "Validation`.\n"
                                    "If "
                                    "does "
                                    "you "
                                    "may "
                                    "receive "
                                    "the "
                                    "black "
                                    "result "
                                    "or "
                                    "blurry "
                                    "result."
                                ],
                                "color": "#222",
                                "bgcolor": "#000",
                            },
                            {
                                "id": 184,
                                "type": "eda6aac8-9c39-4624-87c9-724f2476e1df",
                                "pos": [-241.45867801873754, 3542.6599222304485],
                                "size": [340.6166687011719, 100],
                                "flags": {"collapsed": False},
                                "order": 9,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "duration (seconds)",
                                        "localized_name": "values.a",
                                        "name": "values.a",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 431,
                                    },
                                    {
                                        "label": "fps",
                                        "localized_name": "values.b",
                                        "name": "values.b",
                                        "shape": 7,
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 432,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "label": "frame_count (int)",
                                        "name": "INT_1",
                                        "type": "INT",
                                        "links": [483, 488],
                                    }
                                ],
                                "properties": {"previewExposures": []},
                            },
                            {
                                "id": 187,
                                "type": "14976831-5775-498d-8034-87d3fc4b3a03",
                                "pos": [-222.83531961956731, 3004.782117905328],
                                "size": [332.6833190917969, 76],
                                "flags": {},
                                "order": 10,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "dimensity",
                                        "localized_name": "values.a",
                                        "name": "values.a",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 440,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "label": "validated_value",
                                        "localized_name": "INT",
                                        "name": "INT",
                                        "type": "INT",
                                        "links": [439, 462],
                                    }
                                ],
                                "properties": {"previewExposures": []},
                            },
                            {
                                "id": 190,
                                "type": "04aa6e80-d31f-4d13-b38e-e362d9050769",
                                "pos": [-231.4272074025198, 3275.405785596231],
                                "size": [334.6499938964844, 76],
                                "flags": {},
                                "order": 11,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "dimensity",
                                        "localized_name": "values.a",
                                        "name": "values.a",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 441,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "label": "validated_value",
                                        "localized_name": "INT",
                                        "name": "INT",
                                        "type": "INT",
                                        "links": [442, 465],
                                    }
                                ],
                                "properties": {"previewExposures": []},
                            },
                            {
                                "id": 203,
                                "type": "PreviewAny",
                                "pos": [136.3739150487038, 3271.9124803935283],
                                "size": [225, 152],
                                "flags": {},
                                "order": 13,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "source",
                                        "name": "source",
                                        "type": "*",
                                        "link": 465,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "STRING",
                                        "name": "STRING",
                                        "type": "STRING",
                                        "links": None,
                                    }
                                ],
                                "title": "[preview] height",
                                "properties": {"Node name for S&R": "PreviewAny"},
                                "widgets_values": [None, None, None],
                            },
                            {
                                "id": 204,
                                "type": "PreviewAny",
                                "pos": [-121.23865754381188, 3694.7105490598055],
                                "size": [225, 152],
                                "flags": {},
                                "order": 14,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "source",
                                        "name": "source",
                                        "type": "*",
                                        "link": 488,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "STRING",
                                        "name": "STRING",
                                        "type": "STRING",
                                        "links": None,
                                    }
                                ],
                                "title": "[preview] frame_count",
                                "properties": {"Node name for S&R": "PreviewAny"},
                                "widgets_values": [None, None, None],
                            },
                            {
                                "id": 202,
                                "type": "PreviewAny",
                                "pos": [136.7384456600812, 3002.223270393423],
                                "size": [225, 152],
                                "flags": {},
                                "order": 12,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "source",
                                        "name": "source",
                                        "type": "*",
                                        "link": 462,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "STRING",
                                        "name": "STRING",
                                        "type": "STRING",
                                        "links": None,
                                    }
                                ],
                                "title": "[preview] width",
                                "properties": {"Node name for S&R": "PreviewAny"},
                                "widgets_values": [None, None, None],
                            },
                        ],
                        "groups": [],
                        "links": [
                            {
                                "id": 120,
                                "origin_id": 87,
                                "origin_slot": 0,
                                "target_id": 75,
                                "target_slot": 0,
                                "type": "IMAGE",
                            },
                            {
                                "id": 142,
                                "origin_id": 93,
                                "origin_slot": 1,
                                "target_id": 89,
                                "target_slot": 2,
                                "type": "INT",
                            },
                            {
                                "id": 208,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 95,
                                "target_slot": 0,
                                "type": "FLOAT",
                            },
                            {
                                "id": 209,
                                "origin_id": -10,
                                "origin_slot": 1,
                                "target_id": 91,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 327,
                                "origin_id": 76,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 0,
                                "type": "LATENT",
                            },
                            {
                                "id": 328,
                                "origin_id": 89,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 1,
                                "type": "LATENT",
                            },
                            {
                                "id": 367,
                                "origin_id": 157,
                                "origin_slot": 0,
                                "target_id": 89,
                                "target_slot": 0,
                                "type": "VAE",
                            },
                            {
                                "id": 386,
                                "origin_id": -10,
                                "origin_slot": 2,
                                "target_id": 87,
                                "target_slot": 0,
                                "type": "IMAGE",
                            },
                            {
                                "id": 387,
                                "origin_id": 75,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 3,
                                "type": "IMAGE",
                            },
                            {
                                "id": 431,
                                "origin_id": 95,
                                "origin_slot": 0,
                                "target_id": 184,
                                "target_slot": 0,
                                "type": "FLOAT",
                            },
                            {
                                "id": 432,
                                "origin_id": 91,
                                "origin_slot": 0,
                                "target_id": 184,
                                "target_slot": 1,
                                "type": "INT",
                            },
                            {
                                "id": 439,
                                "origin_id": 187,
                                "origin_slot": 0,
                                "target_id": 76,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 440,
                                "origin_id": 87,
                                "origin_slot": 1,
                                "target_id": 187,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 441,
                                "origin_id": 87,
                                "origin_slot": 2,
                                "target_id": 190,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 442,
                                "origin_id": 190,
                                "origin_slot": 0,
                                "target_id": 76,
                                "target_slot": 1,
                                "type": "INT",
                            },
                            {
                                "id": 462,
                                "origin_id": 187,
                                "origin_slot": 0,
                                "target_id": 202,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 465,
                                "origin_id": 190,
                                "origin_slot": 0,
                                "target_id": 203,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 483,
                                "origin_id": 184,
                                "origin_slot": 0,
                                "target_id": 89,
                                "target_slot": 1,
                                "type": "INT",
                            },
                            {
                                "id": 485,
                                "origin_id": 91,
                                "origin_slot": 0,
                                "target_id": 93,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 486,
                                "origin_id": 93,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 2,
                                "type": "FLOAT",
                            },
                            {
                                "id": 488,
                                "origin_id": 184,
                                "origin_slot": 0,
                                "target_id": 204,
                                "target_slot": 0,
                                "type": "INT",
                            },
                        ],
                        "extra": {},
                    },
                    {
                        "id": "30f35274-1bfd-4ef1-8a1f-83077a47ab02",
                        "version": 1,
                        "state": {
                            "lastGroupId": 0,
                            "lastNodeId": 255,
                            "lastLinkId": 584,
                            "lastRerouteId": 0,
                        },
                        "revision": 0,
                        "config": {},
                        "name": "[Kinjia] Settings",
                        "inputNode": {
                            "id": -10,
                            "bounding": [
                                -1196.553988675269,
                                2594.431792964143,
                                132.41666412353516,
                                148,
                            ],
                        },
                        "outputNode": {
                            "id": -20,
                            "bounding": [
                                278.4458163063754,
                                2544.431792964143,
                                128,
                                128,
                            ],
                        },
                        "inputs": [
                            {
                                "id": "9fe7a16c-bd74-43af-85b6-f895aa7dfe3f",
                                "name": "unet_name",
                                "type": "COMBO",
                                "linkIds": [555],
                                "label": "distilled_model",
                                "pos": [-1088.1373245517339, 2618.431792964143],
                            },
                            {
                                "id": "079316ce-afc8-4fd7-ab55-64f57e7790de",
                                "name": "clip_name1",
                                "type": "COMBO",
                                "linkIds": [556],
                                "label": "gemma3_clip",
                                "pos": [-1088.1373245517339, 2638.431792964143],
                            },
                            {
                                "id": "875c9a8b-664d-48ea-bd17-601fb4241a42",
                                "name": "clip_name2",
                                "type": "COMBO",
                                "linkIds": [557],
                                "label": "ltx_encoder",
                                "pos": [-1088.1373245517339, 2658.431792964143],
                            },
                            {
                                "id": "558bbfb1-ac54-4dbb-b344-f5fe02ca0e01",
                                "name": "vae_name",
                                "type": "COMBO",
                                "linkIds": [503],
                                "label": "video_vae",
                                "pos": [-1088.1373245517339, 2678.431792964143],
                            },
                            {
                                "id": "45638f63-f04a-41f1-a3e0-d7ab864b4bfd",
                                "name": "vae_name_1",
                                "type": "COMBO",
                                "linkIds": [504],
                                "label": "audio_vae",
                                "pos": [-1088.1373245517339, 2698.431792964143],
                            },
                        ],
                        "outputs": [
                            {
                                "id": "4a7032b4-bf20-4d8b-85ab-82f17a2612e3",
                                "name": "MODEL_1",
                                "type": "MODEL",
                                "linkIds": [559],
                                "label": "model",
                                "pos": [302.4458163063754, 2568.431792964143],
                            },
                            {
                                "id": "46470964-1e78-43c4-a60c-7b1ec002706d",
                                "name": "CLIP",
                                "type": "CLIP",
                                "linkIds": [558],
                                "label": "clip",
                                "pos": [302.4458163063754, 2588.431792964143],
                            },
                            {
                                "id": "c0f1c5eb-5b4d-49cb-8447-3cadf62c30bb",
                                "name": "VAE",
                                "type": "VAE",
                                "linkIds": [506],
                                "label": "video_vae",
                                "pos": [302.4458163063754, 2608.431792964143],
                            },
                            {
                                "id": "57fbcad7-32bf-41f7-ba64-4fc82e1f83b4",
                                "name": "VAE_1",
                                "type": "VAE",
                                "linkIds": [505],
                                "label": "audio_vae",
                                "pos": [302.4458163063754, 2628.431792964143],
                            },
                        ],
                        "widgets": [],
                        "nodes": [
                            {
                                "id": 218,
                                "type": "VAELoaderKJ",
                                "pos": [-1009.3227664425578, 2657.146963183867],
                                "size": [466.75, 168],
                                "flags": {},
                                "order": 0,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "vae_name",
                                        "name": "vae_name",
                                        "type": "COMBO",
                                        "widget": {"name": "vae_name"},
                                        "link": 503,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "VAE",
                                        "name": "VAE",
                                        "type": "VAE",
                                        "links": [506],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "VAELoaderKJ",
                                    "cnr_id": "comfyui-kjnodes",
                                    "ver": "1.3.1",
                                    "ue_properties": {
                                        "widget_ue_connectable": {},
                                        "input_ue_unconnectable": {},
                                        "version": "7.1",
                                    },
                                },
                                "widgets_values": [
                                    "LTX23_video_vae_bf16.safetensors",
                                    "main_device",
                                    "bf16",
                                ],
                                "color": "#322",
                                "bgcolor": "#533",
                            },
                            {
                                "id": 219,
                                "type": "VAELoaderKJ",
                                "pos": [-1009.0789797281884, 2876.4496014489846],
                                "size": [466.75, 168],
                                "flags": {},
                                "order": 1,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "vae_name",
                                        "name": "vae_name",
                                        "type": "COMBO",
                                        "widget": {"name": "vae_name"},
                                        "link": 504,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "VAE",
                                        "name": "VAE",
                                        "type": "VAE",
                                        "links": [505],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "VAELoaderKJ",
                                    "cnr_id": "comfyui-kjnodes",
                                    "ver": "1.3.1",
                                    "ue_properties": {
                                        "widget_ue_connectable": {},
                                        "input_ue_unconnectable": {},
                                        "version": "7.1",
                                    },
                                },
                                "widgets_values": [
                                    "LTX23_audio_vae_bf16.safetensors",
                                    "main_device",
                                    "bf16",
                                ],
                                "color": "#322",
                                "bgcolor": "#533",
                            },
                            {
                                "id": 239,
                                "type": "DualCLIPLoaderGGUF",
                                "pos": [-1006.6116295258585, 2416.3843850276444],
                                "size": [569.1666870117188, 196],
                                "flags": {},
                                "order": 2,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "clip_name1",
                                        "name": "clip_name1",
                                        "type": "COMBO",
                                        "widget": {"name": "clip_name1"},
                                        "link": 556,
                                    },
                                    {
                                        "localized_name": "clip_name2",
                                        "name": "clip_name2",
                                        "type": "COMBO",
                                        "widget": {"name": "clip_name2"},
                                        "link": 557,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "CLIP",
                                        "name": "CLIP",
                                        "type": "CLIP",
                                        "links": [558],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "DualCLIPLoaderGGUF"
                                },
                                "widgets_values": [
                                    "gemma-3-12b-it-qat-UD-Q2_K_XL.gguf",
                                    "ltx-2.3-22b-dev_embeddings_connectors.safetensors",
                                    "ltxv",
                                ],
                            },
                            {
                                "id": 240,
                                "type": "UnetLoaderGGUF",
                                "pos": [-1008.5995022281645, 2261.607063974776],
                                "size": [575.2666625976562, 104],
                                "flags": {},
                                "order": 3,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "unet_name",
                                        "name": "unet_name",
                                        "type": "COMBO",
                                        "widget": {"name": "unet_name"},
                                        "link": 555,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "MODEL",
                                        "name": "MODEL",
                                        "type": "MODEL",
                                        "links": [559],
                                    }
                                ],
                                "properties": {"Node name for S&R": "UnetLoaderGGUF"},
                                "widgets_values": [
                                    "Unknown/ltx-2.3-22b-dev-UD-Q2_K.gguf"
                                ],
                            },
                        ],
                        "groups": [],
                        "links": [
                            {
                                "id": 503,
                                "origin_id": -10,
                                "origin_slot": 3,
                                "target_id": 218,
                                "target_slot": 0,
                                "type": "COMBO",
                            },
                            {
                                "id": 504,
                                "origin_id": -10,
                                "origin_slot": 4,
                                "target_id": 219,
                                "target_slot": 0,
                                "type": "COMBO",
                            },
                            {
                                "id": 505,
                                "origin_id": 219,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 3,
                                "type": "VAE",
                            },
                            {
                                "id": 506,
                                "origin_id": 218,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 2,
                                "type": "VAE",
                            },
                            {
                                "id": 555,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 240,
                                "target_slot": 0,
                                "type": "COMBO",
                            },
                            {
                                "id": 556,
                                "origin_id": -10,
                                "origin_slot": 1,
                                "target_id": 239,
                                "target_slot": 0,
                                "type": "COMBO",
                            },
                            {
                                "id": 557,
                                "origin_id": -10,
                                "origin_slot": 2,
                                "target_id": 239,
                                "target_slot": 1,
                                "type": "COMBO",
                            },
                            {
                                "id": 558,
                                "origin_id": 239,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 1,
                                "type": "CLIP",
                            },
                            {
                                "id": 559,
                                "origin_id": 240,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 0,
                                "type": "MODEL",
                            },
                        ],
                        "extra": {},
                    },
                    {
                        "id": "eda6aac8-9c39-4624-87c9-724f2476e1df",
                        "version": 1,
                        "state": {
                            "lastGroupId": 0,
                            "lastNodeId": 255,
                            "lastLinkId": 584,
                            "lastRerouteId": 0,
                        },
                        "revision": 0,
                        "config": {},
                        "name": "Frame Calculation",
                        "inputNode": {
                            "id": -10,
                            "bounding": [
                                -924.5464317491604,
                                4380.421533947579,
                                155.11666870117188,
                                88,
                            ],
                        },
                        "outputNode": {
                            "id": -20,
                            "bounding": [
                                -62.88650744207882,
                                4390.421533947579,
                                144.9000015258789,
                                68,
                            ],
                        },
                        "inputs": [
                            {
                                "id": "c6c3d8b6-43d9-42b3-b3a4-bf1185a3dd24",
                                "name": "values.a",
                                "type": "FLOAT,INT,BOOLEAN",
                                "linkIds": [149],
                                "localized_name": "values.a",
                                "label": "duration (seconds)",
                                "pos": [-793.4297630479886, 4404.421533947579],
                            },
                            {
                                "id": "293760fb-3226-4b45-82ae-cadcf1f42af6",
                                "name": "values.b",
                                "type": "FLOAT,INT,BOOLEAN",
                                "linkIds": [150],
                                "localized_name": "values.b",
                                "label": "fps",
                                "shape": 7,
                                "pos": [-793.4297630479886, 4424.421533947579],
                            },
                        ],
                        "outputs": [
                            {
                                "id": "67860450-0b27-4c39-8fbb-3efc0f6d23f8",
                                "name": "INT_1",
                                "type": "INT",
                                "linkIds": [487],
                                "label": "frame_count (int)",
                                "pos": [-38.88650744207882, 4414.421533947579],
                            }
                        ],
                        "widgets": [],
                        "nodes": [
                            {
                                "id": 96,
                                "type": "ComfyMathExpression",
                                "pos": [-736.5464317491604, 4339.185637909268],
                                "size": [225, 164],
                                "flags": {"collapsed": False},
                                "order": 1,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "a",
                                        "localized_name": "values.a",
                                        "name": "values.a",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 149,
                                    },
                                    {
                                        "label": "b",
                                        "localized_name": "values.b",
                                        "name": "values.b",
                                        "shape": 7,
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 150,
                                    },
                                    {
                                        "label": "c",
                                        "localized_name": "values.c",
                                        "name": "values.c",
                                        "shape": 7,
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": None,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "FLOAT",
                                        "name": "FLOAT",
                                        "type": "FLOAT",
                                        "links": None,
                                    },
                                    {
                                        "localized_name": "INT",
                                        "name": "INT",
                                        "type": "INT",
                                        "links": [460],
                                    },
                                    {
                                        "localized_name": "BOOL",
                                        "name": "BOOL",
                                        "type": "BOOLEAN",
                                        "links": None,
                                    },
                                ],
                                "title": "frames",
                                "properties": {
                                    "Node name for S&R": "ComfyMathExpression"
                                },
                                "widgets_values": ["a * b"],
                            },
                            {
                                "id": 183,
                                "type": "ComfyMathExpression",
                                "pos": [-735.320489809611, 4811.827517125361],
                                "size": [299.0833435058594, 200],
                                "flags": {},
                                "order": 2,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "a",
                                        "localized_name": "values.a",
                                        "name": "values.a",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 461,
                                    },
                                    {
                                        "label": "b",
                                        "localized_name": "values.b",
                                        "name": "values.b",
                                        "shape": 7,
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": None,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "FLOAT",
                                        "name": "FLOAT",
                                        "type": "FLOAT",
                                        "links": None,
                                    },
                                    {
                                        "localized_name": "INT",
                                        "name": "INT",
                                        "type": "INT",
                                        "links": [446, 450],
                                    },
                                    {
                                        "localized_name": "BOOL",
                                        "name": "BOOL",
                                        "type": "BOOLEAN",
                                        "links": None,
                                    },
                                ],
                                "title": "correction",
                                "properties": {
                                    "Node name for S&R": "ComfyMathExpression"
                                },
                                "widgets_values": ["a * 8 + 1"],
                            },
                            {
                                "id": 193,
                                "type": "ComfyMathExpression",
                                "pos": [-734.5939190615316, 5162.701707348348],
                                "size": [400, 200],
                                "flags": {},
                                "order": 3,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "a",
                                        "localized_name": "values.a",
                                        "name": "values.a",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 446,
                                    },
                                    {
                                        "label": "b",
                                        "localized_name": "values.b",
                                        "name": "values.b",
                                        "shape": 7,
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 449,
                                    },
                                    {
                                        "label": "c",
                                        "localized_name": "values.c",
                                        "name": "values.c",
                                        "shape": 7,
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": None,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "FLOAT",
                                        "name": "FLOAT",
                                        "type": "FLOAT",
                                        "links": None,
                                    },
                                    {
                                        "localized_name": "INT",
                                        "name": "INT",
                                        "type": "INT",
                                        "links": None,
                                    },
                                    {
                                        "localized_name": "BOOL",
                                        "name": "BOOL",
                                        "type": "BOOLEAN",
                                        "links": [447],
                                    },
                                ],
                                "properties": {
                                    "Node name for S&R": "ComfyMathExpression"
                                },
                                "widgets_values": ["a < b"],
                            },
                            {
                                "id": 194,
                                "type": "ComfySwitchNode",
                                "pos": [-133.41913384410293, 5078.477882895587],
                                "size": [270, 124],
                                "flags": {},
                                "order": 4,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "on_false",
                                        "name": "on_false",
                                        "type": "INT",
                                        "link": 450,
                                    },
                                    {
                                        "localized_name": "on_true",
                                        "name": "on_true",
                                        "type": "INT",
                                        "link": 448,
                                    },
                                    {
                                        "localized_name": "switch",
                                        "name": "switch",
                                        "type": "BOOLEAN",
                                        "widget": {"name": "switch"},
                                        "link": 447,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "output",
                                        "name": "output",
                                        "type": "INT",
                                        "links": [487],
                                    }
                                ],
                                "properties": {"Node name for S&R": "ComfySwitchNode"},
                                "widgets_values": [False],
                            },
                            {
                                "id": 196,
                                "type": "PrimitiveInt",
                                "pos": [-1129.2617576242612, 5030.362625453301],
                                "size": [270, 82],
                                "flags": {},
                                "order": 0,
                                "mode": 0,
                                "inputs": [],
                                "outputs": [
                                    {
                                        "localized_name": "INT",
                                        "name": "INT",
                                        "type": "INT",
                                        "links": [448, 449],
                                    }
                                ],
                                "properties": {"Node name for S&R": "PrimitiveInt"},
                                "widgets_values": [9, "fixed"],
                            },
                            {
                                "id": 201,
                                "type": "ComfyMathExpression",
                                "pos": [-735.5348953595237, 4558.077612293833],
                                "size": [299.0833435058594, 200],
                                "flags": {},
                                "order": 5,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "a",
                                        "localized_name": "values.a",
                                        "name": "values.a",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 460,
                                    },
                                    {
                                        "label": "b",
                                        "localized_name": "values.b",
                                        "name": "values.b",
                                        "shape": 7,
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": None,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "FLOAT",
                                        "name": "FLOAT",
                                        "type": "FLOAT",
                                        "links": None,
                                    },
                                    {
                                        "localized_name": "INT",
                                        "name": "INT",
                                        "type": "INT",
                                        "links": [461],
                                    },
                                    {
                                        "localized_name": "BOOL",
                                        "name": "BOOL",
                                        "type": "BOOLEAN",
                                        "links": None,
                                    },
                                ],
                                "title": "correction",
                                "properties": {
                                    "Node name for S&R": "ComfyMathExpression"
                                },
                                "widgets_values": ["a / 8 + 1"],
                            },
                        ],
                        "groups": [],
                        "links": [
                            {
                                "id": 149,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 96,
                                "target_slot": 0,
                                "type": "FLOAT",
                            },
                            {
                                "id": 150,
                                "origin_id": -10,
                                "origin_slot": 1,
                                "target_id": 96,
                                "target_slot": 1,
                                "type": "INT",
                            },
                            {
                                "id": 446,
                                "origin_id": 183,
                                "origin_slot": 1,
                                "target_id": 193,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 447,
                                "origin_id": 193,
                                "origin_slot": 2,
                                "target_id": 194,
                                "target_slot": 2,
                                "type": "BOOLEAN",
                            },
                            {
                                "id": 448,
                                "origin_id": 196,
                                "origin_slot": 0,
                                "target_id": 194,
                                "target_slot": 1,
                                "type": "INT",
                            },
                            {
                                "id": 449,
                                "origin_id": 196,
                                "origin_slot": 0,
                                "target_id": 193,
                                "target_slot": 1,
                                "type": "INT",
                            },
                            {
                                "id": 450,
                                "origin_id": 183,
                                "origin_slot": 1,
                                "target_id": 194,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 460,
                                "origin_id": 96,
                                "origin_slot": 1,
                                "target_id": 201,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 461,
                                "origin_id": 201,
                                "origin_slot": 1,
                                "target_id": 183,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 487,
                                "origin_id": 194,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 0,
                                "type": "INT",
                            },
                        ],
                        "extra": {},
                    },
                    {
                        "id": "14976831-5775-498d-8034-87d3fc4b3a03",
                        "version": 1,
                        "state": {
                            "lastGroupId": 0,
                            "lastNodeId": 255,
                            "lastLinkId": 584,
                            "lastRerouteId": 0,
                        },
                        "revision": 0,
                        "config": {},
                        "name": "Dimensity Validation",
                        "inputNode": {
                            "id": -10,
                            "bounding": [
                                34.533314806798046,
                                3185.691001983322,
                                128,
                                68,
                            ],
                        },
                        "outputNode": {
                            "id": -20,
                            "bounding": [
                                933.438531359792,
                                3185.691001983322,
                                134.4000015258789,
                                68,
                            ],
                        },
                        "inputs": [
                            {
                                "id": "2de8ba3e-fb3f-4eb1-8ff1-c1c12ac56da2",
                                "name": "values.a",
                                "type": "FLOAT,INT,BOOLEAN",
                                "linkIds": [435],
                                "localized_name": "values.a",
                                "label": "dimensity",
                                "pos": [138.53331480679805, 3209.691001983322],
                            }
                        ],
                        "outputs": [
                            {
                                "id": "c52f005c-2824-4a94-8319-fd4d8fba3f9d",
                                "name": "INT",
                                "type": "INT",
                                "linkIds": [437],
                                "localized_name": "INT",
                                "label": "validated_value",
                                "pos": [957.438531359792, 3209.691001983322],
                            }
                        ],
                        "widgets": [],
                        "nodes": [
                            {
                                "id": 185,
                                "type": "ComfyMathExpression",
                                "pos": [222.53331480679805, 3134.0672485306923],
                                "size": [225, 200],
                                "flags": {},
                                "order": 0,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "a",
                                        "localized_name": "values.a",
                                        "name": "values.a",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 435,
                                    },
                                    {
                                        "label": "b",
                                        "localized_name": "values.b",
                                        "name": "values.b",
                                        "shape": 7,
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": None,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "FLOAT",
                                        "name": "FLOAT",
                                        "type": "FLOAT",
                                        "links": None,
                                    },
                                    {
                                        "localized_name": "INT",
                                        "name": "INT",
                                        "type": "INT",
                                        "links": [436],
                                    },
                                    {
                                        "localized_name": "BOOL",
                                        "name": "BOOL",
                                        "type": "BOOLEAN",
                                        "links": None,
                                    },
                                ],
                                "properties": {
                                    "Node name for S&R": "ComfyMathExpression"
                                },
                                "widgets_values": ["a * 0.015625"],
                            },
                            {
                                "id": 186,
                                "type": "ComfyMathExpression",
                                "pos": [473.438531359792, 3135.314755435952],
                                "size": [400, 200],
                                "flags": {},
                                "order": 1,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "a",
                                        "localized_name": "values.a",
                                        "name": "values.a",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 436,
                                    },
                                    {
                                        "label": "b",
                                        "localized_name": "values.b",
                                        "name": "values.b",
                                        "shape": 7,
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": None,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "FLOAT",
                                        "name": "FLOAT",
                                        "type": "FLOAT",
                                        "links": None,
                                    },
                                    {
                                        "localized_name": "INT",
                                        "name": "INT",
                                        "type": "INT",
                                        "links": [437],
                                    },
                                    {
                                        "localized_name": "BOOL",
                                        "name": "BOOL",
                                        "type": "BOOLEAN",
                                        "links": None,
                                    },
                                ],
                                "properties": {
                                    "Node name for S&R": "ComfyMathExpression"
                                },
                                "widgets_values": ["a * 64"],
                            },
                        ],
                        "groups": [],
                        "links": [
                            {
                                "id": 436,
                                "origin_id": 185,
                                "origin_slot": 1,
                                "target_id": 186,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 435,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 185,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 437,
                                "origin_id": 186,
                                "origin_slot": 1,
                                "target_id": -20,
                                "target_slot": 0,
                                "type": "INT",
                            },
                        ],
                        "extra": {},
                    },
                    {
                        "id": "04aa6e80-d31f-4d13-b38e-e362d9050769",
                        "version": 1,
                        "state": {
                            "lastGroupId": 0,
                            "lastNodeId": 255,
                            "lastLinkId": 584,
                            "lastRerouteId": 0,
                        },
                        "revision": 0,
                        "config": {},
                        "name": "Dimensity Validation",
                        "inputNode": {
                            "id": -10,
                            "bounding": [
                                34.533314806798046,
                                3185.691001983322,
                                128,
                                68,
                            ],
                        },
                        "outputNode": {
                            "id": -20,
                            "bounding": [
                                933.438531359792,
                                3185.691001983322,
                                134.4000015258789,
                                68,
                            ],
                        },
                        "inputs": [
                            {
                                "id": "2de8ba3e-fb3f-4eb1-8ff1-c1c12ac56da2",
                                "name": "values.a",
                                "type": "FLOAT,INT,BOOLEAN",
                                "linkIds": [435],
                                "localized_name": "values.a",
                                "label": "dimensity",
                                "pos": [138.53331480679805, 3209.691001983322],
                            }
                        ],
                        "outputs": [
                            {
                                "id": "c52f005c-2824-4a94-8319-fd4d8fba3f9d",
                                "name": "INT",
                                "type": "INT",
                                "linkIds": [458],
                                "localized_name": "INT",
                                "label": "validated_value",
                                "pos": [957.438531359792, 3209.691001983322],
                            }
                        ],
                        "widgets": [],
                        "nodes": [
                            {
                                "id": 188,
                                "type": "ComfyMathExpression",
                                "pos": [222.53331480679805, 3134.0672485306923],
                                "size": [225, 200],
                                "flags": {},
                                "order": 1,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "a",
                                        "localized_name": "values.a",
                                        "name": "values.a",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 435,
                                    },
                                    {
                                        "label": "b",
                                        "localized_name": "values.b",
                                        "name": "values.b",
                                        "shape": 7,
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": None,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "FLOAT",
                                        "name": "FLOAT",
                                        "type": "FLOAT",
                                        "links": None,
                                    },
                                    {
                                        "localized_name": "INT",
                                        "name": "INT",
                                        "type": "INT",
                                        "links": [436],
                                    },
                                    {
                                        "localized_name": "BOOL",
                                        "name": "BOOL",
                                        "type": "BOOLEAN",
                                        "links": None,
                                    },
                                ],
                                "properties": {
                                    "Node name for S&R": "ComfyMathExpression"
                                },
                                "widgets_values": ["a * 0.015625"],
                            },
                            {
                                "id": 189,
                                "type": "ComfyMathExpression",
                                "pos": [222.82266112104568, 3388.9320133422284],
                                "size": [400, 200],
                                "flags": {},
                                "order": 2,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "a",
                                        "localized_name": "values.a",
                                        "name": "values.a",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 436,
                                    },
                                    {
                                        "label": "b",
                                        "localized_name": "values.b",
                                        "name": "values.b",
                                        "shape": 7,
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": None,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "FLOAT",
                                        "name": "FLOAT",
                                        "type": "FLOAT",
                                        "links": None,
                                    },
                                    {
                                        "localized_name": "INT",
                                        "name": "INT",
                                        "type": "INT",
                                        "links": [452, 457],
                                    },
                                    {
                                        "localized_name": "BOOL",
                                        "name": "BOOL",
                                        "type": "BOOLEAN",
                                        "links": None,
                                    },
                                ],
                                "properties": {
                                    "Node name for S&R": "ComfyMathExpression"
                                },
                                "widgets_values": ["a * 64"],
                            },
                            {
                                "id": 197,
                                "type": "SimpleCalculatorKJ",
                                "pos": [223.24788886602937, 3645.6106582931025],
                                "size": [400, 200],
                                "flags": {},
                                "order": 3,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "a",
                                        "localized_name": "variables.a",
                                        "name": "variables.a",
                                        "shape": 7,
                                        "type": "INT,FLOAT,BOOLEAN",
                                        "link": 452,
                                    },
                                    {
                                        "label": "b",
                                        "localized_name": "variables.b",
                                        "name": "variables.b",
                                        "shape": 7,
                                        "type": "INT,FLOAT,BOOLEAN",
                                        "link": 453,
                                    },
                                    {
                                        "label": "c",
                                        "localized_name": "variables.c",
                                        "name": "variables.c",
                                        "shape": 7,
                                        "type": "INT,FLOAT,BOOLEAN",
                                        "link": None,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "FLOAT",
                                        "name": "FLOAT",
                                        "type": "FLOAT",
                                        "links": None,
                                    },
                                    {
                                        "localized_name": "INT",
                                        "name": "INT",
                                        "type": "INT",
                                        "links": [],
                                    },
                                    {
                                        "localized_name": "BOOLEAN",
                                        "name": "BOOLEAN",
                                        "type": "BOOLEAN",
                                        "links": [455],
                                    },
                                ],
                                "properties": {
                                    "Node name for S&R": "SimpleCalculatorKJ"
                                },
                                "widgets_values": ["a < b"],
                            },
                            {
                                "id": 198,
                                "type": "PrimitiveInt",
                                "pos": [-119.49062576790703, 3667.8959617245123],
                                "size": [270, 82],
                                "flags": {},
                                "order": 0,
                                "mode": 0,
                                "inputs": [],
                                "outputs": [
                                    {
                                        "localized_name": "INT",
                                        "name": "INT",
                                        "type": "INT",
                                        "links": [453, 456],
                                    }
                                ],
                                "properties": {"Node name for S&R": "PrimitiveInt"},
                                "widgets_values": [64, "fixed"],
                            },
                            {
                                "id": 199,
                                "type": "ComfySwitchNode",
                                "pos": [705.6813538182663, 3566.6388942685676],
                                "size": [270, 124],
                                "flags": {},
                                "order": 4,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "on_false",
                                        "name": "on_false",
                                        "type": "INT",
                                        "link": 457,
                                    },
                                    {
                                        "localized_name": "on_true",
                                        "name": "on_true",
                                        "type": "INT",
                                        "link": 456,
                                    },
                                    {
                                        "localized_name": "switch",
                                        "name": "switch",
                                        "type": "BOOLEAN",
                                        "widget": {"name": "switch"},
                                        "link": 455,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "output",
                                        "name": "output",
                                        "type": "INT",
                                        "links": [458],
                                    }
                                ],
                                "properties": {"Node name for S&R": "ComfySwitchNode"},
                                "widgets_values": [False],
                            },
                        ],
                        "groups": [],
                        "links": [
                            {
                                "id": 436,
                                "origin_id": 188,
                                "origin_slot": 1,
                                "target_id": 189,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 435,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 188,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 452,
                                "origin_id": 189,
                                "origin_slot": 1,
                                "target_id": 197,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 453,
                                "origin_id": 198,
                                "origin_slot": 0,
                                "target_id": 197,
                                "target_slot": 1,
                                "type": "INT",
                            },
                            {
                                "id": 455,
                                "origin_id": 197,
                                "origin_slot": 2,
                                "target_id": 199,
                                "target_slot": 2,
                                "type": "BOOLEAN",
                            },
                            {
                                "id": 456,
                                "origin_id": 198,
                                "origin_slot": 0,
                                "target_id": 199,
                                "target_slot": 1,
                                "type": "INT",
                            },
                            {
                                "id": 457,
                                "origin_id": 189,
                                "origin_slot": 1,
                                "target_id": 199,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 458,
                                "origin_id": 199,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 0,
                                "type": "INT",
                            },
                        ],
                        "extra": {},
                    },
                ]
            },
            "config": {},
            "extra": {
                "ds": {
                    "scale": 0.8672275245390825,
                    "offset": [-928.1734645477719, -441.6637915441105],
                },
                "frontendVersion": "1.46.10",
                "VHS_latentpreview": False,
                "VHS_latentpreviewrate": 0,
                "VHS_MetadataImage": True,
                "VHS_KeepIntermediate": True,
            },
            "version": 0.4,
        }
    }


workflow = build_workflow()
prompt = json.loads(json.dumps(workflow))
extra_pnginfo = build_extra_pnginfo()


# Workflow execution
def generate(
    prompt_text: str,
    image_path: str,
    duration: float,
    fps: int,
    unload_models: bool | None = None,
):
    bootstrap_comfyui_runtime()
    add_extra_model_paths()
    import_custom_nodes()

    # Node imports
    from nodes import (
        CLIPTextEncode,
        ConditioningZeroOut,
        LoadImage,
        NODE_CLASS_MAPPINGS,
    )

    import torch

    try:
        with torch.inference_mode():
            loadimage = LoadImage()
            loadimage_246 = loadimage.load_image(image=image_path)
            vaeloaderkj = NODE_CLASS_MAPPINGS["VAELoaderKJ"]()
            vaeloaderkj_254_218 = vaeloaderkj.load_vae(
                vae_name="LTX23_video_vae_bf16.safetensors",
                device="main_device",
                weight_dtype="bf16",
            )
            vaeloaderkj_254_219 = vaeloaderkj.load_vae(
                vae_name="LTX23_audio_vae_bf16.safetensors",
                device="main_device",
                weight_dtype="bf16",
            )
            dualcliploadergguf = NODE_CLASS_MAPPINGS["DualCLIPLoaderGGUF"]()
            dualcliploadergguf_254_239 = dualcliploadergguf.load_clip(
                clip_name1="gemma-3-12b-it-qat-UD-Q2_K_XL.gguf",
                clip_name2="ltx-2.3_text_projection_bf16.safetensors",
                type="ltxv",
            )
            unetloadergguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
            unetloadergguf_254_240 = unetloadergguf.load_unet(
                unet_name="Unknown/ltx-2.3-22b-distilled-1.1-UD-Q2_K.gguf"
            )
            intconstant = NODE_CLASS_MAPPINGS["INTConstant"]()
            intconstant_252_91 = intconstant.get_value(value=fps)
            primitivefloat = NODE_CLASS_MAPPINGS["PrimitiveFloat"]()
            primitivefloat_252_95 = primitivefloat.EXECUTE_NORMALIZED(value=duration)
            primitiveint = NODE_CLASS_MAPPINGS["PrimitiveInt"]()
            primitiveint_252_184_196 = primitiveint.EXECUTE_NORMALIZED(value=9)
            primitiveint_252_190_198 = primitiveint.EXECUTE_NORMALIZED(value=64)
            cliptextencode = CLIPTextEncode()
            cliptextencode_247_86 = cliptextencode.encode(
                text=prompt_text, clip=get_value_at_index(dualcliploadergguf_254_239, 0)
            )
            randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
            node_248_85_noise_seed = prompt["248:85"]["inputs"]["noise_seed"] = (
                random.randint(1, 2**64)
            )
            randomnoise_248_85 = randomnoise.EXECUTE_NORMALIZED(
                noise_seed=node_248_85_noise_seed
            )
            ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
            ksamplerselect_248_81 = ksamplerselect.EXECUTE_NORMALIZED(
                sampler_name="euler_ancestral"
            )
            comfynumberconvert = NODE_CLASS_MAPPINGS["ComfyNumberConvert"]()
            conditioningzeroout = ConditioningZeroOut()
            ltxvconditioning = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
            cfgguider = NODE_CLASS_MAPPINGS["CFGGuider"]()
            getimagesizeandcount = NODE_CLASS_MAPPINGS["GetImageSizeAndCount"]()
            ltxvpreprocess = NODE_CLASS_MAPPINGS["LTXVPreprocess"]()
            comfyswitchnode = NODE_CLASS_MAPPINGS["ComfySwitchNode"]()
            emptyltxvlatentvideo = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()
            ltxvimgtovideoinplace = NODE_CLASS_MAPPINGS["LTXVImgToVideoInplace"]()
            ltxvemptylatentaudio = NODE_CLASS_MAPPINGS["LTXVEmptyLatentAudio"]()
            ltxvconcatavlatent = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
            ltxvscheduler = NODE_CLASS_MAPPINGS["LTXVScheduler"]()
            samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
            ltxvseparateavlatent = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
            ltxvspatiotemporaltiledvaedecode = NODE_CLASS_MAPPINGS[
                "LTXVSpatioTemporalTiledVAEDecode"
            ]()
            ltxvaudiovaedecode = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
            vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
            previewany = NODE_CLASS_MAPPINGS["PreviewAny"]()
            for q in range(1):
                comfynumberconvert_252_93 = comfynumberconvert.EXECUTE_NORMALIZED(
                    value=get_value_at_index(intconstant_252_91, 0)
                )
                conditioningzeroout_247_84 = conditioningzeroout.zero_out(
                    conditioning=get_value_at_index(cliptextencode_247_86, 0)
                )
                ltxvconditioning_247_90 = ltxvconditioning.EXECUTE_NORMALIZED(
                    frame_rate=get_value_at_index(comfynumberconvert_252_93, 0),
                    positive=get_value_at_index(cliptextencode_247_86, 0),
                    negative=get_value_at_index(conditioningzeroout_247_84, 0),
                )
                cfgguider_248_80 = cfgguider.EXECUTE_NORMALIZED(
                    cfg=1,
                    model=get_value_at_index(unetloadergguf_254_240, 0),
                    positive=get_value_at_index(ltxvconditioning_247_90, 0),
                    negative=get_value_at_index(ltxvconditioning_247_90, 1),
                )
                getimagesizeandcount_252_87 = getimagesizeandcount.getsize(
                    image=get_value_at_index(loadimage_246, 0)
                )
                ltxvpreprocess_252_75 = ltxvpreprocess.EXECUTE_NORMALIZED(
                    img_compression=18,
                    image=get_value_at_index(getimagesizeandcount_252_87, 0),
                )
                comfyswitchnode_252_190_199 = comfyswitchnode.EXECUTE_NORMALIZED(
                    switch=["252:190:197", 2],
                    on_false=["252:190:189", 1],
                    on_true=get_value_at_index(primitiveint_252_190_198, 0),
                )
                emptyltxvlatentvideo_252_76 = emptyltxvlatentvideo.EXECUTE_NORMALIZED(
                    width=["252:187:186", 1],
                    height=get_value_at_index(comfyswitchnode_252_190_199, 0),
                    length=97,
                    batch_size=1,
                )
                ltxvimgtovideoinplace_248_145 = (
                    ltxvimgtovideoinplace.EXECUTE_NORMALIZED(
                        strength=1,
                        bypass=False,
                        vae=get_value_at_index(vaeloaderkj_254_218, 0),
                        image=get_value_at_index(ltxvpreprocess_252_75, 0),
                        latent=get_value_at_index(emptyltxvlatentvideo_252_76, 0),
                    )
                )
                comfyswitchnode_252_184_194 = comfyswitchnode.EXECUTE_NORMALIZED(
                    switch=["252:184:193", 2],
                    on_false=["252:184:183", 1],
                    on_true=get_value_at_index(primitiveint_252_184_196, 0),
                )
                ltxvemptylatentaudio_252_89 = ltxvemptylatentaudio.EXECUTE_NORMALIZED(
                    frames_number=get_value_at_index(comfyswitchnode_252_184_194, 0),
                    frame_rate=get_value_at_index(comfynumberconvert_252_93, 1),
                    batch_size=1,
                    audio_vae=get_value_at_index(vaeloaderkj_254_219, 0),
                )
                ltxvconcatavlatent_248_144 = ltxvconcatavlatent.EXECUTE_NORMALIZED(
                    video_latent=get_value_at_index(ltxvimgtovideoinplace_248_145, 0),
                    audio_latent=get_value_at_index(ltxvemptylatentaudio_252_89, 0),
                )
                ltxvscheduler_248_255 = ltxvscheduler.EXECUTE_NORMALIZED(
                    steps=8,
                    max_shift=2.05,
                    base_shift=0.95,
                    stretch=True,
                    terminal=0.1,
                    latent=get_value_at_index(ltxvconcatavlatent_248_144, 0),
                )
                samplercustomadvanced_248_79 = samplercustomadvanced.EXECUTE_NORMALIZED(
                    noise=get_value_at_index(randomnoise_248_85, 0),
                    guider=get_value_at_index(cfgguider_248_80, 0),
                    sampler=get_value_at_index(ksamplerselect_248_81, 0),
                    sigmas=get_value_at_index(ltxvscheduler_248_255, 0),
                    latent_image=get_value_at_index(ltxvconcatavlatent_248_144, 0),
                )
                ltxvseparateavlatent_248_78 = ltxvseparateavlatent.EXECUTE_NORMALIZED(
                    av_latent=get_value_at_index(samplercustomadvanced_248_79, 0)
                )
                ltxvspatiotemporaltiledvaedecode_249_117 = (
                    ltxvspatiotemporaltiledvaedecode.decode_spatial_temporal(
                        spatial_tiles=4,
                        spatial_overlap=8,
                        temporal_tile_length=64,
                        temporal_overlap=8,
                        last_frame_fix=False,
                        working_device="auto",
                        working_dtype="auto",
                        vae=get_value_at_index(vaeloaderkj_254_218, 0),
                        latents=get_value_at_index(ltxvseparateavlatent_248_78, 0),
                    )
                )
                ltxvaudiovaedecode_249_92 = ltxvaudiovaedecode.EXECUTE_NORMALIZED(
                    samples=get_value_at_index(ltxvseparateavlatent_248_78, 1),
                    audio_vae=get_value_at_index(vaeloaderkj_254_219, 0),
                )
                vhs_videocombine_250 = vhs_videocombine.combine_video(
                    frame_rate=get_value_at_index(comfynumberconvert_252_93, 0),
                    loop_count=0,
                    filename_prefix="ltx23",
                    format="video/h264-mp4",
                    pix_fmt="yuv420p",
                    crf=19,
                    save_metadata=True,
                    trim_to_audio=False,
                    pingpong=False,
                    save_output=True,
                    images=get_value_at_index(
                        ltxvspatiotemporaltiledvaedecode_249_117, 0
                    ),
                    audio=get_value_at_index(ltxvaudiovaedecode_249_92, 0),
                    unique_id=15204198120938839924,
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                )
                previewany_252_203 = previewany.main(
                    source=get_value_at_index(comfyswitchnode_252_190_199, 0)
                )
                previewany_252_204 = previewany.main(
                    source=get_value_at_index(comfyswitchnode_252_184_194, 0)
                )
                previewany_252_202 = previewany.main(source=["252:187:186", 1])

                import folder_paths
                import os
                import glob as glob_module

                output_dir = folder_paths.get_output_directory()
                pattern = os.path.join(output_dir, "ltx23_*.mp4")
                files = sorted(glob_module.glob(pattern), key=os.path.getmtime)
                if files:
                    return files[-1]
                return None
    finally:
        cleanup_comfyui_runtime(unload_models=unload_models)
