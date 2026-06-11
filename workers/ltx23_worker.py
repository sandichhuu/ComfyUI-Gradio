import json
import os
import random
from typing import Any
import math

from workers.comfy_worker import (
    get_value_at_index,
    bootstrap_comfyui_runtime,
    add_extra_model_paths,
    import_custom_nodes,
    cleanup_comfyui_runtime,
)

def build_workflow() -> dict[str, Any]:
    return {
        "287": {
            "inputs": {
                "frame_rate": ["395", 0],
                "loop_count": 0,
                "filename_prefix": "ltx23",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": False,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
                "images": ["289:275", 0],
                "audio": ["289:92", 0],
            },
            "class_type": "VHS_VideoCombine",
            "_meta": {"title": "Video Combine 🎥🅥🅗🅢"},
        },
        "364": {
            "inputs": {"image": "hanfu.jpg"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"},
        },
        "378": {
            "inputs": {"value": 11},
            "class_type": "INTConstant",
            "_meta": {"title": "FPS Int"},
        },
        "380": {
            "inputs": {
                "width": ["393", 1],
                "height": ["393", 2],
                "length": ["390", 0],
                "batch_size": 1,
            },
            "class_type": "EmptyLTXVLatentVideo",
            "_meta": {"title": "EmptyLTXVLatentVideo"},
        },
        "381": {
            "inputs": {
                "frames_number": ["390", 0],
                "frame_rate": ["378", 0],
                "batch_size": 1,
                "audio_vae": ["286:219", 0],
            },
            "class_type": "LTXVEmptyLatentAudio",
            "_meta": {"title": "LTXV Empty Latent Audio"},
        },
        "384": {
            "inputs": {"img_compression": 18, "image": ["393", 0]},
            "class_type": "LTXVPreprocess",
            "_meta": {"title": "LTXV Preprocess"},
        },
        "390": {
            "inputs": {"value": 65},
            "class_type": "INTConstant",
            "_meta": {"title": "FrameCount"},
        },
        "391": {
            "inputs": {"value": 129},
            "class_type": "INTConstant",
            "_meta": {"title": "Width"},
        },
        "392": {
            "inputs": {"value": 129},
            "class_type": "INTConstant",
            "_meta": {"title": "Height"},
        },
        "393": {
            "inputs": {
                "width": ["391", 0],
                "height": ["392", 0],
                "upscale_method": "bicubic",
                "keep_proportion": "crop",
                "pad_color": "0, 0, 0",
                "crop_position": "center",
                "divisible_by": 2,
                "device": "cpu",
                "image": ["364", 0],
            },
            "class_type": "ImageResizeKJv2",
            "_meta": {"title": "Resize Image v2"},
        },
        "395": {
            "inputs": {"value": 11},
            "class_type": "PrimitiveFloat",
            "_meta": {"title": "FPS Float"},
        },
        "286:218": {
            "inputs": {
                "vae_name": "ltx-2.3-22b-distilled_video_vae.safetensors",
                "device": "main_device",
                "weight_dtype": "bf16",
            },
            "class_type": "VAELoaderKJ",
            "_meta": {"title": "VAELoader KJ"},
        },
        "286:219": {
            "inputs": {
                "vae_name": "ltx-2.3-22b-distilled_audio_vae.safetensors",
                "device": "main_device",
                "weight_dtype": "bf16",
            },
            "class_type": "VAELoaderKJ",
            "_meta": {"title": "VAELoader KJ"},
        },
        "286:239": {
            "inputs": {
                "clip_name1": "gemma-3-12b-it-qat-UD-Q2_K_XL.gguf",
                "clip_name2": "ltx-2.3-22b-distilled_embeddings_connectors.safetensors",
                "type": "ltxv",
            },
            "class_type": "DualCLIPLoaderGGUF",
            "_meta": {"title": "DualCLIPLoader (GGUF)"},
        },
        "286:240": {
            "inputs": {"unet_name": "Unknown/ltx-2.3-22b-distilled-1.1-UD-Q2_K.gguf"},
            "class_type": "UnetLoaderGGUF",
            "_meta": {"title": "Unet Loader (GGUF)"},
        },
        "283:86": {
            "inputs": {
                "text": "Character start flying to the sky",
                "clip": ["286:239", 0],
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"},
        },
        "283:84": {
            "inputs": {"conditioning": ["283:86", 0]},
            "class_type": "ConditioningZeroOut",
            "_meta": {"title": "ConditioningZeroOut"},
        },
        "283:90": {
            "inputs": {
                "frame_rate": ["395", 0],
                "positive": ["283:86", 0],
                "negative": ["283:84", 0],
            },
            "class_type": "LTXVConditioning",
            "_meta": {"title": "LTXVConditioning"},
        },
        "284:78": {
            "inputs": {"av_latent": ["284:79", 0]},
            "class_type": "LTXVSeparateAVLatent",
            "_meta": {"title": "LTXVSeparateAVLatent"},
        },
        "284:85": {
            "inputs": {"noise_seed": 10},
            "class_type": "RandomNoise",
            "_meta": {"title": "RandomNoise"},
        },
        "284:79": {
            "inputs": {
                "noise": ["284:85", 0],
                "guider": ["284:80", 0],
                "sampler": ["284:81", 0],
                "sigmas": ["284:255", 0],
                "latent_image": ["284:144", 0],
            },
            "class_type": "SamplerCustomAdvanced",
            "_meta": {"title": "SamplerCustomAdvanced"},
        },
        "284:145": {
            "inputs": {
                "strength": 1,
                "bypass": False,
                "vae": ["286:218", 0],
                "image": ["384", 0],
                "latent": ["380", 0],
            },
            "class_type": "LTXVImgToVideoInplace",
            "_meta": {"title": "LTXVImgToVideoInplace"},
        },
        "284:144": {
            "inputs": {"video_latent": ["284:145", 0], "audio_latent": ["381", 0]},
            "class_type": "LTXVConcatAVLatent",
            "_meta": {"title": "LTXVConcatAVLatent"},
        },
        "284:80": {
            "inputs": {
                "cfg": 1,
                "model": ["286:240", 0],
                "positive": ["283:90", 0],
                "negative": ["283:90", 1],
            },
            "class_type": "CFGGuider",
            "_meta": {"title": "CFG Guider"},
        },
        "284:81": {
            "inputs": {"sampler_name": "euler_ancestral"},
            "class_type": "KSamplerSelect",
            "_meta": {"title": "KSamplerSelect"},
        },
        "284:255": {
            "inputs": {
                "steps": 8,
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1,
                "latent": ["284:144", 0],
            },
            "class_type": "LTXVScheduler",
            "_meta": {"title": "LTXVScheduler"},
        },
        "289:92": {
            "inputs": {"samples": ["284:78", 1], "audio_vae": ["286:219", 0]},
            "class_type": "LTXVAudioVAEDecode",
            "_meta": {"title": "LTXV Audio VAE Decode"},
        },
        "289:275": {
            "inputs": {"samples": ["284:78", 0], "vae": ["286:218", 0]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
        },
    }

def build_extra_pnginfo() -> dict[str, Any] | None:
    return {
        "workflow": {
            "id": "4ecb70e4-8de9-44cc-8d0c-15171188fa13",
            "revision": 0,
            "last_node_id": 395,
            "last_link_id": 824,
            "nodes": [
                {
                    "id": 278,
                    "type": "SetNode",
                    "pos": [2860.5080081627852, 1002.5406481522916],
                    "size": [225, 8],
                    "flags": {"collapsed": True},
                    "order": 14,
                    "mode": 0,
                    "inputs": [
                        {"name": "CONDITIONING", "type": "CONDITIONING", "link": 607}
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
                    "id": 279,
                    "type": "SetNode",
                    "pos": [2355.2622608055685, 1318.2222216258403],
                    "size": [225, 8],
                    "flags": {"collapsed": True},
                    "order": 9,
                    "mode": 0,
                    "inputs": [{"name": "VAE", "type": "VAE", "link": 608}],
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
                    "id": 280,
                    "type": "SetNode",
                    "pos": [2355.262260805569, 1369.4991243605857],
                    "size": [225, 8],
                    "flags": {"collapsed": True},
                    "order": 10,
                    "mode": 0,
                    "inputs": [{"name": "VAE", "type": "VAE", "link": 609}],
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
                    "id": 282,
                    "type": "SetNode",
                    "pos": [2860.5563990986216, 1081.103736300067],
                    "size": [225, 8],
                    "flags": {"collapsed": True},
                    "order": 15,
                    "mode": 0,
                    "inputs": [
                        {"name": "CONDITIONING", "type": "CONDITIONING", "link": 611}
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
                    "id": 283,
                    "type": "fd5b34a1-d163-4ace-bafc-1dc09e136bcc",
                    "pos": [2352.314167950462, 982.4120184075734],
                    "size": [453.2833251953125, 279.75],
                    "flags": {},
                    "order": 8,
                    "mode": 0,
                    "inputs": [{"name": "clip", "type": "CLIP", "link": 612}],
                    "outputs": [
                        {"name": "positive", "type": "CONDITIONING", "links": [607]},
                        {"name": "negative", "type": "CONDITIONING", "links": [611]},
                    ],
                    "properties": {"previewExposures": []},
                    "widgets_values": ["Character start flying to the " "sky"],
                    "color": "#232",
                    "bgcolor": "#353",
                },
                {
                    "id": 284,
                    "type": "649a3df7-755a-4259-9f4f-35ace1bbc724",
                    "pos": [2681.477465507165, 1360.2135496466742],
                    "size": [285.6166687011719, 224],
                    "flags": {},
                    "order": 19,
                    "mode": 0,
                    "inputs": [
                        {
                            "label": "distilled_model",
                            "name": "model",
                            "type": "MODEL",
                            "link": 613,
                        },
                        {
                            "label": "video_latent",
                            "name": "latent",
                            "type": "LATENT",
                            "link": 793,
                        },
                        {"name": "audio_latent", "type": "LATENT", "link": 794},
                    ],
                    "outputs": [
                        {"name": "video_latent", "type": "LATENT", "links": [620]},
                        {"name": "audio_latent", "type": "LATENT", "links": [621]},
                    ],
                    "properties": {"previewExposures": []},
                    "widgets_values": [1, "euler_ancestral", 8],
                    "color": "#223",
                    "bgcolor": "#335",
                },
                {
                    "id": 287,
                    "type": "VHS_VideoCombine",
                    "pos": [3295.728930693509, 1002.2410044587746],
                    "size": [459.4666748046875, 1276.0833740234375],
                    "flags": {},
                    "order": 21,
                    "mode": 0,
                    "inputs": [
                        {"name": "images", "type": "IMAGE", "link": 736},
                        {"name": "audio", "shape": 7, "type": "AUDIO", "link": 618},
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
                            "link": 619,
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
                        "save_metadata": False,
                        "trim_to_audio": False,
                        "pingpong": False,
                        "save_output": True,
                        "videopreview": {
                            "hidden": False,
                            "paused": False,
                            "params": {
                                "filename": "ltx23_00019-audio.mp4",
                                "subfolder": "",
                                "type": "output",
                                "format": "video/h264-mp4",
                                "frame_rate": 25,
                                "workflow": "ltx23_00019.png",
                                "fullpath": "/comfy/output/ltx23_00019-audio.mp4",
                            },
                        },
                    },
                },
                {
                    "id": 286,
                    "type": "ee84b951-7b5e-42f3-8477-94927a4e62a6",
                    "pos": [1773.2527802472123, 976.856749606987],
                    "size": [552.2166748046875, 388],
                    "flags": {},
                    "order": 0,
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
                            "links": [613],
                        },
                        {
                            "label": "clip",
                            "name": "CLIP",
                            "type": "CLIP",
                            "links": [612],
                        },
                        {
                            "label": "video_vae",
                            "name": "VAE",
                            "type": "VAE",
                            "links": [608],
                        },
                        {
                            "label": "audio_vae",
                            "name": "VAE_1",
                            "type": "VAE",
                            "links": [609],
                        },
                    ],
                    "properties": {"previewExposures": []},
                    "widgets_values": [
                        "Unknown/ltx-2.3-22b-distilled-1.1-UD-Q2_K.gguf",
                        "gemma-3-12b-it-qat-UD-Q2_K_XL.gguf",
                        "ltx-2.3-22b-distilled_embeddings_connectors.safetensors",
                        "ltx-2.3-22b-distilled_video_vae.safetensors",
                        "ltx-2.3-22b-distilled_audio_vae.safetensors",
                    ],
                    "color": "#432",
                    "bgcolor": "#653",
                },
                {
                    "id": 364,
                    "type": "LoadImage",
                    "pos": [630.5812032529307, 980.7353722880327],
                    "size": [540.7000122070312, 340],
                    "flags": {},
                    "order": 1,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [
                        {"name": "IMAGE", "type": "IMAGE", "links": [816]},
                        {"name": "MASK", "type": "MASK", "links": None},
                    ],
                    "properties": {"Node name for S&R": "LoadImage"},
                    "widgets_values": ["hanfu.jpg", "image"],
                },
                {
                    "id": 378,
                    "type": "INTConstant",
                    "pos": [722.0333414739051, 2141.1198875287973],
                    "size": [270.6166687011719, 104],
                    "flags": {},
                    "order": 2,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "value", "type": "INT", "links": [822]}],
                    "title": "FPS Int",
                    "properties": {"Node name for S&R": "INTConstant"},
                    "widgets_values": [11],
                    "color": "#1b4669",
                    "bgcolor": "#29699c",
                },
                {
                    "id": 380,
                    "type": "EmptyLTXVLatentVideo",
                    "pos": [2179.0513165875154, 1860.6461376367417],
                    "size": [270, 176],
                    "flags": {},
                    "order": 17,
                    "mode": 0,
                    "inputs": [
                        {
                            "name": "width",
                            "type": "INT",
                            "widget": {"name": "width"},
                            "link": 820,
                        },
                        {
                            "name": "height",
                            "type": "INT",
                            "widget": {"name": "height"},
                            "link": 821,
                        },
                        {
                            "name": "length",
                            "type": "INT",
                            "widget": {"name": "length"},
                            "link": 815,
                        },
                    ],
                    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [793]}],
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
                    "id": 381,
                    "type": "LTXVEmptyLatentAudio",
                    "pos": [2182.687680223879, 2086.1006830912884],
                    "size": [270, 144],
                    "flags": {},
                    "order": 11,
                    "mode": 0,
                    "inputs": [
                        {"name": "audio_vae", "type": "VAE", "link": 795},
                        {
                            "name": "frames_number",
                            "type": "INT",
                            "widget": {"name": "frames_number"},
                            "link": 814,
                        },
                        {
                            "name": "frame_rate",
                            "type": "INT",
                            "widget": {"name": "frame_rate"},
                            "link": 822,
                        },
                    ],
                    "outputs": [{"name": "Latent", "type": "LATENT", "links": [794]}],
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
                    "id": 383,
                    "type": "GetNode",
                    "pos": [795.1369220990498, 2533.0435497494623],
                    "size": [225, 104],
                    "flags": {"collapsed": False},
                    "order": 3,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "VAE", "type": "VAE", "links": [795]}],
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
                    "id": 384,
                    "type": "LTXVPreprocess",
                    "pos": [2181.3077539508768, 2297.246137636742],
                    "size": [271.5333251953125, 80.71666717529297],
                    "flags": {},
                    "order": 16,
                    "mode": 0,
                    "inputs": [{"name": "image", "type": "IMAGE", "link": 819}],
                    "outputs": [
                        {"name": "output_image", "type": "IMAGE", "links": [796]}
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
                    "id": 390,
                    "type": "INTConstant",
                    "pos": [727.0401483252548, 1993.908965342988],
                    "size": [270.6166687011719, 104],
                    "flags": {"collapsed": False},
                    "order": 4,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "value", "type": "INT", "links": [814, 815]}],
                    "title": "FrameCount",
                    "properties": {"Node name for S&R": "INTConstant"},
                    "widgets_values": [65],
                    "color": "#1b4669",
                    "bgcolor": "#29699c",
                },
                {
                    "id": 391,
                    "type": "INTConstant",
                    "pos": [724.584376037456, 1688.5355140210922],
                    "size": [270.6166687011719, 104],
                    "flags": {"collapsed": False},
                    "order": 5,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "value", "type": "INT", "links": [817]}],
                    "title": "Width",
                    "properties": {"Node name for S&R": "INTConstant"},
                    "widgets_values": [129],
                    "color": "#1b4669",
                    "bgcolor": "#29699c",
                },
                {
                    "id": 392,
                    "type": "INTConstant",
                    "pos": [723.0313226034239, 1843.0387781086563],
                    "size": [270.6166687011719, 104],
                    "flags": {"collapsed": False},
                    "order": 6,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "value", "type": "INT", "links": [818]}],
                    "title": "Height",
                    "properties": {"Node name for S&R": "INTConstant"},
                    "widgets_values": [129],
                    "color": "#1b4669",
                    "bgcolor": "#29699c",
                },
                {
                    "id": 289,
                    "type": "4eb8d8e1-d7b4-4ffc-9410-1a1990b6d58d",
                    "pos": [3019.0156429108642, 1360.787927530866],
                    "size": [225, 148],
                    "flags": {},
                    "order": 20,
                    "mode": 0,
                    "inputs": [
                        {
                            "label": "video_latent",
                            "name": "latents",
                            "type": "LATENT",
                            "link": 620,
                        },
                        {
                            "label": "audio_latent",
                            "name": "samples",
                            "type": "LATENT",
                            "link": 621,
                        },
                    ],
                    "outputs": [
                        {
                            "label": "frames",
                            "name": "image",
                            "type": "IMAGE",
                            "links": [736],
                        },
                        {
                            "label": "audio",
                            "name": "Audio",
                            "type": "AUDIO",
                            "links": [618],
                        },
                        {
                            "label": "fps",
                            "name": "FLOAT",
                            "type": "FLOAT",
                            "links": [619],
                        },
                    ],
                    "properties": {"previewExposures": []},
                    "color": "#322",
                    "bgcolor": "#533",
                },
                {
                    "id": 367,
                    "type": "SetNode",
                    "pos": [2526.260938401213, 2321.865447562757],
                    "size": [225, 8],
                    "flags": {"collapsed": True},
                    "order": 18,
                    "mode": 0,
                    "inputs": [{"name": "IMAGE", "type": "IMAGE", "link": 796}],
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
                    "id": 395,
                    "type": "PrimitiveFloat",
                    "pos": [720.9187448853406, 2290.0923260116247],
                    "size": [270, 58],
                    "flags": {"collapsed": False},
                    "order": 7,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "FLOAT", "type": "FLOAT", "links": [824]}],
                    "title": "FPS Float",
                    "properties": {"Node name for S&R": "PrimitiveFloat"},
                    "widgets_values": [11],
                },
                {
                    "id": 366,
                    "type": "SetNode",
                    "pos": [1053.2859471891206, 2317.4827084929857],
                    "size": [225, 50],
                    "flags": {"collapsed": True},
                    "order": 13,
                    "mode": 0,
                    "inputs": [{"name": "FLOAT", "type": "FLOAT", "link": 824}],
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
                    "id": 393,
                    "type": "ImageResizeKJv2",
                    "pos": [1230.7640257170158, 1459.9970417644922],
                    "size": [385, 404],
                    "flags": {},
                    "order": 12,
                    "mode": 0,
                    "inputs": [
                        {"name": "image", "type": "IMAGE", "link": 816},
                        {"name": "mask", "shape": 7, "type": "MASK", "link": None},
                        {
                            "name": "width",
                            "type": "INT",
                            "widget": {"name": "width"},
                            "link": 817,
                        },
                        {
                            "name": "height",
                            "type": "INT",
                            "widget": {"name": "height"},
                            "link": 818,
                        },
                    ],
                    "outputs": [
                        {"name": "IMAGE", "type": "IMAGE", "links": [819]},
                        {"name": "width", "type": "INT", "links": [820]},
                        {"name": "height", "type": "INT", "links": [821]},
                        {"name": "mask", "type": "MASK", "links": None},
                    ],
                    "properties": {"Node name for S&R": "ImageResizeKJv2"},
                    "widgets_values": [
                        512,
                        512,
                        "bicubic",
                        "crop",
                        "0, 0, 0",
                        "center",
                        2,
                        "cpu",
                    ],
                },
            ],
            "links": [
                [607, 283, 0, 278, 0, "CONDITIONING"],
                [608, 286, 2, 279, 0, "VAE"],
                [609, 286, 3, 280, 0, "VAE"],
                [611, 283, 1, 282, 0, "CONDITIONING"],
                [612, 286, 1, 283, 0, "CLIP"],
                [613, 286, 0, 284, 0, "MODEL"],
                [618, 289, 1, 287, 1, "AUDIO"],
                [619, 289, 2, 287, 4, "FLOAT"],
                [620, 284, 0, 289, 0, "LATENT"],
                [621, 284, 1, 289, 1, "LATENT"],
                [736, 289, 0, 287, 0, "IMAGE"],
                [793, 380, 0, 284, 1, "LATENT"],
                [794, 381, 0, 284, 2, "LATENT"],
                [795, 383, 0, 381, 0, "VAE"],
                [796, 384, 0, 367, 0, "IMAGE"],
                [814, 390, 0, 381, 1, "INT"],
                [815, 390, 0, 380, 2, "INT"],
                [816, 364, 0, 393, 0, "IMAGE"],
                [817, 391, 0, 393, 2, "INT"],
                [818, 392, 0, 393, 3, "INT"],
                [819, 393, 0, 384, 0, "IMAGE"],
                [820, 393, 1, 380, 0, "INT"],
                [821, 393, 2, 380, 1, "INT"],
                [822, 378, 0, 381, 2, "INT"],
                [824, 395, 0, 366, 0, "FLOAT"],
            ],
            "groups": [],
            "definitions": {
                "subgraphs": [
                    {
                        "id": "fd5b34a1-d163-4ace-bafc-1dc09e136bcc",
                        "version": 1,
                        "state": {
                            "lastGroupId": 0,
                            "lastNodeId": 395,
                            "lastLinkId": 824,
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
                                    "secondTabText": "Send " "Back",
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
                                    "secondTabText": "Send " "Back",
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
                        "id": "649a3df7-755a-4259-9f4f-35ace1bbc724",
                        "version": 1,
                        "state": {
                            "lastGroupId": 0,
                            "lastNodeId": 395,
                            "lastLinkId": 824,
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
                                    "secondTabText": "Send " "Back",
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
                                    "secondTabText": "Send " "Back",
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
                                    "secondTabText": "Send " "Back",
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
                                    "secondTabText": "Send " "Back",
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
                                    "secondTabText": "Send " "Back",
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
                                    "secondTabText": "Send " "Back",
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
                                    "secondTabText": "Send " "Back",
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
                        "id": "ee84b951-7b5e-42f3-8477-94927a4e62a6",
                        "version": 1,
                        "state": {
                            "lastGroupId": 0,
                            "lastNodeId": 395,
                            "lastLinkId": 824,
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
                                    "ltx-2.3-22b-distilled_embeddings_connectors.safetensors",
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
                        "id": "4eb8d8e1-d7b4-4ffc-9410-1a1990b6d58d",
                        "version": 1,
                        "state": {
                            "lastGroupId": 0,
                            "lastNodeId": 395,
                            "lastLinkId": 824,
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
                                "linkIds": [603],
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
                                "linkIds": [604],
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
                                        "label": "Audio " "VAE",
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
                                    "secondTabText": "Send " "Back",
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
                                "order": 0,
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
                            {
                                "id": 275,
                                "type": "VAEDecode",
                                "pos": [3516.220575263245, 2799.590810148425],
                                "size": [242.3469019058707, 46],
                                "flags": {},
                                "order": 4,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "samples",
                                        "name": "samples",
                                        "type": "LATENT",
                                        "link": 603,
                                    },
                                    {
                                        "localized_name": "vae",
                                        "name": "vae",
                                        "type": "VAE",
                                        "link": 602,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "IMAGE",
                                        "name": "IMAGE",
                                        "type": "IMAGE",
                                        "links": [604],
                                    }
                                ],
                                "properties": {"Node name for S&R": "VAEDecode"},
                                "widgets_values": [],
                            },
                            {
                                "id": 160,
                                "type": "GetNode",
                                "pos": [3033.391763836357, 3020.808236152122],
                                "size": [266.49198725913675, 58],
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
                                "id": 159,
                                "type": "GetNode",
                                "pos": [3033.0559920961036, 2819.487822133166],
                                "size": [267.1580612050311, 58],
                                "flags": {"collapsed": False},
                                "order": 2,
                                "mode": 0,
                                "inputs": [],
                                "outputs": [
                                    {"name": "VAE", "type": "VAE", "links": [602]}
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
                        ],
                        "groups": [],
                        "links": [
                            {
                                "id": 263,
                                "origin_id": -10,
                                "origin_slot": 1,
                                "target_id": 92,
                                "target_slot": 0,
                                "type": "LATENT",
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
                            {
                                "id": 602,
                                "origin_id": 159,
                                "origin_slot": 0,
                                "target_id": 275,
                                "target_slot": 1,
                                "type": "VAE",
                            },
                            {
                                "id": 603,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 275,
                                "target_slot": 0,
                                "type": "LATENT",
                            },
                            {
                                "id": 604,
                                "origin_id": 275,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 0,
                                "type": "IMAGE",
                            },
                        ],
                        "extra": {},
                    },
                ]
            },
            "config": {},
            "extra": {
                "ds": {
                    "scale": 0.5929134837981407,
                    "offset": [-308.9295491025214, -888.5766932151855],
                },
                "frontendVersion": "1.46.13",
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

def validate_frame_size(frame_size: int) -> int:
    return max(32, (frame_size // 32) * 32)

def validate_frame_count(duration: float, fps: int) -> int:
    frame_count = duration * fps
    count_int = max(1, int(frame_count))
    ceil_steps = (count_int + 6) // 8
    return max(9, (ceil_steps * 8) + 1)

def generate(
    prompt_text: str,
    image_path: str,
    duration: float,
    fps: int,
    image_compression: int,
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
        VAEDecode,
    )

    from PIL import Image
    with Image.open(image_path) as img:
        width, height = img.size

    import torch

    try:
        with torch.inference_mode():
            loadimage = LoadImage()
            loadimage_364 = loadimage.load_image(image=image_path)
            intconstant = NODE_CLASS_MAPPINGS["INTConstant"]()
            intconstant_378 = intconstant.get_value(value=fps)
            intconstant_390 = intconstant.get_value(value=validate_frame_count(duration, fps))
            intconstant_391 = intconstant.get_value(value=validate_frame_size(width))
            intconstant_392 = intconstant.get_value(value=validate_frame_size(height))
            primitivefloat = NODE_CLASS_MAPPINGS["PrimitiveFloat"]()
            primitivefloat_395 = primitivefloat.EXECUTE_NORMALIZED(value=float(fps))
            vaeloaderkj = NODE_CLASS_MAPPINGS["VAELoaderKJ"]()
            vaeloaderkj_286_218 = vaeloaderkj.load_vae(
                vae_name="ltx-2.3-22b-distilled_video_vae.safetensors",
                device="main_device",
                weight_dtype="bf16",
            )
            vaeloaderkj_286_219 = vaeloaderkj.load_vae(
                vae_name="ltx-2.3-22b-distilled_audio_vae.safetensors",
                device="main_device",
                weight_dtype="bf16",
            )
            dualcliploadergguf = NODE_CLASS_MAPPINGS["DualCLIPLoaderGGUF"]()
            dualcliploadergguf_286_239 = dualcliploadergguf.load_clip(
                clip_name1="gemma-3-12b-it-qat-UD-Q2_K_XL.gguf",
                clip_name2="ltx-2.3-22b-distilled_embeddings_connectors.safetensors",
                type="ltxv",
            )
            unetloadergguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
            unetloadergguf_286_240 = unetloadergguf.load_unet(
                unet_name="Unknown/ltx-2.3-22b-distilled-1.1-UD-Q2_K.gguf"
            )
            cliptextencode = CLIPTextEncode()
            cliptextencode_283_86 = cliptextencode.encode(
                text=prompt_text,
                clip=get_value_at_index(dualcliploadergguf_286_239, 0),
            )
            randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
            node_284_85_noise_seed = prompt["284:85"]["inputs"]["noise_seed"] = (
                random.randint(1, 2**64)
            )
            randomnoise_284_85 = randomnoise.EXECUTE_NORMALIZED(
                noise_seed=node_284_85_noise_seed
            )
            ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
            ksamplerselect_284_81 = ksamplerselect.EXECUTE_NORMALIZED(
                sampler_name="euler_ancestral"
            )
            conditioningzeroout = ConditioningZeroOut()
            ltxvconditioning = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
            cfgguider = NODE_CLASS_MAPPINGS["CFGGuider"]()
            imageresizekjv2 = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
            ltxvpreprocess = NODE_CLASS_MAPPINGS["LTXVPreprocess"]()
            emptyltxvlatentvideo = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()
            ltxvimgtovideoinplace = NODE_CLASS_MAPPINGS["LTXVImgToVideoInplace"]()
            ltxvemptylatentaudio = NODE_CLASS_MAPPINGS["LTXVEmptyLatentAudio"]()
            ltxvconcatavlatent = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
            ltxvscheduler = NODE_CLASS_MAPPINGS["LTXVScheduler"]()
            samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
            ltxvseparateavlatent = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
            vaedecode = VAEDecode()
            ltxvaudiovaedecode = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
            vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
            for q in range(1):
                conditioningzeroout_283_84 = conditioningzeroout.zero_out(
                    conditioning=get_value_at_index(cliptextencode_283_86, 0)
                )
                ltxvconditioning_283_90 = ltxvconditioning.EXECUTE_NORMALIZED(
                    frame_rate=get_value_at_index(primitivefloat_395, 0),
                    positive=get_value_at_index(cliptextencode_283_86, 0),
                    negative=get_value_at_index(conditioningzeroout_283_84, 0),
                )
                cfgguider_284_80 = cfgguider.EXECUTE_NORMALIZED(
                    cfg=1,
                    model=get_value_at_index(unetloadergguf_286_240, 0),
                    positive=get_value_at_index(ltxvconditioning_283_90, 0),
                    negative=get_value_at_index(ltxvconditioning_283_90, 1),
                )
                imageresizekjv2_393 = imageresizekjv2.resize(
                    width=get_value_at_index(intconstant_391, 0),
                    height=get_value_at_index(intconstant_392, 0),
                    upscale_method="bicubic",
                    keep_proportion="crop",
                    pad_color="0, 0, 0",
                    crop_position="center",
                    divisible_by=2,
                    device="cpu",
                    image=get_value_at_index(loadimage_364, 0),
                    unique_id=11550909831398452233,
                )
                ltxvpreprocess_384 = ltxvpreprocess.EXECUTE_NORMALIZED(
                    img_compression=image_compression, image=get_value_at_index(imageresizekjv2_393, 0)
                )
                emptyltxvlatentvideo_380 = emptyltxvlatentvideo.EXECUTE_NORMALIZED(
                    width=get_value_at_index(imageresizekjv2_393, 1),
                    height=get_value_at_index(imageresizekjv2_393, 2),
                    length=get_value_at_index(intconstant_390, 0),
                    batch_size=1,
                )
                ltxvimgtovideoinplace_284_145 = (
                    ltxvimgtovideoinplace.EXECUTE_NORMALIZED(
                        strength=1,
                        bypass=False,
                        vae=get_value_at_index(vaeloaderkj_286_218, 0),
                        image=get_value_at_index(ltxvpreprocess_384, 0),
                        latent=get_value_at_index(emptyltxvlatentvideo_380, 0),
                    )
                )
                ltxvemptylatentaudio_381 = ltxvemptylatentaudio.EXECUTE_NORMALIZED(
                    frames_number=get_value_at_index(intconstant_390, 0),
                    frame_rate=get_value_at_index(intconstant_378, 0),
                    batch_size=1,
                    audio_vae=get_value_at_index(vaeloaderkj_286_219, 0),
                )
                ltxvconcatavlatent_284_144 = ltxvconcatavlatent.EXECUTE_NORMALIZED(
                    video_latent=get_value_at_index(ltxvimgtovideoinplace_284_145, 0),
                    audio_latent=get_value_at_index(ltxvemptylatentaudio_381, 0),
                )
                ltxvscheduler_284_255 = ltxvscheduler.EXECUTE_NORMALIZED(
                    steps=8,
                    max_shift=2.05,
                    base_shift=0.95,
                    stretch=True,
                    terminal=0.1,
                    latent=get_value_at_index(ltxvconcatavlatent_284_144, 0),
                )
                samplercustomadvanced_284_79 = samplercustomadvanced.EXECUTE_NORMALIZED(
                    noise=get_value_at_index(randomnoise_284_85, 0),
                    guider=get_value_at_index(cfgguider_284_80, 0),
                    sampler=get_value_at_index(ksamplerselect_284_81, 0),
                    sigmas=get_value_at_index(ltxvscheduler_284_255, 0),
                    latent_image=get_value_at_index(ltxvconcatavlatent_284_144, 0),
                )
                ltxvseparateavlatent_284_78 = ltxvseparateavlatent.EXECUTE_NORMALIZED(
                    av_latent=get_value_at_index(samplercustomadvanced_284_79, 0)
                )
                vaedecode_289_275 = vaedecode.decode(
                    samples=get_value_at_index(ltxvseparateavlatent_284_78, 0),
                    vae=get_value_at_index(vaeloaderkj_286_218, 0),
                )
                ltxvaudiovaedecode_289_92 = ltxvaudiovaedecode.EXECUTE_NORMALIZED(
                    samples=get_value_at_index(ltxvseparateavlatent_284_78, 1),
                    audio_vae=get_value_at_index(vaeloaderkj_286_219, 0),
                )
                vhs_videocombine_287 = vhs_videocombine.combine_video(
                    frame_rate=get_value_at_index(primitivefloat_395, 0),
                    loop_count=0,
                    filename_prefix="ltx23",
                    format="video/h264-mp4",
                    pix_fmt="yuv420p",
                    crf=19,
                    save_metadata=False,
                    trim_to_audio=False,
                    pingpong=False,
                    save_output=True,
                    images=get_value_at_index(vaedecode_289_275, 0),
                    audio=get_value_at_index(ltxvaudiovaedecode_289_92, 0),
                    unique_id=6064538059145465649,
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                )

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