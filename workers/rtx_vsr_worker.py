# Imports
import json
import os
import random
import sys
from typing import Sequence, Mapping, Any, Union


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Return a sequence or mapping result item by index."""
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def get_comfyui_path() -> str:
    """Return the configured ComfyUI path, preferring COMFYUI_PATH when set."""
    comfyui_path = os.environ.get("COMFYUI_PATH")
    if comfyui_path:
        return comfyui_path
    return find_path("ComfyUI")


def find_path(name: str, path: str = None) -> str:
    """Recursively search parent folders until the named entry is found."""
    if path is None:
        path = os.getcwd()

    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None

    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """Add the ComfyUI checkout to sys.path."""
    comfyui_path = get_comfyui_path()
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        if comfyui_path in sys.path:
            sys.path.remove(comfyui_path)
        sys.path.insert(0, comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """Load ComfyUI extra model paths configuration when available."""
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def bootstrap_comfyui_runtime() -> None:
    """Mirror the allocator-related ComfyUI startup steps before torch import."""
    add_comfyui_directory_to_sys_path()

    import comfy.options

    comfy.options.enable_args_parsing()

    from comfy.cli_args import args

    if os.name == "nt":
        os.environ["MIMALLOC_PURGE_DELAY"] = "0"

    if args.default_device is not None:
        default_dev = args.default_device
        devices = list(range(32))
        devices.remove(default_dev)
        devices.insert(0, default_dev)
        devices = ",".join(map(str, devices))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)
        os.environ["HIP_VISIBLE_DEVICES"] = str(devices)

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.cuda_device)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.cuda_device)

    if args.oneapi_device_selector is not None:
        os.environ["ONEAPI_DEVICE_SELECTOR"] = args.oneapi_device_selector

    if args.deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import cuda_malloc

    if "rocm" in cuda_malloc.get_torch_version_noimport():
        os.environ["OCL_SET_SVM_SIZE"] = "262144"


def cleanup_comfyui_runtime(unload_models: bool | None = None) -> None:
    """Best-effort cleanup for embedded or repeated generated-script execution."""
    import gc

    def run_cleanup_hook(name: str, should_run: bool = True) -> None:
        if not should_run or not hasattr(model_management, name):
            return
        cleanup_fn = getattr(model_management, name)
        try:
            cleanup_fn()
        except Exception as exc:
            warnings.warn(
                f"ComfyUI cleanup hook {name} failed during teardown: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    should_unload = unload_models
    if should_unload is None:
        should_unload = os.environ.get(
            "COMFYUI_TOPYTHON_UNLOAD_MODELS", ""
        ).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    try:
        import comfy.model_management as model_management
    except ModuleNotFoundError:
        gc.collect()
        return

    run_cleanup_hook("cleanup_models_gc")
    run_cleanup_hook("unload_all_models", should_run=should_unload)
    run_cleanup_hook("soft_empty_cache")
    gc.collect()


def import_custom_nodes() -> None:
    """Initialize ComfyUI custom nodes in the exporter runtime."""
    comfyui_path = get_comfyui_path()
    if comfyui_path and comfyui_path not in sys.path:
        sys.path.insert(0, comfyui_path)

    import asyncio
    import execution
    from nodes import init_extra_nodes

    if comfyui_path in sys.path:
        sys.path.remove(comfyui_path)
    sys.path.insert(0, comfyui_path)

    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    asyncio.run(init_extra_nodes())


# Workflow data
def build_workflow() -> dict[str, Any]:
    return {
        "52": {
            "inputs": {
                "video": "1新的消息 3.ts",
                "force_rate": 0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "start_time": 0,
                "format": "None",
            },
            "class_type": "VHS_LoadVideoFFmpeg",
            "_meta": {"title": "Load Video FFmpeg (Upload) 🎥🅥🅗🅢"},
        },
        "59": {
            "inputs": {
                "frame_rate": ["82", 0],
                "loop_count": 0,
                "filename_prefix": "rife",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
                "images": ["85:67", 0],
                "audio": ["52", 2],
            },
            "class_type": "VHS_VideoCombine",
            "_meta": {"title": "Video Combine 🎥🅥🅗🅢"},
        },
        "82": {
            "inputs": {"video_info": ["52", 3]},
            "class_type": "VHS_VideoInfoSource",
            "_meta": {"title": "Video Info (Source) 🎥🅥🅗🅢"},
        },
        "85:87": {
            "inputs": {
                "upscale_method": "bicubic",
                "scale_by": 0.5,
                "image": ["52", 0],
            },
            "class_type": "ImageScaleBy",
            "_meta": {"title": "Upscale Image By"},
        },
        "85:67": {
            "inputs": {
                "resize_type": "scale by multiplier",
                "resize_type.scale": 2,
                "quality": "ULTRA",
                "images": ["85:87", 0],
            },
            "class_type": "RTXVideoSuperResolution",
            "_meta": {"title": "RTX Video Super Resolution"},
        },
    }


def build_extra_pnginfo() -> dict[str, Any] | None:
    return {
        "workflow": {
            "id": "56fe86ee-efb1-4a41-b0e5-687bc7d424c3",
            "revision": 0,
            "last_node_id": 89,
            "last_link_id": 214,
            "nodes": [
                {
                    "id": 52,
                    "type": "VHS_LoadVideoFFmpeg",
                    "pos": [878.7633315557457, 674.241310571694],
                    "size": [313.45001220703125, 821.8001923544822],
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "inputs": [
                        {
                            "name": "meta_batch",
                            "shape": 7,
                            "type": "VHS_BatchManager",
                            "link": None,
                        },
                        {"name": "vae", "shape": 7, "type": "VAE", "link": None},
                    ],
                    "outputs": [
                        {"name": "IMAGE", "type": "IMAGE", "links": [206]},
                        {"name": "mask", "type": "MASK", "links": None},
                        {"name": "audio", "type": "AUDIO", "links": [193]},
                        {"name": "video_info", "type": "VHS_VIDEOINFO", "links": [166]},
                    ],
                    "properties": {"Node name for S&R": "VHS_LoadVideoFFmpeg"},
                    "widgets_values": {
                        "video": "1新的消息 3.ts",
                        "force_rate": 0,
                        "custom_width": 0,
                        "custom_height": 0,
                        "frame_load_cap": 0,
                        "start_time": 0,
                        "format": "None",
                        "videopreview": {
                            "hidden": False,
                            "paused": False,
                            "params": {
                                "filename": "1新的消息 3.ts",
                                "type": "input",
                                "format": "video/ts",
                                "force_rate": 0,
                                "custom_width": 0,
                                "custom_height": 0,
                                "frame_load_cap": 0,
                                "start_time": 0,
                            },
                        },
                    },
                },
                {
                    "id": 82,
                    "type": "VHS_VideoInfoSource",
                    "pos": [1283.291457198852, 1041.0504585322687],
                    "size": [247.6666717529297, 106],
                    "flags": {"collapsed": True},
                    "order": 2,
                    "mode": 0,
                    "inputs": [
                        {"name": "video_info", "type": "VHS_VIDEOINFO", "link": 166}
                    ],
                    "outputs": [
                        {"name": "fps🟨", "type": "FLOAT", "links": [167]},
                        {"name": "frame_count🟨", "type": "INT", "links": None},
                        {"name": "duration🟨", "type": "FLOAT", "links": None},
                        {"name": "width🟨", "type": "INT", "links": None},
                        {"name": "height🟨", "type": "INT", "links": None},
                    ],
                    "properties": {"Node name for S&R": "VHS_VideoInfoSource"},
                    "widgets_values": {},
                },
                {
                    "id": 88,
                    "type": "4a0d8b8f-cf71-4a61-bae8-2d01f9e0b861",
                    "pos": [1589.6089051736612, 1090.6019861743646],
                    "size": [372.87034702726555, 58],
                    "flags": {},
                    "order": 3,
                    "mode": 4,
                    "inputs": [{"name": "image", "type": "IMAGE", "link": 207}],
                    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [208]}],
                    "properties": {"previewExposures": []},
                },
                {
                    "id": 85,
                    "type": "448143d3-f043-44bd-aec4-76c552b7adb2",
                    "pos": [1281.1034133448654, 1089.8290292875938],
                    "size": [264.1636754609699, 58],
                    "flags": {},
                    "order": 1,
                    "mode": 0,
                    "inputs": [
                        {
                            "label": "images",
                            "name": "on_false",
                            "type": "IMAGE",
                            "link": 206,
                        }
                    ],
                    "outputs": [{"name": "output", "type": "IMAGE", "links": [207]}],
                    "properties": {"previewExposures": []},
                },
                {
                    "id": 59,
                    "type": "VHS_VideoCombine",
                    "pos": [2040.0655326361154, 688.7917959020084],
                    "size": [340.20001220703125, 916.3550216674804],
                    "flags": {},
                    "order": 4,
                    "mode": 0,
                    "inputs": [
                        {"name": "images", "type": "IMAGE", "link": 208},
                        {"name": "audio", "shape": 7, "type": "AUDIO", "link": 193},
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
                            "link": 167,
                        },
                    ],
                    "outputs": [
                        {"name": "Filenames", "type": "VHS_FILENAMES", "links": None}
                    ],
                    "properties": {"Node name for S&R": "VHS_VideoCombine"},
                    "widgets_values": {
                        "frame_rate": 8,
                        "loop_count": 0,
                        "filename_prefix": "rife",
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
                                "filename": "rife_00013-audio.mp4",
                                "subfolder": "",
                                "type": "output",
                                "format": "video/h264-mp4",
                                "frame_rate": 25,
                                "workflow": "rife_00013.png",
                                "fullpath": "/comfy/output/rife_00013-audio.mp4",
                            },
                        },
                    },
                },
            ],
            "links": [
                [166, 52, 3, 82, 0, "VHS_VIDEOINFO"],
                [167, 82, 0, 59, 4, "FLOAT"],
                [193, 52, 2, 59, 1, "AUDIO"],
                [206, 52, 0, 85, 0, "IMAGE"],
                [207, 85, 0, 88, 0, "IMAGE"],
                [208, 88, 0, 59, 0, "IMAGE"],
            ],
            "groups": [],
            "definitions": {
                "subgraphs": [
                    {
                        "id": "448143d3-f043-44bd-aec4-76c552b7adb2",
                        "version": 1,
                        "state": {
                            "lastGroupId": 0,
                            "lastNodeId": 89,
                            "lastLinkId": 214,
                            "lastRerouteId": 0,
                        },
                        "revision": 0,
                        "config": {},
                        "name": "RTX Video Super Resolution",
                        "inputNode": {
                            "id": -10,
                            "bounding": [
                                1110.9704775686666,
                                927.5630088088056,
                                128,
                                68,
                            ],
                        },
                        "outputNode": {
                            "id": -20,
                            "bounding": [
                                2418.7027641345703,
                                927.5630088088056,
                                128,
                                68,
                            ],
                        },
                        "inputs": [
                            {
                                "id": "59aaac50-eccf-409c-a72c-61c15f3c3abf",
                                "name": "on_false",
                                "type": "IMAGE",
                                "linkIds": [204, 212],
                                "localized_name": "on_false",
                                "label": "images",
                                "pos": [1214.9704775686666, 951.5630088088056],
                            }
                        ],
                        "outputs": [
                            {
                                "id": "36800efd-2963-41c1-aff4-ee5f0a6c0115",
                                "name": "output",
                                "type": "IMAGE",
                                "linkIds": [214],
                                "localized_name": "output",
                                "pos": [2442.7027641345703, 951.5630088088056],
                            }
                        ],
                        "widgets": [],
                        "nodes": [
                            {
                                "id": 87,
                                "type": "ImageScaleBy",
                                "pos": [1311.0561508266846, 1099.4466397969438],
                                "size": [320.51931147879463, 82],
                                "flags": {},
                                "order": 1,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "image",
                                        "name": "image",
                                        "type": "IMAGE",
                                        "link": 212,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "IMAGE",
                                        "name": "IMAGE",
                                        "type": "IMAGE",
                                        "links": [195, 213],
                                    }
                                ],
                                "properties": {"Node name for S&R": "ImageScaleBy"},
                                "widgets_values": ["bicubic", 0.5],
                            },
                            {
                                "id": 67,
                                "type": "RTXVideoSuperResolution",
                                "pos": [1314.2569804314016, 934.0404028400236],
                                "size": [320.4536357444359, 106],
                                "flags": {},
                                "order": 0,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "images",
                                        "name": "images",
                                        "type": "IMAGE",
                                        "link": 213,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "upscaled_images",
                                        "name": "upscaled_images",
                                        "type": "IMAGE",
                                        "links": [202, 214],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "RTXVideoSuperResolution"
                                },
                                "widgets_values": [
                                    "scale by multiplier",
                                    2,
                                    "ULTRA",
                                ],
                            },
                            {
                                "id": 89,
                                "type": "RAMCleanup",
                                "pos": [1316.7406021970835, 741.9139606198975],
                                "size": [270, 130],
                                "flags": {},
                                "order": 2,
                                "mode": 4,
                                "inputs": [
                                    {
                                        "localized_name": "anything",
                                        "name": "anything",
                                        "shape": 7,
                                        "type": "*",
                                        "link": 204,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "output",
                                        "name": "output",
                                        "type": "*",
                                        "links": [],
                                    }
                                ],
                                "properties": {"Node name for S&R": "RAMCleanup"},
                                "widgets_values": [True, True, True, 3],
                            },
                        ],
                        "groups": [],
                        "links": [
                            {
                                "id": 204,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 89,
                                "target_slot": 0,
                                "type": "*",
                            },
                            {
                                "id": 212,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 87,
                                "target_slot": 0,
                                "type": "IMAGE",
                            },
                            {
                                "id": 213,
                                "origin_id": 87,
                                "origin_slot": 0,
                                "target_id": 67,
                                "target_slot": 0,
                                "type": "IMAGE",
                            },
                            {
                                "id": 214,
                                "origin_id": 67,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 0,
                                "type": "IMAGE",
                            },
                        ],
                        "extra": {},
                    },
                    {
                        "id": "4a0d8b8f-cf71-4a61-bae8-2d01f9e0b861",
                        "version": 1,
                        "state": {
                            "lastGroupId": 0,
                            "lastNodeId": 89,
                            "lastLinkId": 214,
                            "lastRerouteId": 0,
                        },
                        "revision": 0,
                        "config": {},
                        "name": "GAN 2x",
                        "inputNode": {
                            "id": -10,
                            "bounding": [
                                1114.5139687043998,
                                1057.9474374059664,
                                128,
                                88,
                            ],
                        },
                        "outputNode": {
                            "id": -20,
                            "bounding": [
                                1692.3172989775483,
                                1057.9474374059664,
                                128,
                                68,
                            ],
                        },
                        "inputs": [
                            {
                                "id": "b328805d-5333-46e5-a465-38c90ad31692",
                                "name": "image",
                                "type": "IMAGE",
                                "linkIds": [209],
                                "localized_name": "image",
                                "pos": [1218.5139687043998, 1081.9474374059664],
                            },
                            {
                                "id": "9b800c1e-2e23-414a-aaa7-2c902cc182f7",
                                "name": "model_name",
                                "type": "COMBO",
                                "linkIds": [199],
                                "pos": [1218.5139687043998, 1101.9474374059664],
                            },
                        ],
                        "outputs": [
                            {
                                "id": "c35d5fa2-6cf2-484b-8c18-df0b2dd93a79",
                                "name": "IMAGE",
                                "type": "IMAGE",
                                "linkIds": [211],
                                "localized_name": "IMAGE",
                                "pos": [1716.3172989775483, 1081.9474374059664],
                            }
                        ],
                        "widgets": [],
                        "nodes": [
                            {
                                "id": 64,
                                "type": "UpscaleModelLoader",
                                "pos": [1311.3393674923457, 1196.9098832980653],
                                "size": [320.9779314852026, 65.71785090491198],
                                "flags": {},
                                "order": 1,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "model_name",
                                        "name": "model_name",
                                        "type": "COMBO",
                                        "widget": {"name": "model_name"},
                                        "link": 199,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "UPSCALE_MODEL",
                                        "name": "UPSCALE_MODEL",
                                        "type": "UPSCALE_MODEL",
                                        "links": [130],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "UpscaleModelLoader",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.10.0",
                                    "models": [
                                        {
                                            "name": "RealESRGAN_x4plus.safetensors",
                                            "url": "https://huggingface.co/Comfy-Org/Real-ESRGAN_repackaged/resolve/main/RealESRGAN_x4plus.safetensors",
                                            "directory": "upscale_models",
                                        }
                                    ],
                                },
                                "widgets_values": ["RealESRGAN_x2plus.pth"],
                            },
                            {
                                "id": 61,
                                "type": "ImageUpscaleWithModel",
                                "pos": [1308.8179451388949, 1088.760364981686],
                                "size": [320, 46],
                                "flags": {},
                                "order": 0,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "upscale_model",
                                        "name": "upscale_model",
                                        "type": "UPSCALE_MODEL",
                                        "link": 130,
                                    },
                                    {
                                        "localized_name": "image",
                                        "name": "image",
                                        "type": "IMAGE",
                                        "link": 209,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "IMAGE",
                                        "name": "IMAGE",
                                        "type": "IMAGE",
                                        "links": [210],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "ImageUpscaleWithModel",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.10.0",
                                },
                                "widgets_values": [],
                            },
                            {
                                "id": 68,
                                "type": "ImageScaleBy",
                                "pos": [1302.5139687043998, 951.2671406089553],
                                "size": [320.51931147879463, 82],
                                "flags": {},
                                "order": 2,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "image",
                                        "name": "image",
                                        "type": "IMAGE",
                                        "link": 210,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "IMAGE",
                                        "name": "IMAGE",
                                        "type": "IMAGE",
                                        "links": [178, 211],
                                    }
                                ],
                                "properties": {"Node name for S&R": "ImageScaleBy"},
                                "widgets_values": ["bicubic", 0.5],
                            },
                        ],
                        "groups": [],
                        "links": [
                            {
                                "id": 130,
                                "origin_id": 64,
                                "origin_slot": 0,
                                "target_id": 61,
                                "target_slot": 0,
                                "type": "UPSCALE_MODEL",
                            },
                            {
                                "id": 199,
                                "origin_id": -10,
                                "origin_slot": 1,
                                "target_id": 64,
                                "target_slot": 0,
                                "type": "COMBO",
                            },
                            {
                                "id": 209,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 61,
                                "target_slot": 1,
                                "type": "IMAGE",
                            },
                            {
                                "id": 210,
                                "origin_id": 61,
                                "origin_slot": 0,
                                "target_id": 68,
                                "target_slot": 0,
                                "type": "IMAGE",
                            },
                            {
                                "id": 211,
                                "origin_id": 68,
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
                    "scale": 0.9532315978713161,
                    "offset": [-976.9540229214226, -518.6137238007319],
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


def generate(
    video_path: str,
    unload_models: bool | None = None,
):
    bootstrap_comfyui_runtime()
    add_extra_model_paths()
    import_custom_nodes()

    from nodes import NODE_CLASS_MAPPINGS, ImageScaleBy
    import folder_paths
    import os
    import glob as glob_module
    import torch

    try:
        with torch.inference_mode():
            vhs_loadvideoffmpeg = NODE_CLASS_MAPPINGS["VHS_LoadVideoFFmpeg"]()
            vhs_loadvideoffmpeg_52 = vhs_loadvideoffmpeg.load_video(
                video=video_path,
                force_rate=0,
                custom_width=0,
                custom_height=0,
                frame_load_cap=0,
                start_time=0,
                format="None",
                unique_id=6739176214643946654,
            )
            vhs_videoinfosource = NODE_CLASS_MAPPINGS["VHS_VideoInfoSource"]()
            imagescaleby = ImageScaleBy()
            rtxvideosuperresolution = NODE_CLASS_MAPPINGS["RTXVideoSuperResolution"]()
            vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

            for q in range(1):
                vhs_videoinfosource_82 = vhs_videoinfosource.get_video_info(
                    video_info=get_value_at_index(vhs_loadvideoffmpeg_52, 3)
                )
                imagescaleby_85_87 = imagescaleby.upscale(
                    upscale_method="bicubic",
                    scale_by=0.5,
                    image=get_value_at_index(vhs_loadvideoffmpeg_52, 0),
                )
                rtxvideosuperresolution_85_67 = (
                    rtxvideosuperresolution.EXECUTE_NORMALIZED(
                        resize_type="scale by multiplier",
                        **{"resize_type.scale": 2},
                        quality="ULTRA",
                        images=get_value_at_index(imagescaleby_85_87, 0),
                    )
                )
                vhs_videocombine_59 = vhs_videocombine.combine_video(
                    frame_rate=get_value_at_index(vhs_videoinfosource_82, 0),
                    loop_count=0,
                    filename_prefix="rtx_vsr",
                    format="video/h264-mp4",
                    pix_fmt="yuv420p",
                    crf=19,
                    save_metadata=True,
                    trim_to_audio=False,
                    pingpong=False,
                    save_output=True,
                    images=get_value_at_index(rtxvideosuperresolution_85_67, 0),
                    audio=get_value_at_index(vhs_loadvideoffmpeg_52, 2),
                    unique_id=2276563143481976898,
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                )

                output_dir = folder_paths.get_output_directory()
                pattern = os.path.join(output_dir, "rtx_vsr_*.mp4")
                files = sorted(glob_module.glob(pattern), key=os.path.getmtime)
                if files:
                    return files[-1]
                return None
    finally:
        cleanup_comfyui_runtime(unload_models=unload_models)


# Workflow execution
def main(unload_models: bool | None = None):
    bootstrap_comfyui_runtime()
    add_extra_model_paths()
    import_custom_nodes()

    # Node imports
    from nodes import ImageScaleBy, NODE_CLASS_MAPPINGS

    import torch

    try:
        with torch.inference_mode():
            vhs_loadvideoffmpeg = NODE_CLASS_MAPPINGS["VHS_LoadVideoFFmpeg"]()
            vhs_loadvideoffmpeg_52 = vhs_loadvideoffmpeg.load_video(
                video="1\u65b0\u7684\u6d88\u606f 3.ts",
                force_rate=0,
                custom_width=0,
                custom_height=0,
                frame_load_cap=0,
                start_time=0,
                format="None",
                unique_id=6739176214643946654,
            )
            vhs_videoinfosource = NODE_CLASS_MAPPINGS["VHS_VideoInfoSource"]()
            imagescaleby = ImageScaleBy()
            rtxvideosuperresolution = NODE_CLASS_MAPPINGS["RTXVideoSuperResolution"]()
            vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
            for q in range(1):
                vhs_videoinfosource_82 = vhs_videoinfosource.get_video_info(
                    video_info=get_value_at_index(vhs_loadvideoffmpeg_52, 3)
                )
                imagescaleby_85_87 = imagescaleby.upscale(
                    upscale_method="bicubic",
                    scale_by=0.5,
                    image=get_value_at_index(vhs_loadvideoffmpeg_52, 0),
                )
                rtxvideosuperresolution_85_67 = (
                    rtxvideosuperresolution.EXECUTE_NORMALIZED(
                        resize_type="scale by multiplier",
                        **{"resize_type.scale": 2},
                        quality="ULTRA",
                        images=get_value_at_index(imagescaleby_85_87, 0),
                    )
                )
                vhs_videocombine_59 = vhs_videocombine.combine_video(
                    frame_rate=get_value_at_index(vhs_videoinfosource_82, 0),
                    loop_count=0,
                    filename_prefix="rife",
                    format="video/h264-mp4",
                    pix_fmt="yuv420p",
                    crf=19,
                    save_metadata=True,
                    trim_to_audio=False,
                    pingpong=False,
                    save_output=True,
                    images=get_value_at_index(rtxvideosuperresolution_85_67, 0),
                    audio=get_value_at_index(vhs_loadvideoffmpeg_52, 2),
                    unique_id=2276563143481976898,
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                )
    finally:
        cleanup_comfyui_runtime(unload_models=unload_models)


# Entrypoint
if __name__ == "__main__":
    main()
