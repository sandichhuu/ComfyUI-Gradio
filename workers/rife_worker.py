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
        "6": {
            "inputs": {
                "frame_rate": ["53:51:43", 0],
                "loop_count": 0,
                "filename_prefix": "rife",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": False,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
                "images": ["12", 0],
                "audio": ["9", 2],
            },
            "class_type": "VHS_VideoCombine",
            "_meta": {"title": "Video Combine 🎥🅥🅗🅢"},
        },
        "9": {
            "inputs": {
                "video": "rife_00013.mp4",
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
        "12": {
            "inputs": {
                "clear_cache_after_n_frames": 100,
                "multiplier": ["53:51:48", 0],
                "keep_model_loaded": False,
                "frames": ["37", 0],
                "rife_trt_model": ["13", 0],
            },
            "class_type": "AutoRifeTensorrt",
            "_meta": {"title": "Auto RIFE TensorRT"},
        },
        "13": {
            "inputs": {
                "model": "rife49_ensemble_True_scale_1_sim",
                "precision": "fp32",
                "resolution_profile": "medium",
            },
            "class_type": "AutoLoadRifeTensorrtModel",
            "_meta": {"title": "(Down)load RIFE TensorRT Model"},
        },
        "37": {
            "inputs": {
                "width": ["53:22:31", 0],
                "height": ["53:22:32", 0],
                "upscale_method": "nvidia_rtx_vsr",
                "keep_proportion": "crop",
                "pad_color": "0, 0, 0",
                "crop_position": "center",
                "divisible_by": 2,
                "device": "gpu",
                "image": ["9", 0],
            },
            "class_type": "ImageResizeKJv2",
            "_meta": {"title": "Resize Image v2"},
        },
        "53:52": {
            "inputs": {"video_info": ["9", 3]},
            "class_type": "VHS_VideoInfoSource",
            "_meta": {"title": "Video Info (Source) 🎥🅥🅗🅢"},
        },
        "53:22:17": {
            "inputs": {
                "expression": "a < b",
                "values.a": ["53:22:28", 0],
                "values.b": ["53:22:21", 0],
            },
            "class_type": "ComfyMathExpression",
            "_meta": {"title": "a < 672"},
        },
        "53:22:31": {
            "inputs": {
                "switch": ["53:22:26", 2],
                "on_false": ["53:22:25", 0],
                "on_true": ["53:22:18", 0],
            },
            "class_type": "ComfySwitchNode",
            "_meta": {"title": "LOW"},
        },
        "53:22:28": {
            "inputs": {
                "switch": ["53:22:26", 2],
                "on_false": ["53:52", 4],
                "on_true": ["53:52", 3],
            },
            "class_type": "ComfySwitchNode",
            "_meta": {"title": "LOW"},
        },
        "53:22:26": {
            "inputs": {
                "expression": "a < b",
                "values.a": ["53:52", 3],
                "values.b": ["53:52", 4],
            },
            "class_type": "ComfyMathExpression",
            "_meta": {"title": "Math Expression"},
        },
        "53:22:30": {
            "inputs": {
                "switch": ["53:22:26", 2],
                "on_false": ["53:52", 3],
                "on_true": ["53:52", 4],
            },
            "class_type": "ComfySwitchNode",
            "_meta": {"title": "HIGHT"},
        },
        "53:22:32": {
            "inputs": {
                "switch": ["53:22:26", 2],
                "on_false": ["53:22:18", 0],
                "on_true": ["53:22:25", 0],
            },
            "class_type": "ComfySwitchNode",
            "_meta": {"title": "HIGHT"},
        },
        "53:22:18": {
            "inputs": {
                "switch": ["53:22:17", 2],
                "on_false": ["53:22:28", 0],
                "on_true": ["53:22:21", 0],
            },
            "class_type": "ComfySwitchNode",
            "_meta": {"title": "LOW"},
        },
        "53:22:25": {
            "inputs": {
                "switch": ["53:22:17", 2],
                "on_false": ["53:22:30", 0],
                "on_true": ["53:22:24", 1],
            },
            "class_type": "ComfySwitchNode",
            "_meta": {"title": "HIGHT"},
        },
        "53:22:24": {
            "inputs": {
                "expression": "a * b",
                "values.a": ["53:22:23", 0],
                "values.b": ["53:22:30", 0],
            },
            "class_type": "ComfyMathExpression",
            "_meta": {"title": "Math Expression"},
        },
        "53:22:21": {
            "inputs": {"value": 672},
            "class_type": "PrimitiveInt",
            "_meta": {"title": "Int"},
        },
        "53:22:23": {
            "inputs": {
                "expression": "a * 1.0 / b",
                "values.a": ["53:22:21", 0],
                "values.b": ["53:22:28", 0],
            },
            "class_type": "ComfyMathExpression",
            "_meta": {"title": "672 / b"},
        },
        "53:51:49": {
            "inputs": {
                "expression": "a > b",
                "values.a": ["53:52", 0],
                "values.b": ["53:51:42", 0],
            },
            "class_type": "ComfyMathExpression",
            "_meta": {"title": "Math Expression"},
        },
        "53:51:50": {
            "inputs": {"value": 1},
            "class_type": "PrimitiveInt",
            "_meta": {"title": "Int"},
        },
        "53:51:48": {
            "inputs": {
                "switch": ["53:51:49", 2],
                "on_false": ["53:51:47", 1],
                "on_true": ["53:51:50", 0],
            },
            "class_type": "ComfySwitchNode",
            "_meta": {"title": "Switch"},
        },
        "53:51:42": {
            "inputs": {"value": 60},
            "class_type": "PrimitiveInt",
            "_meta": {"title": "Int"},
        },
        "53:51:43": {
            "inputs": {
                "expression": "a * b",
                "values.a": ["53:51:48", 0],
                "values.b": ["53:52", 0],
            },
            "class_type": "ComfyMathExpression",
            "_meta": {"title": "FPS Target"},
        },
        "53:51:47": {
            "inputs": {
                "expression": "a * 1.0 / b",
                "values.a": ["53:51:42", 0],
                "values.b": ["53:52", 0],
            },
            "class_type": "ComfyMathExpression",
            "_meta": {"title": "Math Expression"},
        },
    }


def build_extra_pnginfo() -> dict[str, Any] | None:
    return {
        "workflow": {
            "id": "089b4f11-62b3-4100-a193-a2d8b5ca1b26",
            "revision": 0,
            "last_node_id": 53,
            "last_link_id": 130,
            "nodes": [
                {
                    "id": 6,
                    "type": "VHS_VideoCombine",
                    "pos": [1630.3414152877228, 722.872461765029],
                    "size": [340.20001220703125, 915.9738311767578],
                    "flags": {},
                    "order": 5,
                    "mode": 0,
                    "inputs": [
                        {"name": "images", "type": "IMAGE", "link": 22},
                        {"name": "audio", "shape": 7, "type": "AUDIO", "link": 14},
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
                            "link": 121,
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
                        "save_metadata": False,
                        "trim_to_audio": False,
                        "pingpong": False,
                        "save_output": True,
                        "videopreview": {
                            "hidden": False,
                            "paused": False,
                            "params": {
                                "filename": "rife_00014.mp4",
                                "subfolder": "",
                                "type": "output",
                                "format": "video/h264-mp4",
                                "frame_rate": 50,
                                "workflow": "rife_00014.png",
                                "fullpath": "/comfy/output/rife_00014.mp4",
                            },
                        },
                    },
                },
                {
                    "id": 12,
                    "type": "AutoRifeTensorrt",
                    "pos": [1169.3725903748923, 732.8621146418981],
                    "size": [404.79998779296875, 216],
                    "flags": {},
                    "order": 4,
                    "mode": 0,
                    "inputs": [
                        {"name": "frames", "type": "IMAGE", "link": 93},
                        {
                            "name": "rife_trt_model",
                            "type": "RIFE_TRT_MODEL",
                            "link": 21,
                        },
                        {
                            "name": "multiplier",
                            "type": "INT",
                            "widget": {"name": "multiplier"},
                            "link": 127,
                        },
                    ],
                    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [22]}],
                    "properties": {"Node name for S&R": "AutoRifeTensorrt"},
                    "widgets_values": [100, 2, False],
                },
                {
                    "id": 13,
                    "type": "AutoLoadRifeTensorrtModel",
                    "pos": [-237.83329050333742, 825.0176400789862],
                    "size": [406.7166748046875, 168],
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "inputs": [
                        {
                            "name": "custom_config",
                            "shape": 7,
                            "type": "RIFE_RESOLUTION_CONFIG",
                            "link": None,
                        }
                    ],
                    "outputs": [
                        {
                            "name": "rife_trt_model",
                            "type": "RIFE_TRT_MODEL",
                            "links": [21],
                        }
                    ],
                    "properties": {"Node name for S&R": "AutoLoadRifeTensorrtModel"},
                    "widgets_values": [
                        "rife49_ensemble_True_scale_1_sim",
                        "fp32",
                        "medium",
                    ],
                },
                {
                    "id": 37,
                    "type": "ImageResizeKJv2",
                    "pos": [803.2119520603674, 826.9511377305934],
                    "size": [332.26666259765625, 468],
                    "flags": {},
                    "order": 3,
                    "mode": 0,
                    "inputs": [
                        {"name": "image", "type": "IMAGE", "link": 84},
                        {"name": "mask", "shape": 7, "type": "MASK", "link": None},
                        {
                            "name": "width",
                            "type": "INT",
                            "widget": {"name": "width"},
                            "link": 128,
                        },
                        {
                            "name": "height",
                            "type": "INT",
                            "widget": {"name": "height"},
                            "link": 129,
                        },
                    ],
                    "outputs": [
                        {"name": "IMAGE", "type": "IMAGE", "links": [93]},
                        {"name": "width", "type": "INT", "links": None},
                        {"name": "height", "type": "INT", "links": None},
                        {"name": "mask", "type": "MASK", "links": None},
                    ],
                    "properties": {"Node name for S&R": "ImageResizeKJv2"},
                    "widgets_values": [
                        512,
                        512,
                        "nvidia_rtx_vsr",
                        "crop",
                        "0, 0, 0",
                        "center",
                        2,
                        "gpu",
                    ],
                },
                {
                    "id": 53,
                    "type": "0b3e9d59-10df-4754-b7af-5c7d5bbc08a0",
                    "pos": [531.0396802816622, 863.14565865012],
                    "size": [233.5141444830257, 118],
                    "flags": {},
                    "order": 2,
                    "mode": 0,
                    "inputs": [
                        {"name": "video_info", "type": "VHS_VIDEOINFO", "link": 119},
                        {
                            "label": "target_fps",
                            "name": "value",
                            "type": "INT",
                            "widget": {"name": "value"},
                            "link": None,
                        },
                    ],
                    "outputs": [
                        {
                            "label": "actual_fps",
                            "name": "FLOAT",
                            "type": "FLOAT",
                            "links": [121],
                        },
                        {
                            "label": "multiplier",
                            "name": "output",
                            "type": "INT",
                            "links": [127],
                        },
                        {
                            "label": "width",
                            "name": "output_1",
                            "type": "INT",
                            "links": [128],
                        },
                        {
                            "label": "height",
                            "name": "INT",
                            "type": "INT",
                            "links": [129],
                        },
                    ],
                    "properties": {"previewExposures": []},
                },
                {
                    "id": 9,
                    "type": "VHS_LoadVideoFFmpeg",
                    "pos": [197.63780689611391, 704.8261579871867],
                    "size": [313.45001220703125, 820.7986564766425],
                    "flags": {},
                    "order": 1,
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
                        {"name": "IMAGE", "type": "IMAGE", "links": [84]},
                        {"name": "mask", "type": "MASK", "links": None},
                        {"name": "audio", "type": "AUDIO", "links": [14]},
                        {"name": "video_info", "type": "VHS_VIDEOINFO", "links": [119]},
                    ],
                    "properties": {"Node name for S&R": "VHS_LoadVideoFFmpeg"},
                    "widgets_values": {
                        "video": "rife_00013.mp4",
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
                                "filename": "rife_00013.mp4",
                                "type": "input",
                                "format": "video/mp4",
                                "force_rate": 0,
                                "custom_width": 0,
                                "custom_height": 0,
                                "frame_load_cap": 0,
                                "start_time": 0,
                            },
                        },
                    },
                },
            ],
            "links": [
                [14, 9, 2, 6, 1, "AUDIO"],
                [21, 13, 0, 12, 1, "RIFE_TRT_MODEL"],
                [22, 12, 0, 6, 0, "IMAGE"],
                [84, 9, 0, 37, 0, "IMAGE"],
                [93, 37, 0, 12, 0, "IMAGE"],
                [119, 9, 3, 53, 0, "VHS_VIDEOINFO"],
                [121, 53, 0, 6, 4, "FLOAT"],
                [127, 53, 1, 12, 2, "INT"],
                [128, 53, 2, 37, 2, "INT"],
                [129, 53, 3, 37, 3, "INT"],
            ],
            "groups": [],
            "definitions": {
                "subgraphs": [
                    {
                        "id": "80bfb28a-9007-4908-9447-28fab825e3e1",
                        "version": 1,
                        "state": {
                            "lastGroupId": 0,
                            "lastNodeId": 53,
                            "lastLinkId": 130,
                            "lastRerouteId": 0,
                        },
                        "revision": 0,
                        "config": {},
                        "name": "Validate Width|Height",
                        "inputNode": {
                            "id": -10,
                            "bounding": [
                                205.54275088547922,
                                1415.0026213507413,
                                128,
                                88,
                            ],
                        },
                        "outputNode": {
                            "id": -20,
                            "bounding": [
                                1844.5509934634977,
                                1436.4377607963804,
                                128,
                                88,
                            ],
                        },
                        "inputs": [
                            {
                                "id": "9a9347ae-f644-48d7-8cbc-d3b7cbaa6eda",
                                "name": "values.a",
                                "type": "FLOAT,INT,BOOLEAN",
                                "linkIds": [47, 54, 65],
                                "localized_name": "values.a",
                                "label": "width",
                                "pos": [309.5427508854792, 1439.0026213507413],
                            },
                            {
                                "id": "7ff3a42a-9a05-49f4-bb5d-54b60fef3462",
                                "name": "values.b",
                                "type": "FLOAT,INT,BOOLEAN",
                                "linkIds": [48, 55, 64],
                                "localized_name": "values.b",
                                "label": "height",
                                "shape": 7,
                                "pos": [309.5427508854792, 1459.0026213507413],
                            },
                        ],
                        "outputs": [
                            {
                                "id": "0f7fc20f-4dc0-4364-8072-1861d505b556",
                                "name": "output",
                                "type": "INT",
                                "linkIds": [76],
                                "label": "width",
                                "pos": [1868.5509934634977, 1460.4377607963804],
                            },
                            {
                                "id": "2fa13026-17e8-4dbe-a693-16440ece2616",
                                "name": "INT",
                                "type": "INT",
                                "linkIds": [75],
                                "label": "height",
                                "pos": [1868.5509934634977, 1480.4377607963804],
                            },
                        ],
                        "widgets": [],
                        "nodes": [
                            {
                                "id": 17,
                                "type": "ComfyMathExpression",
                                "pos": [744.4533580578341, 1267.1896454457763],
                                "size": [225, 164],
                                "flags": {},
                                "order": 1,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "a",
                                        "localized_name": "values.a",
                                        "name": "values.a",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 56,
                                    },
                                    {
                                        "label": "b",
                                        "localized_name": "values.b",
                                        "name": "values.b",
                                        "shape": 7,
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 57,
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
                                        "links": [],
                                    },
                                    {
                                        "localized_name": "BOOL",
                                        "name": "BOOL",
                                        "type": "BOOLEAN",
                                        "links": [27, 42],
                                    },
                                ],
                                "title": "a < 672",
                                "properties": {
                                    "Node name for S&R": "ComfyMathExpression"
                                },
                                "widgets_values": ["a < b"],
                            },
                            {
                                "id": 31,
                                "type": "ComfySwitchNode",
                                "pos": [1441.8893513685884, 1221.157402384477],
                                "size": [270, 124],
                                "flags": {},
                                "order": 9,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "on_false",
                                        "name": "on_false",
                                        "type": "INT",
                                        "link": 72,
                                    },
                                    {
                                        "localized_name": "on_true",
                                        "name": "on_true",
                                        "type": "INT",
                                        "link": 71,
                                    },
                                    {
                                        "localized_name": "switch",
                                        "name": "switch",
                                        "type": "BOOLEAN",
                                        "widget": {"name": "switch"},
                                        "link": 69,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "output",
                                        "name": "output",
                                        "type": "INT",
                                        "links": [76],
                                    }
                                ],
                                "title": "LOW",
                                "properties": {"Node name for S&R": "ComfySwitchNode"},
                                "widgets_values": [False],
                            },
                            {
                                "id": 28,
                                "type": "ComfySwitchNode",
                                "pos": [426.21116344678194, 1239.084634546961],
                                "size": [270, 124],
                                "flags": {},
                                "order": 7,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "on_false",
                                        "name": "on_false",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 55,
                                    },
                                    {
                                        "localized_name": "on_true",
                                        "name": "on_true",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 54,
                                    },
                                    {
                                        "localized_name": "switch",
                                        "name": "switch",
                                        "type": "BOOLEAN",
                                        "widget": {"name": "switch"},
                                        "link": 53,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "output",
                                        "name": "output",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "links": [56, 62, 67],
                                    }
                                ],
                                "title": "LOW",
                                "properties": {"Node name for S&R": "ComfySwitchNode"},
                                "widgets_values": [False],
                            },
                            {
                                "id": 26,
                                "type": "ComfyMathExpression",
                                "pos": [427.3534563527143, 1370.5618980229406],
                                "size": [225, 164],
                                "flags": {},
                                "order": 6,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "a",
                                        "localized_name": "values.a",
                                        "name": "values.a",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 47,
                                    },
                                    {
                                        "label": "b",
                                        "localized_name": "values.b",
                                        "name": "values.b",
                                        "shape": 7,
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 48,
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
                                        "links": [],
                                    },
                                    {
                                        "localized_name": "BOOL",
                                        "name": "BOOL",
                                        "type": "BOOLEAN",
                                        "links": [53, 63, 69, 70],
                                    },
                                ],
                                "properties": {
                                    "Node name for S&R": "ComfyMathExpression"
                                },
                                "widgets_values": ["a < b"],
                            },
                            {
                                "id": 30,
                                "type": "ComfySwitchNode",
                                "pos": [428.5187412482067, 1576.214994000832],
                                "size": [270, 124],
                                "flags": {},
                                "order": 8,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "on_false",
                                        "name": "on_false",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 65,
                                    },
                                    {
                                        "localized_name": "on_true",
                                        "name": "on_true",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 64,
                                    },
                                    {
                                        "localized_name": "switch",
                                        "name": "switch",
                                        "type": "BOOLEAN",
                                        "widget": {"name": "switch"},
                                        "link": 63,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "output",
                                        "name": "output",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "links": [66, 68],
                                    }
                                ],
                                "title": "HIGHT",
                                "properties": {"Node name for S&R": "ComfySwitchNode"},
                                "widgets_values": [False],
                            },
                            {
                                "id": 32,
                                "type": "ComfySwitchNode",
                                "pos": [1440.4596807484634, 1534.5344346440534],
                                "size": [270, 124],
                                "flags": {},
                                "order": 10,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "on_false",
                                        "name": "on_false",
                                        "type": "INT",
                                        "link": 74,
                                    },
                                    {
                                        "localized_name": "on_true",
                                        "name": "on_true",
                                        "type": "INT",
                                        "link": 73,
                                    },
                                    {
                                        "localized_name": "switch",
                                        "name": "switch",
                                        "type": "BOOLEAN",
                                        "widget": {"name": "switch"},
                                        "link": 70,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "output",
                                        "name": "output",
                                        "type": "INT",
                                        "links": [75],
                                    }
                                ],
                                "title": "HIGHT",
                                "properties": {"Node name for S&R": "ComfySwitchNode"},
                                "widgets_values": [False],
                            },
                            {
                                "id": 18,
                                "type": "ComfySwitchNode",
                                "pos": [1025.7098562266765, 1312.9433825603244],
                                "size": [270, 124],
                                "flags": {},
                                "order": 2,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "on_false",
                                        "name": "on_false",
                                        "type": "INT",
                                        "link": 67,
                                    },
                                    {
                                        "localized_name": "on_true",
                                        "name": "on_true",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 32,
                                    },
                                    {
                                        "localized_name": "switch",
                                        "name": "switch",
                                        "type": "BOOLEAN",
                                        "widget": {"name": "switch"},
                                        "link": 27,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "output",
                                        "name": "output",
                                        "type": "INT",
                                        "links": [71, 74],
                                    }
                                ],
                                "title": "LOW",
                                "properties": {"Node name for S&R": "ComfySwitchNode"},
                                "widgets_values": [False],
                            },
                            {
                                "id": 25,
                                "type": "ComfySwitchNode",
                                "pos": [1027.8882778824345, 1446.9059250768532],
                                "size": [270, 124],
                                "flags": {},
                                "order": 5,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "on_false",
                                        "name": "on_false",
                                        "type": "INT",
                                        "link": 68,
                                    },
                                    {
                                        "localized_name": "on_true",
                                        "name": "on_true",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 43,
                                    },
                                    {
                                        "localized_name": "switch",
                                        "name": "switch",
                                        "type": "BOOLEAN",
                                        "widget": {"name": "switch"},
                                        "link": 42,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "output",
                                        "name": "output",
                                        "type": "INT",
                                        "links": [72, 73],
                                    }
                                ],
                                "title": "HIGHT",
                                "properties": {"Node name for S&R": "ComfySwitchNode"},
                                "widgets_values": [False],
                            },
                            {
                                "id": 24,
                                "type": "ComfyMathExpression",
                                "pos": [742.9614571111586, 1652.6005439985836],
                                "size": [225, 164],
                                "flags": {},
                                "order": 4,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "a",
                                        "localized_name": "values.a",
                                        "name": "values.a",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 83,
                                    },
                                    {
                                        "label": "b",
                                        "localized_name": "values.b",
                                        "name": "values.b",
                                        "shape": 7,
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 66,
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
                                        "links": [],
                                    },
                                    {
                                        "localized_name": "INT",
                                        "name": "INT",
                                        "type": "INT",
                                        "links": [43],
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
                                "widgets_values": ["a * b"],
                            },
                            {
                                "id": 21,
                                "type": "PrimitiveInt",
                                "pos": [746.1297170678765, 1142.2565107663356],
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
                                        "links": [32, 57, 61],
                                    }
                                ],
                                "properties": {"Node name for S&R": "PrimitiveInt"},
                                "widgets_values": [672, "fixed"],
                            },
                            {
                                "id": 23,
                                "type": "ComfyMathExpression",
                                "pos": [744.9035116263547, 1466.2409927204912],
                                "size": [225, 164],
                                "flags": {},
                                "order": 3,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "a",
                                        "localized_name": "values.a",
                                        "name": "values.a",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 61,
                                    },
                                    {
                                        "label": "b",
                                        "localized_name": "values.b",
                                        "name": "values.b",
                                        "shape": 7,
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 62,
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
                                        "links": [83],
                                    },
                                    {
                                        "localized_name": "INT",
                                        "name": "INT",
                                        "type": "INT",
                                        "links": [],
                                    },
                                    {
                                        "localized_name": "BOOL",
                                        "name": "BOOL",
                                        "type": "BOOLEAN",
                                        "links": None,
                                    },
                                ],
                                "title": "672 / b",
                                "properties": {
                                    "Node name for S&R": "ComfyMathExpression"
                                },
                                "widgets_values": ["a * 1.0 / b"],
                            },
                        ],
                        "groups": [],
                        "links": [
                            {
                                "id": 32,
                                "origin_id": 21,
                                "origin_slot": 0,
                                "target_id": 18,
                                "target_slot": 1,
                                "type": "INT",
                            },
                            {
                                "id": 27,
                                "origin_id": 17,
                                "origin_slot": 2,
                                "target_id": 18,
                                "target_slot": 2,
                                "type": "BOOLEAN",
                            },
                            {
                                "id": 42,
                                "origin_id": 17,
                                "origin_slot": 2,
                                "target_id": 25,
                                "target_slot": 2,
                                "type": "BOOLEAN",
                            },
                            {
                                "id": 43,
                                "origin_id": 24,
                                "origin_slot": 1,
                                "target_id": 25,
                                "target_slot": 1,
                                "type": "INT",
                            },
                            {
                                "id": 47,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 26,
                                "target_slot": 0,
                                "type": "FLOAT,INT,BOOLEAN",
                            },
                            {
                                "id": 48,
                                "origin_id": -10,
                                "origin_slot": 1,
                                "target_id": 26,
                                "target_slot": 1,
                                "type": "FLOAT,INT,BOOLEAN",
                            },
                            {
                                "id": 53,
                                "origin_id": 26,
                                "origin_slot": 2,
                                "target_id": 28,
                                "target_slot": 2,
                                "type": "BOOLEAN",
                            },
                            {
                                "id": 54,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 28,
                                "target_slot": 1,
                                "type": "FLOAT,INT,BOOLEAN",
                            },
                            {
                                "id": 55,
                                "origin_id": -10,
                                "origin_slot": 1,
                                "target_id": 28,
                                "target_slot": 0,
                                "type": "FLOAT,INT,BOOLEAN",
                            },
                            {
                                "id": 56,
                                "origin_id": 28,
                                "origin_slot": 0,
                                "target_id": 17,
                                "target_slot": 0,
                                "type": "FLOAT,INT,BOOLEAN",
                            },
                            {
                                "id": 57,
                                "origin_id": 21,
                                "origin_slot": 0,
                                "target_id": 17,
                                "target_slot": 1,
                                "type": "INT",
                            },
                            {
                                "id": 61,
                                "origin_id": 21,
                                "origin_slot": 0,
                                "target_id": 23,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 62,
                                "origin_id": 28,
                                "origin_slot": 0,
                                "target_id": 23,
                                "target_slot": 1,
                                "type": "FLOAT,INT,BOOLEAN",
                            },
                            {
                                "id": 63,
                                "origin_id": 26,
                                "origin_slot": 2,
                                "target_id": 30,
                                "target_slot": 2,
                                "type": "BOOLEAN",
                            },
                            {
                                "id": 64,
                                "origin_id": -10,
                                "origin_slot": 1,
                                "target_id": 30,
                                "target_slot": 1,
                                "type": "FLOAT,INT,BOOLEAN",
                            },
                            {
                                "id": 65,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 30,
                                "target_slot": 0,
                                "type": "FLOAT,INT,BOOLEAN",
                            },
                            {
                                "id": 66,
                                "origin_id": 30,
                                "origin_slot": 0,
                                "target_id": 24,
                                "target_slot": 1,
                                "type": "FLOAT,INT,BOOLEAN",
                            },
                            {
                                "id": 67,
                                "origin_id": 28,
                                "origin_slot": 0,
                                "target_id": 18,
                                "target_slot": 0,
                                "type": "FLOAT,INT,BOOLEAN",
                            },
                            {
                                "id": 68,
                                "origin_id": 30,
                                "origin_slot": 0,
                                "target_id": 25,
                                "target_slot": 0,
                                "type": "FLOAT,INT,BOOLEAN",
                            },
                            {
                                "id": 69,
                                "origin_id": 26,
                                "origin_slot": 2,
                                "target_id": 31,
                                "target_slot": 2,
                                "type": "BOOLEAN",
                            },
                            {
                                "id": 70,
                                "origin_id": 26,
                                "origin_slot": 2,
                                "target_id": 32,
                                "target_slot": 2,
                                "type": "BOOLEAN",
                            },
                            {
                                "id": 71,
                                "origin_id": 18,
                                "origin_slot": 0,
                                "target_id": 31,
                                "target_slot": 1,
                                "type": "INT",
                            },
                            {
                                "id": 72,
                                "origin_id": 25,
                                "origin_slot": 0,
                                "target_id": 31,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 73,
                                "origin_id": 25,
                                "origin_slot": 0,
                                "target_id": 32,
                                "target_slot": 1,
                                "type": "INT",
                            },
                            {
                                "id": 74,
                                "origin_id": 18,
                                "origin_slot": 0,
                                "target_id": 32,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 75,
                                "origin_id": 32,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 1,
                                "type": "INT",
                            },
                            {
                                "id": 76,
                                "origin_id": 31,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 83,
                                "origin_id": 23,
                                "origin_slot": 0,
                                "target_id": 24,
                                "target_slot": 0,
                                "type": "FLOAT",
                            },
                        ],
                        "extra": {},
                    },
                    {
                        "id": "afaa09b7-2eca-412c-9c1e-17a7e979df5a",
                        "version": 1,
                        "state": {
                            "lastGroupId": 0,
                            "lastNodeId": 53,
                            "lastLinkId": 130,
                            "lastRerouteId": 0,
                        },
                        "revision": 0,
                        "config": {},
                        "name": "Fps Calc",
                        "inputNode": {
                            "id": -10,
                            "bounding": [
                                -678.6961099181169,
                                425.90097391913673,
                                128,
                                88,
                            ],
                        },
                        "outputNode": {
                            "id": -20,
                            "bounding": [
                                1190.9689125130985,
                                415.90097391913673,
                                128,
                                88,
                            ],
                        },
                        "inputs": [
                            {
                                "id": "6ec45481-262d-4e6e-89a0-eef7ab400984",
                                "name": "values.a",
                                "type": "FLOAT,INT,BOOLEAN",
                                "linkIds": [103, 102, 96],
                                "localized_name": "values.a",
                                "label": "source_fps",
                                "pos": [-574.6961099181169, 449.90097391913673],
                            },
                            {
                                "id": "8fe6ae81-afbe-4307-b02e-6e50ddca91ec",
                                "name": "value",
                                "type": "INT",
                                "linkIds": [114],
                                "label": "target_fps",
                                "pos": [-574.6961099181169, 469.90097391913673],
                            },
                        ],
                        "outputs": [
                            {
                                "id": "1c79ffd4-41df-4477-9822-03f893247b5e",
                                "name": "output",
                                "type": "INT",
                                "linkIds": [107],
                                "localized_name": "output",
                                "label": "multiplier",
                                "pos": [1214.9689125130985, 439.90097391913673],
                            },
                            {
                                "id": "3a59af65-0e90-45c2-a508-576b8357735d",
                                "name": "FLOAT",
                                "type": "FLOAT",
                                "linkIds": [97, 97],
                                "localized_name": "FLOAT",
                                "label": "actual_fps",
                                "pos": [1214.9689125130985, 459.90097391913673],
                            },
                        ],
                        "widgets": [],
                        "nodes": [
                            {
                                "id": 49,
                                "type": "ComfyMathExpression",
                                "pos": [-487.5470343221414, 252.6981104152982],
                                "size": [400, 200],
                                "flags": {},
                                "order": 5,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "a",
                                        "localized_name": "values.a",
                                        "name": "values.a",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 103,
                                    },
                                    {
                                        "label": "b",
                                        "localized_name": "values.b",
                                        "name": "values.b",
                                        "shape": 7,
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 104,
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
                                        "links": [105],
                                    },
                                ],
                                "properties": {
                                    "Node name for S&R": "ComfyMathExpression"
                                },
                                "widgets_values": ["a > b"],
                            },
                            {
                                "id": 50,
                                "type": "PrimitiveInt",
                                "pos": [-353.10547690288996, 746.8020703685521],
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
                                        "links": [108],
                                    }
                                ],
                                "properties": {"Node name for S&R": "PrimitiveInt"},
                                "widgets_values": [1, "fixed"],
                            },
                            {
                                "id": 48,
                                "type": "ComfySwitchNode",
                                "pos": [230.2629633455232, 423.0356926672857],
                                "size": [270, 78],
                                "flags": {},
                                "order": 4,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "on_false",
                                        "name": "on_false",
                                        "type": "INT",
                                        "link": 106,
                                    },
                                    {
                                        "localized_name": "on_true",
                                        "name": "on_true",
                                        "type": "INT",
                                        "link": 108,
                                    },
                                    {
                                        "localized_name": "switch",
                                        "name": "switch",
                                        "type": "BOOLEAN",
                                        "widget": {"name": "switch"},
                                        "link": 105,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "output",
                                        "name": "output",
                                        "type": "INT",
                                        "links": [107, 109],
                                    }
                                ],
                                "properties": {"Node name for S&R": "ComfySwitchNode"},
                                "widgets_values": [False],
                            },
                            {
                                "id": 42,
                                "type": "PrimitiveInt",
                                "pos": [-490.6961099181169, 120.99987746972138],
                                "size": [314.97089736713303, 82],
                                "flags": {},
                                "order": 1,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "value",
                                        "name": "value",
                                        "type": "INT",
                                        "widget": {"name": "value"},
                                        "link": 114,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "INT",
                                        "name": "INT",
                                        "type": "INT",
                                        "links": [101, 104],
                                    }
                                ],
                                "properties": {"Node name for S&R": "PrimitiveInt"},
                                "widgets_values": [60, "fixed"],
                            },
                            {
                                "id": 43,
                                "type": "ComfyMathExpression",
                                "pos": [752.6297704479545, 542.5033674740939],
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
                                        "link": 109,
                                    },
                                    {
                                        "label": "b",
                                        "localized_name": "values.b",
                                        "name": "values.b",
                                        "shape": 7,
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 96,
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
                                        "links": [97],
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
                                        "links": None,
                                    },
                                ],
                                "title": "FPS Target",
                                "properties": {
                                    "Node name for S&R": "ComfyMathExpression"
                                },
                                "widgets_values": ["a * b"],
                            },
                            {
                                "id": 47,
                                "type": "ComfyMathExpression",
                                "pos": [-485.4419865946947, 503.4652617026675],
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
                                        "link": 101,
                                    },
                                    {
                                        "label": "b",
                                        "localized_name": "values.b",
                                        "name": "values.b",
                                        "shape": 7,
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 102,
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
                                        "links": [106],
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
                                "widgets_values": ["a * 1.0 / b"],
                            },
                        ],
                        "groups": [],
                        "links": [
                            {
                                "id": 104,
                                "origin_id": 42,
                                "origin_slot": 0,
                                "target_id": 49,
                                "target_slot": 1,
                                "type": "INT",
                            },
                            {
                                "id": 101,
                                "origin_id": 42,
                                "origin_slot": 0,
                                "target_id": 47,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 106,
                                "origin_id": 47,
                                "origin_slot": 1,
                                "target_id": 48,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 108,
                                "origin_id": 50,
                                "origin_slot": 0,
                                "target_id": 48,
                                "target_slot": 1,
                                "type": "INT",
                            },
                            {
                                "id": 105,
                                "origin_id": 49,
                                "origin_slot": 2,
                                "target_id": 48,
                                "target_slot": 2,
                                "type": "BOOLEAN",
                            },
                            {
                                "id": 109,
                                "origin_id": 48,
                                "origin_slot": 0,
                                "target_id": 43,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 103,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 49,
                                "target_slot": 0,
                                "type": "FLOAT",
                            },
                            {
                                "id": 102,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 47,
                                "target_slot": 1,
                                "type": "FLOAT",
                            },
                            {
                                "id": 96,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 43,
                                "target_slot": 1,
                                "type": "FLOAT",
                            },
                            {
                                "id": 107,
                                "origin_id": 48,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 97,
                                "origin_id": 43,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 1,
                                "type": "FLOAT",
                            },
                            {
                                "id": 114,
                                "origin_id": -10,
                                "origin_slot": 1,
                                "target_id": 42,
                                "target_slot": 0,
                                "type": "INT",
                            },
                        ],
                        "extra": {},
                    },
                    {
                        "id": "0b3e9d59-10df-4754-b7af-5c7d5bbc08a0",
                        "version": 1,
                        "state": {
                            "lastGroupId": 0,
                            "lastNodeId": 53,
                            "lastLinkId": 130,
                            "lastRerouteId": 0,
                        },
                        "revision": 0,
                        "config": {},
                        "name": "Config",
                        "inputNode": {
                            "id": -10,
                            "bounding": [338.1088450758484, 848.4531076256688, 128, 88],
                        },
                        "outputNode": {
                            "id": -20,
                            "bounding": [
                                1193.800759466324,
                                818.4531076256688,
                                128,
                                128,
                            ],
                        },
                        "inputs": [
                            {
                                "id": "189e8b02-67f1-442a-b1c4-28c714d142dd",
                                "name": "video_info",
                                "type": "VHS_VIDEOINFO",
                                "linkIds": [115],
                                "localized_name": "video_info",
                                "pos": [442.1088450758484, 872.4531076256688],
                            },
                            {
                                "id": "8e44e205-adc2-4e70-a99b-a94a6526b54f",
                                "name": "value",
                                "type": "INT",
                                "linkIds": [130],
                                "label": "target_fps",
                                "pos": [442.1088450758484, 892.4531076256688],
                            },
                        ],
                        "outputs": [
                            {
                                "id": "4a9b05ec-aba4-4768-b508-2264a1f6b191",
                                "name": "FLOAT",
                                "type": "FLOAT",
                                "linkIds": [112],
                                "localized_name": "FLOAT",
                                "label": "actual_fps",
                                "pos": [1217.800759466324, 842.4531076256688],
                            },
                            {
                                "id": "6fc0d4f5-4242-4ef9-b14f-ab34c94036bb",
                                "name": "output",
                                "type": "INT",
                                "linkIds": [124],
                                "label": "multiplier",
                                "pos": [1217.800759466324, 862.4531076256688],
                            },
                            {
                                "id": "72ac05fd-b1ce-4eac-b793-e634368f6339",
                                "name": "output_1",
                                "type": "INT",
                                "linkIds": [125],
                                "label": "width",
                                "pos": [1217.800759466324, 882.4531076256688],
                            },
                            {
                                "id": "c230c1f5-8dab-4718-b6a9-4f8c39b34c21",
                                "name": "INT",
                                "type": "INT",
                                "linkIds": [126],
                                "label": "height",
                                "pos": [1217.800759466324, 902.4531076256688],
                            },
                        ],
                        "widgets": [],
                        "nodes": [
                            {
                                "id": 52,
                                "type": "VHS_VideoInfoSource",
                                "pos": [526.1088450758484, 840.5094134228187],
                                "size": [247.6666717529297, 106],
                                "flags": {},
                                "order": 2,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "video_info",
                                        "name": "video_info",
                                        "type": "VHS_VIDEOINFO",
                                        "link": 115,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "fps🟨",
                                        "name": "fps🟨",
                                        "type": "FLOAT",
                                        "links": [116],
                                    },
                                    {
                                        "localized_name": "frame_count🟨",
                                        "name": "frame_count🟨",
                                        "type": "INT",
                                        "links": None,
                                    },
                                    {
                                        "localized_name": "duration🟨",
                                        "name": "duration🟨",
                                        "type": "FLOAT",
                                        "links": None,
                                    },
                                    {
                                        "localized_name": "width🟨",
                                        "name": "width🟨",
                                        "type": "INT",
                                        "links": [117],
                                    },
                                    {
                                        "localized_name": "height🟨",
                                        "name": "height🟨",
                                        "type": "INT",
                                        "links": [118],
                                    },
                                ],
                                "properties": {
                                    "Node name for S&R": "VHS_VideoInfoSource"
                                },
                                "widgets_values": {},
                            },
                            {
                                "id": 22,
                                "type": "80bfb28a-9007-4908-9447-28fab825e3e1",
                                "pos": [797.5174342710116, 907.986542910608],
                                "size": [336.2833251953125, 107.18333435058594],
                                "flags": {},
                                "order": 0,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "width",
                                        "localized_name": "values.a",
                                        "name": "values.a",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 117,
                                    },
                                    {
                                        "label": "height",
                                        "localized_name": "values.b",
                                        "name": "values.b",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 118,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "label": "width",
                                        "name": "output",
                                        "type": "INT",
                                        "links": [125],
                                    },
                                    {
                                        "label": "height",
                                        "name": "INT",
                                        "type": "INT",
                                        "links": [126],
                                    },
                                ],
                                "properties": {"previewExposures": []},
                            },
                            {
                                "id": 51,
                                "type": "afaa09b7-2eca-412c-9c1e-17a7e979df5a",
                                "pos": [801.4975189513771, 779.7363379901437],
                                "size": [326.71825456930526, 78],
                                "flags": {},
                                "order": 1,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "label": "source_fps",
                                        "localized_name": "values.a",
                                        "name": "values.a",
                                        "type": "FLOAT,INT,BOOLEAN",
                                        "link": 116,
                                    },
                                    {
                                        "label": "target_fps",
                                        "name": "value",
                                        "type": "INT",
                                        "widget": {"name": "value"},
                                        "link": 130,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "label": "multiplier",
                                        "localized_name": "output",
                                        "name": "output",
                                        "type": "INT",
                                        "links": [124],
                                    },
                                    {
                                        "label": "actual_fps",
                                        "localized_name": "FLOAT",
                                        "name": "FLOAT",
                                        "type": "FLOAT",
                                        "links": [112],
                                    },
                                ],
                                "properties": {"previewExposures": []},
                            },
                        ],
                        "groups": [],
                        "links": [
                            {
                                "id": 116,
                                "origin_id": 52,
                                "origin_slot": 0,
                                "target_id": 51,
                                "target_slot": 0,
                                "type": "FLOAT",
                            },
                            {
                                "id": 117,
                                "origin_id": 52,
                                "origin_slot": 3,
                                "target_id": 22,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 118,
                                "origin_id": 52,
                                "origin_slot": 4,
                                "target_id": 22,
                                "target_slot": 1,
                                "type": "INT",
                            },
                            {
                                "id": 115,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 52,
                                "target_slot": 0,
                                "type": "VHS_VIDEOINFO",
                            },
                            {
                                "id": 112,
                                "origin_id": 51,
                                "origin_slot": 1,
                                "target_id": -20,
                                "target_slot": 0,
                                "type": "FLOAT",
                            },
                            {
                                "id": 124,
                                "origin_id": 51,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 1,
                                "type": "INT",
                            },
                            {
                                "id": 125,
                                "origin_id": 22,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 2,
                                "type": "INT",
                            },
                            {
                                "id": 126,
                                "origin_id": 22,
                                "origin_slot": 1,
                                "target_id": -20,
                                "target_slot": 3,
                                "type": "INT",
                            },
                            {
                                "id": 130,
                                "origin_id": -10,
                                "origin_slot": 1,
                                "target_id": 51,
                                "target_slot": 1,
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
                    "scale": 1.1542798351615216,
                    "offset": [-254.82001130441128, -493.8495386814351],
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
    target_fps: int = 60,
    unload_models: bool | None = None,
):
    bootstrap_comfyui_runtime()
    add_extra_model_paths()
    import_custom_nodes()

    from nodes import NODE_CLASS_MAPPINGS
    import folder_paths
    import math
    import os
    import glob as glob_module
    import torch

    try:
        with torch.inference_mode():
            vhs_loadvideoffmpeg = NODE_CLASS_MAPPINGS["VHS_LoadVideoFFmpeg"]()
            vhs_loadvideoffmpeg_9 = vhs_loadvideoffmpeg.load_video(
                video=video_path,
                force_rate=0,
                custom_width=0,
                custom_height=0,
                frame_load_cap=0,
                start_time=0,
                format="None",
                unique_id=10807253530262432976,
            )
            autoloadrifetensorrtmodel = NODE_CLASS_MAPPINGS[
                "AutoLoadRifeTensorrtModel"
            ]()
            autoloadrifetensorrtmodel_13 = (
                autoloadrifetensorrtmodel.load_rife_tensorrt_model(
                    model="rife49_ensemble_True_scale_1_sim",
                    precision="fp32",
                    resolution_profile="medium",
                )
            )
            primitiveint = NODE_CLASS_MAPPINGS["PrimitiveInt"]()
            primitiveint_53_22_21 = primitiveint.EXECUTE_NORMALIZED(value=672)
            vhs_videoinfosource = NODE_CLASS_MAPPINGS["VHS_VideoInfoSource"]()
            comfyswitchnode = NODE_CLASS_MAPPINGS["ComfySwitchNode"]()
            imageresizekjv2 = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
            autorifetensorrt = NODE_CLASS_MAPPINGS["AutoRifeTensorrt"]()
            vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

            for q in range(1):
                vhs_videoinfosource_53_52 = vhs_videoinfosource.get_video_info(
                    video_info=get_value_at_index(vhs_loadvideoffmpeg_9, 3)
                )

                source_fps = get_value_at_index(vhs_videoinfosource_53_52, 0)
                source_width = get_value_at_index(vhs_videoinfosource_53_52, 3)
                source_height = get_value_at_index(vhs_videoinfosource_53_52, 4)

                if source_fps > target_fps:
                    multiplier = 1
                    actual_fps = source_fps
                else:
                    multiplier = int(math.ceil(target_fps / source_fps))
                    actual_fps = target_fps

                if source_width < source_height:
                    width = source_height
                    height = source_height
                else:
                    width = source_width
                    height = source_width

                if source_height >= 672:
                    height = source_height
                    width = source_height
                else:
                    width = 672
                    height = (
                        int(672 * source_height / source_width)
                        if source_width > 0
                        else 672
                    )

                imageresizekjv2_37 = imageresizekjv2.resize(
                    width=width,
                    height=height,
                    upscale_method="nvidia_rtx_vsr",
                    keep_proportion="crop",
                    pad_color="0, 0, 0",
                    crop_position="center",
                    divisible_by=2,
                    device="gpu",
                    image=get_value_at_index(vhs_loadvideoffmpeg_9, 0),
                    unique_id=17054480493452403746,
                )
                autorifetensorrt_12 = autorifetensorrt.vfi(
                    clear_cache_after_n_frames=100,
                    multiplier=multiplier,
                    keep_model_loaded=False,
                    frames=get_value_at_index(imageresizekjv2_37, 0),
                    rife_trt_model=get_value_at_index(autoloadrifetensorrtmodel_13, 0),
                )
                vhs_videocombine_6 = vhs_videocombine.combine_video(
                    frame_rate=actual_fps,
                    loop_count=0,
                    filename_prefix="rife",
                    format="video/h264-mp4",
                    pix_fmt="yuv420p",
                    crf=19,
                    save_metadata=False,
                    trim_to_audio=False,
                    pingpong=False,
                    save_output=True,
                    images=get_value_at_index(autorifetensorrt_12, 0),
                    audio=get_value_at_index(vhs_loadvideoffmpeg_9, 2),
                    unique_id=16297855488570926581,
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                )

                output_dir = folder_paths.get_output_directory()
                pattern = os.path.join(output_dir, "rife_*.mp4")
                files = sorted(glob_module.glob(pattern), key=os.path.getmtime)
                if files:
                    return files[-1]
                return None
    finally:
        cleanup_comfyui_runtime(unload_models=unload_models)