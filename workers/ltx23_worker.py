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
        "98": {
            "inputs": {"image": "ab67616d00001e021ecdd83cdb58ea92d29ef230.png"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"},
        },
        "168": {
            "inputs": {"sampler_name": "euler_ancestral"},
            "class_type": "KSamplerSelect",
            "_meta": {"title": "KSamplerSelect"},
        },
        "169": {
            "inputs": {
                "steps": 8,
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1,
                "latent": ["182", 0],
            },
            "class_type": "LTXVScheduler",
            "_meta": {"title": "LTXVScheduler"},
        },
        "174": {
            "inputs": {
                "frames_number": ["185", 0],
                "frame_rate": ["236", 1],
                "batch_size": 1,
                "audio_vae": ["211", 0],
            },
            "class_type": "LTXVEmptyLatentAudio",
            "_meta": {"title": "LTXV Empty Latent Audio"},
        },
        "175": {
            "inputs": {
                "noise": ["199", 0],
                "guider": ["193", 0],
                "sampler": ["168", 0],
                "sigmas": ["169", 0],
                "latent_image": ["182", 0],
            },
            "class_type": "SamplerCustomAdvanced",
            "_meta": {"title": "SamplerCustomAdvanced"},
        },
        "182": {
            "inputs": {"video_latent": ["190", 0], "audio_latent": ["174", 0]},
            "class_type": "LTXVConcatAVLatent",
            "_meta": {"title": "LTXVConcatAVLatent"},
        },
        "185": {
            "inputs": {"value": 10},
            "class_type": "PrimitiveInt",
            "_meta": {"title": "Video Length"},
        },
        "188": {
            "inputs": {"img_compression": 33, "image": ["233", 0]},
            "class_type": "LTXVPreprocess",
            "_meta": {"title": "LTXV Preprocess"},
        },
        "189": {
            "inputs": {
                "width": ["233", 1],
                "height": ["233", 2],
                "length": ["185", 0],
                "batch_size": 1,
            },
            "class_type": "EmptyLTXVLatentVideo",
            "_meta": {"title": "EmptyLTXVLatentVideo"},
        },
        "190": {
            "inputs": {
                "strength": 1,
                "bypass": False,
                "vae": ["210", 0],
                "image": ["188", 0],
                "latent": ["189", 0],
            },
            "class_type": "LTXVImgToVideoInplace",
            "_meta": {"title": "LTXVImgToVideoInplace"},
        },
        "193": {
            "inputs": {
                "cfg": 1,
                "model": ["230", 0],
                "positive": ["201", 0],
                "negative": ["201", 1],
            },
            "class_type": "CFGGuider",
            "_meta": {"title": "CFG Guider"},
        },
        "199": {
            "inputs": {"noise_seed": 10},
            "class_type": "RandomNoise",
            "_meta": {"title": "RandomNoise"},
        },
        "201": {
            "inputs": {
                "frame_rate": ["236", 0],
                "positive": ["203", 0],
                "negative": ["226", 0],
            },
            "class_type": "LTXVConditioning",
            "_meta": {"title": "LTXVConditioning"},
        },
        "203": {
            "inputs": {"text": ["230", 2], "clip": ["230", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"},
        },
        "209": {
            "inputs": {
                "clip_name1": "gemma_3_12B_it_fp4_mixed.safetensors",
                "clip_name2": "ltx-2.3_text_projection_bf16.safetensors",
                "type": "ltxv",
                "device": "default",
            },
            "class_type": "DualCLIPLoader",
            "_meta": {"title": "DualCLIPLoader"},
        },
        "210": {
            "inputs": {
                "vae_name": "LTX23_video_vae_bf16.safetensors",
                "device": "main_device",
                "weight_dtype": "bf16",
            },
            "class_type": "VAELoaderKJ",
            "_meta": {"title": "VAELoader KJ"},
        },
        "211": {
            "inputs": {
                "vae_name": "LTX23_audio_vae_bf16.safetensors",
                "device": "main_device",
                "weight_dtype": "bf16",
            },
            "class_type": "VAELoaderKJ",
            "_meta": {"title": "VAELoader KJ"},
        },
        "212": {
            "inputs": {
                "unet_name": "ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors",
                "weight_dtype": "default",
            },
            "class_type": "UNETLoader",
            "_meta": {"title": "Load Diffusion Model"},
        },
        "213": {
            "inputs": {"samples": ["217", 1], "audio_vae": ["211", 0]},
            "class_type": "LTXVAudioVAEDecode",
            "_meta": {"title": "LTXV Audio VAE Decode"},
        },
        "214": {
            "inputs": {"samples": ["217", 0], "vae": ["210", 0]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
        },
        "217": {
            "inputs": {"av_latent": ["175", 1]},
            "class_type": "LTXVSeparateAVLatent",
            "_meta": {"title": "LTXVSeparateAVLatent"},
        },
        "226": {
            "inputs": {"conditioning": ["203", 0]},
            "class_type": "ConditioningZeroOut",
            "_meta": {"title": "ConditioningZeroOut"},
        },
        "230": {
            "inputs": {
                "text": "The objects flying surround",
                "model": ["212", 0],
                "clip": ["209", 0],
            },
            "class_type": "LoraTagLoader",
            "_meta": {"title": "Load LoRA Tag"},
        },
        "233": {
            "inputs": {"image": ["98", 0]},
            "class_type": "GetImageSizeAndCount",
            "_meta": {"title": "Get Image Size & Count"},
        },
        "236": {
            "inputs": {"value": ["237", 0]},
            "class_type": "ComfyNumberConvert",
            "_meta": {"title": "Convert Number"},
        },
        "237": {
            "inputs": {"value": 24},
            "class_type": "INTConstant",
            "_meta": {"title": "FPS"},
        },
        "238": {
            "inputs": {
                "frame_rate": ["236", 0],
                "loop_count": 0,
                "filename_prefix": "LTX23",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": False,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
                "images": ["214", 0],
                "audio": ["213", 0],
            },
            "class_type": "VHS_VideoCombine",
            "_meta": {"title": "Video Combine 🎥🅥🅗🅢"},
        },
    }


def build_extra_pnginfo() -> dict[str, Any] | None:
    return {
        "workflow": {
            "id": "07824bbb-6672-4bb0-ac36-4313a519e35b",
            "revision": 0,
            "last_node_id": 240,
            "last_link_id": 556,
            "nodes": [
                {
                    "id": 182,
                    "type": "LTXVConcatAVLatent",
                    "pos": [172.6215850900826, 3313.2982874986233],
                    "size": [225, 72],
                    "flags": {},
                    "order": 22,
                    "mode": 0,
                    "inputs": [
                        {"name": "video_latent", "type": "LATENT", "link": 441},
                        {"name": "audio_latent", "type": "LATENT", "link": 442},
                    ],
                    "outputs": [
                        {"name": "latent", "type": "LATENT", "links": [418, 431]}
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
                    "id": 188,
                    "type": "LTXVPreprocess",
                    "pos": [-123.67841490991736, 3499.898287498623],
                    "size": [225, 80],
                    "flags": {},
                    "order": 17,
                    "mode": 0,
                    "inputs": [{"name": "image", "type": "IMAGE", "link": 556}],
                    "outputs": [
                        {"name": "output_image", "type": "IMAGE", "links": [454]}
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
                    "widgets_values": [33],
                },
                {
                    "id": 189,
                    "type": "EmptyLTXVLatentVideo",
                    "pos": [-167.3784149099174, 3063.2982874986233],
                    "size": [270, 176],
                    "flags": {},
                    "order": 18,
                    "mode": 0,
                    "inputs": [
                        {
                            "name": "width",
                            "type": "INT",
                            "widget": {"name": "width"},
                            "link": 554,
                        },
                        {
                            "name": "height",
                            "type": "INT",
                            "widget": {"name": "height"},
                            "link": 555,
                        },
                        {
                            "name": "length",
                            "type": "INT",
                            "widget": {"name": "length"},
                            "link": 452,
                        },
                    ],
                    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [455]}],
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
                    "id": 190,
                    "type": "LTXVImgToVideoInplace",
                    "pos": [172.6215850900826, 3073.2982874986233],
                    "size": [270, 156],
                    "flags": {},
                    "order": 20,
                    "mode": 0,
                    "inputs": [
                        {"name": "vae", "type": "VAE", "link": 490},
                        {"name": "image", "type": "IMAGE", "link": 454},
                        {"name": "latent", "type": "LATENT", "link": 455},
                    ],
                    "outputs": [{"name": "latent", "type": "LATENT", "links": [441]}],
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
                    "id": 217,
                    "type": "LTXVSeparateAVLatent",
                    "pos": [1089.2768952222823, 2245.0718911481345],
                    "size": [225, 72],
                    "flags": {},
                    "order": 26,
                    "mode": 0,
                    "inputs": [{"name": "av_latent", "type": "LATENT", "link": 487}],
                    "outputs": [
                        {"name": "video_latent", "type": "LATENT", "links": [483]},
                        {"name": "audio_latent", "type": "LATENT", "links": [482]},
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
                    "id": 175,
                    "type": "SamplerCustomAdvanced",
                    "pos": [793.7257665876041, 2193.2982874986233],
                    "size": [225, 144],
                    "flags": {},
                    "order": 25,
                    "mode": 0,
                    "inputs": [
                        {"name": "noise", "type": "NOISE", "link": 427},
                        {"name": "guider", "type": "GUIDER", "link": 428},
                        {"name": "sampler", "type": "SAMPLER", "link": 429},
                        {"name": "sigmas", "type": "SIGMAS", "link": 430},
                        {"name": "latent_image", "type": "LATENT", "link": 431},
                    ],
                    "outputs": [
                        {"name": "output", "type": "LATENT", "links": []},
                        {"name": "denoised_output", "type": "LATENT", "links": [487]},
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
                    "id": 218,
                    "type": "Reroute",
                    "pos": [-767.2560523002735, 2406.403702535514],
                    "size": [75, 26],
                    "flags": {},
                    "order": 9,
                    "mode": 0,
                    "inputs": [{"name": "", "type": "*", "link": 489}],
                    "outputs": [{"name": "", "type": "VAE", "links": [490, 491]}],
                    "properties": {
                        "showOutputText": False,
                        "horizontal": False,
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.1",
                            "input_ue_unconnectable": {},
                        },
                    },
                },
                {
                    "id": 169,
                    "type": "LTXVScheduler",
                    "pos": [483.7257665876041, 2643.2982874986233],
                    "size": [270, 172],
                    "flags": {},
                    "order": 24,
                    "mode": 0,
                    "inputs": [
                        {"name": "latent", "shape": 7, "type": "LATENT", "link": 418}
                    ],
                    "outputs": [{"name": "SIGMAS", "type": "SIGMAS", "links": [430]}],
                    "properties": {
                        "Node name for S&R": "LTXVScheduler",
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
                    "widgets_values": [8, 2.05, 0.95, True, 0.1],
                },
                {
                    "id": 193,
                    "type": "CFGGuider",
                    "pos": [483.7257665876041, 2343.2982874986233],
                    "size": [270, 128],
                    "flags": {},
                    "order": 23,
                    "mode": 0,
                    "inputs": [
                        {"name": "model", "type": "MODEL", "link": 534},
                        {"name": "positive", "type": "CONDITIONING", "link": 462},
                        {"name": "negative", "type": "CONDITIONING", "link": 463},
                    ],
                    "outputs": [{"name": "GUIDER", "type": "GUIDER", "links": [428]}],
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
                    "id": 168,
                    "type": "KSamplerSelect",
                    "pos": [483.7257665876041, 2513.2982874986233],
                    "size": [270, 80],
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "SAMPLER", "type": "SAMPLER", "links": [429]}],
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
                    "id": 210,
                    "type": "VAELoaderKJ",
                    "pos": [-1278.0597967748454, 2406.344587805419],
                    "size": [466.75, 168],
                    "flags": {},
                    "order": 1,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "VAE", "type": "VAE", "links": [489]}],
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
                    "id": 211,
                    "type": "VAELoaderKJ",
                    "pos": [-1267.1680100604758, 2625.647226070537],
                    "size": [466.75, 168],
                    "flags": {},
                    "order": 2,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "VAE", "type": "VAE", "links": [492]}],
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
                    "id": 219,
                    "type": "Reroute",
                    "pos": [-741.5917472252003, 2626.431544768182],
                    "size": [75, 26],
                    "flags": {},
                    "order": 10,
                    "mode": 0,
                    "inputs": [{"name": "", "type": "*", "link": 492}],
                    "outputs": [{"name": "", "type": "VAE", "links": [493, 494]}],
                    "properties": {
                        "showOutputText": False,
                        "horizontal": False,
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "version": "7.1",
                            "input_ue_unconnectable": {},
                        },
                    },
                },
                {
                    "id": 226,
                    "type": "ConditioningZeroOut",
                    "pos": [-127.15572448046912, 2845.965181782624],
                    "size": [225, 48],
                    "flags": {},
                    "order": 19,
                    "mode": 0,
                    "inputs": [
                        {"name": "conditioning", "type": "CONDITIONING", "link": 505}
                    ],
                    "outputs": [
                        {"name": "CONDITIONING", "type": "CONDITIONING", "links": [506]}
                    ],
                    "properties": {"Node name for S&R": "ConditioningZeroOut"},
                    "widgets_values": [],
                },
                {
                    "id": 199,
                    "type": "RandomNoise",
                    "pos": [483.7257665876041, 2193.2982874986233],
                    "size": [270, 82],
                    "flags": {},
                    "order": 3,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "NOISE", "type": "NOISE", "links": [427]}],
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
                    "id": 203,
                    "type": "CLIPTextEncode",
                    "pos": [-166.27423341239592, 2223.2982874986233],
                    "size": [590, 350],
                    "flags": {},
                    "order": 15,
                    "mode": 0,
                    "inputs": [
                        {"name": "clip", "type": "CLIP", "link": 528},
                        {
                            "name": "text",
                            "type": "STRING",
                            "widget": {"name": "text"},
                            "link": 529,
                        },
                    ],
                    "outputs": [
                        {
                            "name": "CONDITIONING",
                            "type": "CONDITIONING",
                            "links": [470, 505],
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
                    "widgets_values": ["Helloworld\n"],
                    "color": "#232",
                    "bgcolor": "#353",
                },
                {
                    "id": 212,
                    "type": "UNETLoader",
                    "pos": [-1885.9234451292718, 2003.4856129013556],
                    "size": [558.7333374023438, 108],
                    "flags": {},
                    "order": 4,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "MODEL", "type": "MODEL", "links": [525]}],
                    "properties": {
                        "Node name for S&R": "UNETLoader",
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "input_ue_unconnectable": {},
                            "version": "7.1",
                        },
                    },
                    "widgets_values": [
                        "ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors",
                        "default",
                    ],
                    "color": "#322",
                    "bgcolor": "#533",
                },
                {
                    "id": 209,
                    "type": "DualCLIPLoader",
                    "pos": [-1893.9660117298224, 2159.194136869951],
                    "size": [570.75, 172],
                    "flags": {},
                    "order": 5,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "CLIP", "type": "CLIP", "links": [526]}],
                    "properties": {
                        "Node name for S&R": "DualCLIPLoader",
                        "cnr_id": "comfy-core",
                        "ver": "0.8.2",
                        "ue_properties": {
                            "widget_ue_connectable": {},
                            "input_ue_unconnectable": {},
                            "version": "7.1",
                        },
                    },
                    "widgets_values": [
                        "gemma_3_12B_it_fp4_mixed.safetensors",
                        "ltx-2.3_text_projection_bf16.safetensors",
                        "ltxv",
                        "default",
                    ],
                    "color": "#322",
                    "bgcolor": "#533",
                },
                {
                    "id": 225,
                    "type": "PathchSageAttentionKJ",
                    "pos": [-479.17056820211553, 1971.2596373930987],
                    "size": [524.4666748046875, 132],
                    "flags": {},
                    "order": 14,
                    "mode": 4,
                    "inputs": [{"name": "model", "type": "MODEL", "link": 527}],
                    "outputs": [{"name": "MODEL", "type": "MODEL", "links": [534]}],
                    "properties": {"Node name for S&R": "PathchSageAttentionKJ"},
                    "widgets_values": ["sageattn_qk_int8_pv_fp16_cuda", True],
                },
                {
                    "id": 233,
                    "type": "GetImageSizeAndCount",
                    "pos": [-1136.8179534425788, 2932.9916756026364],
                    "size": [323.9333190917969, 144],
                    "flags": {},
                    "order": 13,
                    "mode": 0,
                    "inputs": [{"name": "image", "type": "IMAGE", "link": 538}],
                    "outputs": [
                        {"name": "image", "type": "IMAGE", "links": [556]},
                        {
                            "label": "300 width",
                            "name": "width",
                            "type": "INT",
                            "links": [554],
                        },
                        {
                            "label": "300 height",
                            "name": "height",
                            "type": "INT",
                            "links": [555],
                        },
                        {
                            "label": "1 count",
                            "name": "count",
                            "type": "INT",
                            "links": None,
                        },
                    ],
                    "properties": {"Node name for S&R": "GetImageSizeAndCount"},
                    "widgets_values": [],
                },
                {
                    "id": 185,
                    "type": "PrimitiveInt",
                    "pos": [-1572.8784149099176, 3412.1982874986224],
                    "size": [270, 82],
                    "flags": {},
                    "order": 6,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "INT", "type": "INT", "links": [426, 452]}],
                    "title": "Video Length",
                    "properties": {
                        "Node name for S&R": "PrimitiveInt",
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
                    "widgets_values": [10, "fixed"],
                },
                {
                    "id": 174,
                    "type": "LTXVEmptyLatentAudio",
                    "pos": [-163.74205127355378, 3288.7528329531697],
                    "size": [270, 144],
                    "flags": {},
                    "order": 16,
                    "mode": 0,
                    "inputs": [
                        {"name": "audio_vae", "type": "VAE", "link": 493},
                        {
                            "name": "frames_number",
                            "type": "INT",
                            "widget": {"name": "frames_number"},
                            "link": 426,
                        },
                        {
                            "name": "frame_rate",
                            "type": "INT",
                            "widget": {"name": "frame_rate"},
                            "link": 547,
                        },
                    ],
                    "outputs": [{"name": "Latent", "type": "LATENT", "links": [442]}],
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
                    "id": 201,
                    "type": "LTXVConditioning",
                    "pos": [113.72576658760408, 2823.2982874986233],
                    "size": [270, 104],
                    "flags": {},
                    "order": 21,
                    "mode": 0,
                    "inputs": [
                        {"name": "positive", "type": "CONDITIONING", "link": 470},
                        {"name": "negative", "type": "CONDITIONING", "link": 506},
                        {
                            "name": "frame_rate",
                            "type": "FLOAT",
                            "widget": {"name": "frame_rate"},
                            "link": 546,
                        },
                    ],
                    "outputs": [
                        {"name": "positive", "type": "CONDITIONING", "links": [462]},
                        {"name": "negative", "type": "CONDITIONING", "links": [463]},
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
                    "id": 237,
                    "type": "INTConstant",
                    "pos": [-1570.9773886156618, 3559.36920415991],
                    "size": [270.6166687011719, 104],
                    "flags": {},
                    "order": 7,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "value", "type": "INT", "links": [548]}],
                    "title": "FPS",
                    "properties": {"Node name for S&R": "INTConstant"},
                    "widgets_values": [24],
                    "color": "#1b4669",
                    "bgcolor": "#29699c",
                },
                {
                    "id": 214,
                    "type": "VAEDecode",
                    "pos": [1393.5708153295072, 2186.3926591946306],
                    "size": [240, 72],
                    "flags": {},
                    "order": 27,
                    "mode": 0,
                    "inputs": [
                        {"name": "samples", "type": "LATENT", "link": 483},
                        {"name": "vae", "type": "VAE", "link": 491},
                    ],
                    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [549]}],
                    "properties": {
                        "Node name for S&R": "VAEDecode",
                        "cnr_id": "comfy-core",
                        "ver": "0.3.75",
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
                    "id": 213,
                    "type": "LTXVAudioVAEDecode",
                    "pos": [1394.4799062385982, 2312.756295558267],
                    "size": [240, 72],
                    "flags": {},
                    "order": 28,
                    "mode": 0,
                    "inputs": [
                        {"name": "samples", "type": "LATENT", "link": 482},
                        {
                            "label": "Audio VAE",
                            "name": "audio_vae",
                            "type": "VAE",
                            "link": 494,
                        },
                    ],
                    "outputs": [{"name": "Audio", "type": "AUDIO", "links": [550]}],
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
                    "id": 236,
                    "type": "ComfyNumberConvert",
                    "pos": [-880.1301301556236, 3451.558204159912],
                    "size": [225, 72],
                    "flags": {},
                    "order": 12,
                    "mode": 0,
                    "inputs": [
                        {
                            "label": "value",
                            "name": "value",
                            "type": "INT,FLOAT,STRING,BOOLEAN",
                            "link": 548,
                        }
                    ],
                    "outputs": [
                        {"name": "FLOAT", "type": "FLOAT", "links": [546, 551]},
                        {"name": "INT", "type": "INT", "links": [547]},
                    ],
                    "properties": {"Node name for S&R": "ComfyNumberConvert"},
                    "widgets_values": [],
                },
                {
                    "id": 238,
                    "type": "VHS_VideoCombine",
                    "pos": [1706.901432049964, 2318.0730170260604],
                    "size": [416.3333435058594, 744.3333435058594],
                    "flags": {},
                    "order": 29,
                    "mode": 0,
                    "inputs": [
                        {"name": "images", "type": "IMAGE", "link": 549},
                        {"name": "audio", "shape": 7, "type": "AUDIO", "link": 550},
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
                            "link": 551,
                        },
                    ],
                    "outputs": [
                        {"name": "Filenames", "type": "VHS_FILENAMES", "links": None}
                    ],
                    "properties": {"Node name for S&R": "VHS_VideoCombine"},
                    "widgets_values": {
                        "frame_rate": 8,
                        "loop_count": 0,
                        "filename_prefix": "LTX23",
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
                                "filename": "LTX23_00003-audio.mp4",
                                "subfolder": "",
                                "type": "output",
                                "format": "video/h264-mp4",
                                "frame_rate": 24,
                                "workflow": "LTX23_00003.png",
                                "fullpath": "/comfy/output/LTX23_00003-audio.mp4",
                            },
                        },
                    },
                },
                {
                    "id": 98,
                    "type": "LoadImage",
                    "pos": [-1821.2342391601883, 2838.0374704710057],
                    "size": [488.6333312988281, 519.2333374023438],
                    "flags": {},
                    "order": 8,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [
                        {"name": "IMAGE", "type": "IMAGE", "links": [538]},
                        {"name": "MASK", "type": "MASK", "links": None},
                    ],
                    "properties": {
                        "Node name for S&R": "LoadImage",
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
                    "widgets_values": [
                        "ab67616d00001e021ecdd83cdb58ea92d29ef230.png",
                        "image",
                    ],
                },
                {
                    "id": 230,
                    "type": "LoraTagLoader",
                    "pos": [-987.6860601792171, 2039.2227236663075],
                    "size": [400, 200],
                    "flags": {},
                    "order": 11,
                    "mode": 0,
                    "inputs": [
                        {"name": "model", "type": "MODEL", "link": 525},
                        {"name": "clip", "type": "CLIP", "link": 526},
                    ],
                    "outputs": [
                        {"name": "MODEL", "type": "MODEL", "links": [527]},
                        {"name": "CLIP", "type": "CLIP", "links": [528]},
                        {"name": "STRING", "type": "STRING", "links": [529]},
                    ],
                    "properties": {"Node name for S&R": "LoraTagLoader"},
                    "widgets_values": ["The objects flying surround"],
                },
            ],
            "links": [
                [418, 182, 0, 169, 0, "LATENT"],
                [426, 185, 0, 174, 1, "INT"],
                [427, 199, 0, 175, 0, "NOISE"],
                [428, 193, 0, 175, 1, "GUIDER"],
                [429, 168, 0, 175, 2, "SAMPLER"],
                [430, 169, 0, 175, 3, "SIGMAS"],
                [431, 182, 0, 175, 4, "LATENT"],
                [441, 190, 0, 182, 0, "LATENT"],
                [442, 174, 0, 182, 1, "LATENT"],
                [452, 185, 0, 189, 2, "INT"],
                [454, 188, 0, 190, 1, "IMAGE"],
                [455, 189, 0, 190, 2, "LATENT"],
                [462, 201, 0, 193, 1, "CONDITIONING"],
                [463, 201, 1, 193, 2, "CONDITIONING"],
                [470, 203, 0, 201, 0, "CONDITIONING"],
                [482, 217, 1, 213, 0, "LATENT"],
                [483, 217, 0, 214, 0, "LATENT"],
                [487, 175, 1, 217, 0, "LATENT"],
                [489, 210, 0, 218, 0, "VAE"],
                [490, 218, 0, 190, 0, "VAE"],
                [491, 218, 0, 214, 1, "VAE"],
                [492, 211, 0, 219, 0, "VAE"],
                [493, 219, 0, 174, 0, "VAE"],
                [494, 219, 0, 213, 1, "VAE"],
                [505, 203, 0, 226, 0, "CONDITIONING"],
                [506, 226, 0, 201, 1, "CONDITIONING"],
                [525, 212, 0, 230, 0, "MODEL"],
                [526, 209, 0, 230, 1, "CLIP"],
                [527, 230, 0, 225, 0, "MODEL"],
                [528, 230, 1, 203, 0, "CLIP"],
                [529, 230, 2, 203, 1, "STRING"],
                [534, 225, 0, 193, 0, "MODEL"],
                [538, 98, 0, 233, 0, "IMAGE"],
                [546, 236, 0, 201, 2, "FLOAT"],
                [547, 236, 1, 174, 2, "INT"],
                [548, 237, 0, 236, 0, "INT"],
                [549, 214, 0, 238, 0, "IMAGE"],
                [550, 213, 0, 238, 1, "AUDIO"],
                [551, 236, 0, 238, 4, "FLOAT"],
                [554, 233, 1, 189, 0, "INT"],
                [555, 233, 2, 189, 1, "INT"],
                [556, 233, 0, 188, 0, "IMAGE"],
            ],
            "groups": [],
            "config": {},
            "extra": {
                "ds": {
                    "scale": 0.8264462809917362,
                    "offset": [-549.086089690998, -2059.26360861553],
                },
                "frontendVersion": "1.46.6",
                "workflowRendererVersion": "LG",
                "prompt": {
                    "1": {
                        "inputs": {
                            "ckpt_name": "ltx-av-step-1751000_vocoder_24K.safetensors"
                        },
                        "class_type": "CheckpointLoaderSimple",
                        "_meta": {"title": "Load Checkpoint"},
                    },
                    "2": {
                        "inputs": {
                            "gemma_path": "gemma-3-12b-it-qat-q4_0-unquantized_readout_proj/model/model.safetensors",
                            "ltxv_path": "ltx-av-step-1751000_vocoder_24K.safetensors",
                            "max_length": 1024,
                        },
                        "class_type": "LTXVGemmaCLIPModelLoader",
                        "_meta": {"title": "🅛🅣🅧 Gemma 3 Model " "Loader"},
                    },
                    "3": {
                        "inputs": {"text": "", "clip": ["2", 0]},
                        "class_type": "CLIPTextEncode",
                        "_meta": {"title": "CLIP Text Encode " "(Prompt)"},
                    },
                    "4": {
                        "inputs": {
                            "text": "blurry, low "
                            "quality, still "
                            "frame, frames, "
                            "watermark, "
                            "overlay, titles, "
                            "has blurbox, has "
                            "subtitles",
                            "clip": ["2", 0],
                        },
                        "class_type": "CLIPTextEncode",
                        "_meta": {"title": "CLIP Text Encode " "(Prompt)"},
                    },
                    "8": {
                        "inputs": {"sampler_name": "euler"},
                        "class_type": "KSamplerSelect",
                        "_meta": {"title": "KSamplerSelect"},
                    },
                    "9": {
                        "inputs": {
                            "steps": 20,
                            "max_shift": 2.05,
                            "base_shift": 0.95,
                            "stretch": True,
                            "terminal": 0.1,
                            "latent": ["28", 0],
                        },
                        "class_type": "LTXVScheduler",
                        "_meta": {"title": "LTXVScheduler"},
                    },
                    "11": {
                        "inputs": {"noise_seed": 10},
                        "class_type": "RandomNoise",
                        "_meta": {"title": "RandomNoise"},
                    },
                    "12": {
                        "inputs": {"samples": ["29", 0], "vae": ["1", 2]},
                        "class_type": "VAEDecode",
                        "_meta": {"title": "VAE Decode"},
                    },
                    "13": {
                        "inputs": {
                            "ckpt_name": "ltx-av-step-1751000_vocoder_24K.safetensors"
                        },
                        "class_type": "LTXVAudioVAELoader",
                        "_meta": {"title": "🅛🅣🅧 LTXV Audio " "VAE Loader"},
                    },
                    "14": {
                        "inputs": {"samples": ["29", 1], "audio_vae": ["13", 0]},
                        "class_type": "LTXVAudioVAEDecode",
                        "_meta": {"title": "🅛🅣🅧 LTXV Audio " "VAE Decode"},
                    },
                    "15": {
                        "inputs": {
                            "frame_rate": ["23", 0],
                            "loop_count": 0,
                            "filename_prefix": "AnimateDiff",
                            "format": "video/h264-mp4",
                            "pix_fmt": "yuv420p",
                            "crf": 19,
                            "save_metadata": True,
                            "trim_to_audio": False,
                            "pingpong": False,
                            "save_output": True,
                            "images": ["12", 0],
                            "audio": ["14", 0],
                        },
                        "class_type": "VHS_VideoCombine",
                        "_meta": {"title": "Video Combine " "🎥🅥🅗🅢"},
                    },
                    "17": {
                        "inputs": {
                            "skip_blocks": "29",
                            "model": ["28", 1],
                            "positive": ["22", 0],
                            "negative": ["22", 1],
                            "parameters": ["18", 0],
                        },
                        "class_type": "MultimodalGuider",
                        "_meta": {"title": "🅛🅣🅧 Multimodal " "Guider"},
                    },
                    "18": {
                        "inputs": {
                            "modality": "VIDEO",
                            "cfg": 3,
                            "stg": 0,
                            "rescale": 0,
                            "modality_scale": 3,
                            "parameters": ["19", 0],
                        },
                        "class_type": "GuiderParameters",
                        "_meta": {"title": "🅛🅣🅧 Guider " "Parameters"},
                    },
                    "19": {
                        "inputs": {
                            "modality": "AUDIO",
                            "cfg": 7,
                            "stg": 0,
                            "rescale": 0,
                            "modality_scale": 3,
                        },
                        "class_type": "GuiderParameters",
                        "_meta": {"title": "🅛🅣🅧 Guider " "Parameters"},
                    },
                    "21": {
                        "inputs": {"audioUI": "", "audio": ["14", 0]},
                        "class_type": "PreviewAudio",
                        "_meta": {"title": "PreviewAudio"},
                    },
                    "22": {
                        "inputs": {
                            "frame_rate": ["23", 0],
                            "positive": ["3", 0],
                            "negative": ["4", 0],
                        },
                        "class_type": "LTXVConditioning",
                        "_meta": {"title": "LTXVConditioning"},
                    },
                    "23": {
                        "inputs": {"value": 25},
                        "class_type": "FloatConstant",
                        "_meta": {"title": "Float Constant"},
                    },
                    "26": {
                        "inputs": {
                            "frames_number": ["27", 0],
                            "frame_rate": ["42", 0],
                            "batch_size": 1,
                        },
                        "class_type": "LTXVEmptyLatentAudio",
                        "_meta": {"title": "🅛🅣🅧 LTXV Empty " "Latent Audio"},
                    },
                    "27": {
                        "inputs": {"value": 105},
                        "class_type": "INTConstant",
                        "_meta": {"title": "INT Constant"},
                    },
                    "28": {
                        "inputs": {
                            "video_latent": ["43", 0],
                            "audio_latent": ["26", 0],
                            "model": ["44", 0],
                        },
                        "class_type": "LTXVConcatAVLatent",
                        "_meta": {"title": "🅛🅣🅧 LTXV Concat " "AV Latent"},
                    },
                    "29": {
                        "inputs": {"av_latent": ["41", 0], "model": ["28", 1]},
                        "class_type": "LTXVSeparateAVLatent",
                        "_meta": {"title": "🅛🅣🅧 LTXV " "Separate AV " "Latent"},
                    },
                    "41": {
                        "inputs": {
                            "noise": ["11", 0],
                            "guider": ["17", 0],
                            "sampler": ["8", 0],
                            "sigmas": ["9", 0],
                            "latent_image": ["28", 0],
                        },
                        "class_type": "SamplerCustomAdvanced",
                        "_meta": {"title": "SamplerCustomAdvanced"},
                    },
                    "42": {
                        "inputs": {"a": ["23", 0]},
                        "class_type": "CM_FloatToInt",
                        "_meta": {"title": "FloatToInt"},
                    },
                    "43": {
                        "inputs": {
                            "width": 768,
                            "height": 512,
                            "length": ["27", 0],
                            "batch_size": 1,
                        },
                        "class_type": "EmptyLTXVLatentVideo",
                        "_meta": {"title": "EmptyLTXVLatentVideo"},
                    },
                    "44": {
                        "inputs": {
                            "torch_compile": True,
                            "disable_backup": False,
                            "model": ["1", 0],
                        },
                        "class_type": "LTXVSequenceParallelMultiGPUPatcher",
                        "_meta": {"title": "LTXVSequenceParallelMultiGPUPatcher"},
                    },
                    "45": {
                        "inputs": {"frame_idx": 0, "strength": 1},
                        "class_type": "LTXVAddGuide",
                        "_meta": {"title": "LTXVAddGuide"},
                    },
                },
                "comfy_fork_version": "feature/av_inference@a6994ed1",
                "VHS_latentpreview": False,
                "VHS_latentpreviewrate": 0,
                "VHS_MetadataImage": True,
                "VHS_KeepIntermediate": True,
                "ue_links": [],
                "links_added_by_ue": [],
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
    video_length: int,
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
        DualCLIPLoader,
        LoadImage,
        NODE_CLASS_MAPPINGS,
        UNETLoader,
        VAEDecode,
    )

    import torch

    try:
        with torch.inference_mode():
            loadimage = LoadImage()
            loadimage_98 = loadimage.load_image(image=image_path)
            ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
            ksamplerselect_168 = ksamplerselect.EXECUTE_NORMALIZED(
                sampler_name="euler_ancestral"
            )
            primitiveint = NODE_CLASS_MAPPINGS["PrimitiveInt"]()
            primitiveint_185 = primitiveint.EXECUTE_NORMALIZED(value=video_length)
            randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
            node_199_noise_seed = prompt["199"]["inputs"]["noise_seed"] = (
                random.randint(1, 2**64)
            )
            randomnoise_199 = randomnoise.EXECUTE_NORMALIZED(
                noise_seed=node_199_noise_seed
            )
            unetloader = UNETLoader()
            unetloader_212 = unetloader.load_unet(
                unet_name="ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors",
                weight_dtype="default",
            )
            dualcliploader = DualCLIPLoader()
            dualcliploader_209 = dualcliploader.load_clip(
                clip_name1="gemma_3_12B_it_fp4_mixed.safetensors",
                clip_name2="ltx-2.3_text_projection_bf16.safetensors",
                type="ltxv",
                device="default",
            )
            loratagloader = NODE_CLASS_MAPPINGS["LoraTagLoader"]()
            loratagloader_230 = loratagloader.load_lora(
                text=prompt_text,
                model=get_value_at_index(unetloader_212, 0),
                clip=get_value_at_index(dualcliploader_209, 0),
            )
            cliptextencode = CLIPTextEncode()
            cliptextencode_203 = cliptextencode.encode(
                text=get_value_at_index(loratagloader_230, 2),
                clip=get_value_at_index(loratagloader_230, 1),
            )
            vaeloaderkj = NODE_CLASS_MAPPINGS["VAELoaderKJ"]()
            vaeloaderkj_210 = vaeloaderkj.load_vae(
                vae_name="LTX23_video_vae_bf16.safetensors",
                device="main_device",
                weight_dtype="bf16",
            )
            vaeloaderkj_211 = vaeloaderkj.load_vae(
                vae_name="LTX23_audio_vae_bf16.safetensors",
                device="main_device",
                weight_dtype="bf16",
            )
            intconstant = NODE_CLASS_MAPPINGS["INTConstant"]()
            intconstant_237 = intconstant.get_value(value=fps)
            getimagesizeandcount = NODE_CLASS_MAPPINGS["GetImageSizeAndCount"]()
            ltxvpreprocess = NODE_CLASS_MAPPINGS["LTXVPreprocess"]()
            emptyltxvlatentvideo = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()
            ltxvimgtovideoinplace = NODE_CLASS_MAPPINGS["LTXVImgToVideoInplace"]()
            comfynumberconvert = NODE_CLASS_MAPPINGS["ComfyNumberConvert"]()
            ltxvemptylatentaudio = NODE_CLASS_MAPPINGS["LTXVEmptyLatentAudio"]()
            ltxvconcatavlatent = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
            ltxvscheduler = NODE_CLASS_MAPPINGS["LTXVScheduler"]()
            conditioningzeroout = ConditioningZeroOut()
            ltxvconditioning = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
            cfgguider = NODE_CLASS_MAPPINGS["CFGGuider"]()
            samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
            ltxvseparateavlatent = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
            ltxvaudiovaedecode = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
            vaedecode = VAEDecode()
            vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
            for q in range(1):
                getimagesizeandcount_233 = getimagesizeandcount.getsize(
                    image=get_value_at_index(loadimage_98, 0)
                )
                ltxvpreprocess_188 = ltxvpreprocess.EXECUTE_NORMALIZED(
                    img_compression=33,
                    image=get_value_at_index(getimagesizeandcount_233, 0),
                )
                emptyltxvlatentvideo_189 = emptyltxvlatentvideo.EXECUTE_NORMALIZED(
                    width=get_value_at_index(getimagesizeandcount_233, 1),
                    height=get_value_at_index(getimagesizeandcount_233, 2),
                    length=get_value_at_index(primitiveint_185, 0),
                    batch_size=1,
                )
                ltxvimgtovideoinplace_190 = ltxvimgtovideoinplace.EXECUTE_NORMALIZED(
                    strength=1,
                    bypass=False,
                    vae=get_value_at_index(vaeloaderkj_210, 0),
                    image=get_value_at_index(ltxvpreprocess_188, 0),
                    latent=get_value_at_index(emptyltxvlatentvideo_189, 0),
                )
                comfynumberconvert_236 = comfynumberconvert.EXECUTE_NORMALIZED(
                    value=get_value_at_index(intconstant_237, 0)
                )
                ltxvemptylatentaudio_174 = ltxvemptylatentaudio.EXECUTE_NORMALIZED(
                    frames_number=get_value_at_index(primitiveint_185, 0),
                    frame_rate=get_value_at_index(comfynumberconvert_236, 1),
                    batch_size=1,
                    audio_vae=get_value_at_index(vaeloaderkj_211, 0),
                )
                ltxvconcatavlatent_182 = ltxvconcatavlatent.EXECUTE_NORMALIZED(
                    video_latent=get_value_at_index(ltxvimgtovideoinplace_190, 0),
                    audio_latent=get_value_at_index(ltxvemptylatentaudio_174, 0),
                )
                ltxvscheduler_169 = ltxvscheduler.EXECUTE_NORMALIZED(
                    steps=8,
                    max_shift=2.05,
                    base_shift=0.95,
                    stretch=True,
                    terminal=0.1,
                    latent=get_value_at_index(ltxvconcatavlatent_182, 0),
                )
                conditioningzeroout_226 = conditioningzeroout.zero_out(
                    conditioning=get_value_at_index(cliptextencode_203, 0)
                )
                ltxvconditioning_201 = ltxvconditioning.EXECUTE_NORMALIZED(
                    frame_rate=get_value_at_index(comfynumberconvert_236, 0),
                    positive=get_value_at_index(cliptextencode_203, 0),
                    negative=get_value_at_index(conditioningzeroout_226, 0),
                )
                cfgguider_193 = cfgguider.EXECUTE_NORMALIZED(
                    cfg=1,
                    model=get_value_at_index(loratagloader_230, 0),
                    positive=get_value_at_index(ltxvconditioning_201, 0),
                    negative=get_value_at_index(ltxvconditioning_201, 1),
                )
                samplercustomadvanced_175 = samplercustomadvanced.EXECUTE_NORMALIZED(
                    noise=get_value_at_index(randomnoise_199, 0),
                    guider=get_value_at_index(cfgguider_193, 0),
                    sampler=get_value_at_index(ksamplerselect_168, 0),
                    sigmas=get_value_at_index(ltxvscheduler_169, 0),
                    latent_image=get_value_at_index(ltxvconcatavlatent_182, 0),
                )
                ltxvseparateavlatent_217 = ltxvseparateavlatent.EXECUTE_NORMALIZED(
                    av_latent=get_value_at_index(samplercustomadvanced_175, 1)
                )
                ltxvaudiovaedecode_213 = ltxvaudiovaedecode.EXECUTE_NORMALIZED(
                    samples=get_value_at_index(ltxvseparateavlatent_217, 1),
                    audio_vae=get_value_at_index(vaeloaderkj_211, 0),
                )
                vaedecode_214 = vaedecode.decode(
                    samples=get_value_at_index(ltxvseparateavlatent_217, 0),
                    vae=get_value_at_index(vaeloaderkj_210, 0),
                )
                vhs_videocombine_238 = vhs_videocombine.combine_video(
                    frame_rate=get_value_at_index(comfynumberconvert_236, 0),
                    loop_count=0,
                    filename_prefix="LTX23",
                    format="video/h264-mp4",
                    pix_fmt="yuv420p",
                    crf=19,
                    save_metadata=False,
                    trim_to_audio=False,
                    pingpong=False,
                    save_output=True,
                    images=get_value_at_index(vaedecode_214, 0),
                    audio=get_value_at_index(ltxvaudiovaedecode_213, 0),
                    unique_id=13707171288602021803,
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                )

                import folder_paths
                import os
                import glob as glob_module

                output_dir = folder_paths.get_output_directory()

                pattern = os.path.join(output_dir, "LTX23_*.png")
                files = sorted(glob_module.glob(pattern), key=os.path.getmtime)
                if files:
                    return files[-1]
                return None
    finally:
        cleanup_comfyui_runtime(unload_models=unload_models)
