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
        "9": {
            "inputs": {"filename_prefix": "zit", "images": ["57:8", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
        },
        "57:29": {
            "inputs": {"vae_name": "ae.safetensors"},
            "class_type": "VAELoader",
            "_meta": {"title": "Load VAE"},
        },
        "57:33": {
            "inputs": {"conditioning": ["57:27", 0]},
            "class_type": "ConditioningZeroOut",
            "_meta": {"title": "ConditioningZeroOut"},
        },
        "57:8": {
            "inputs": {"samples": ["57:3", 0], "vae": ["57:29", 0]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
        },
        "57:11": {
            "inputs": {"shift": 3, "model": ["57:66", 0]},
            "class_type": "ModelSamplingAuraFlow",
            "_meta": {"title": "ModelSamplingAuraFlow"},
        },
        "57:3": {
            "inputs": {
                "seed": 295206784373826,
                "steps": 8,
                "cfg": 1,
                "sampler_name": "res_multistep",
                "scheduler": "simple",
                "denoise": 1,
                "model": ["57:11", 0],
                "positive": ["57:27", 0],
                "negative": ["57:33", 0],
                "latent_image": ["57:13", 0],
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
        },
        "57:13": {
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
            "class_type": "EmptySD3LatentImage",
            "_meta": {"title": "EmptySD3LatentImage"},
        },
        "57:28": {
            "inputs": {
                "unet_name": "ZImageTurbo/z_image_turbo_bf16.safetensors",
                "weight_dtype": "default",
            },
            "class_type": "UNETLoader",
            "_meta": {"title": "Load Diffusion Model"},
        },
        "57:30": {
            "inputs": {
                "clip_name": "qwen_3_4b.safetensors",
                "type": "lumina2",
                "device": "default",
            },
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load CLIP"},
        },
        "57:66": {
            "inputs": {
                "text": "Hatsune Miku with red outfit.\n"
                "<lora:76N0PGDVMCA64NA75C2NW7V600:1>",
                "model": ["57:28", 0],
                "clip": ["57:30", 0],
            },
            "class_type": "LoraTagLoader",
            "_meta": {"title": "Load LoRA Tag"},
        },
        "57:27": {
            "inputs": {"text": ["57:66", 2], "clip": ["57:66", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"},
        },
    }


def build_extra_pnginfo() -> dict[str, Any] | None:
    return {
        "workflow": {
            "id": "9ae6082b-c7f4-433c-9971-7a8f65a3ea65",
            "revision": 0,
            "last_node_id": 66,
            "last_link_id": 99,
            "nodes": [
                {
                    "id": 9,
                    "type": "SaveImage",
                    "pos": [569.9998957637683, 199.99998755061938],
                    "size": [780, 660],
                    "flags": {},
                    "order": 1,
                    "mode": 0,
                    "inputs": [{"name": "images", "type": "IMAGE", "link": 62}],
                    "outputs": [],
                    "properties": {
                        "Node name for S&R": "SaveImage",
                        "cnr_id": "comfy-core",
                        "ver": "0.3.64",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                    },
                    "widgets_values": ["zit"],
                },
                {
                    "id": 57,
                    "type": "f2fdebf6-dfaf-43b6-9eb2-7f70613cfdc1",
                    "pos": [130, 200],
                    "size": [400, 470],
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "inputs": [
                        {
                            "label": "prompt",
                            "name": "text",
                            "type": "STRING",
                            "widget": {"name": "text"},
                            "link": None,
                        }
                    ],
                    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [62]}],
                    "properties": {
                        "cnr_id": "comfy-core",
                        "ver": "0.3.73",
                        "enableTabs": False,
                        "tabWidth": 65,
                        "tabXOffset": 10,
                        "hasSecondTab": False,
                        "secondTabText": "Send Back",
                        "secondTabOffset": 80,
                        "secondTabWidth": 65,
                        "proxyWidgetErrorQuarantine": [
                            {
                                "originalEntry": ["3", "control_after_generate"],
                                "reason": "missingSubgraphInput",
                                "attemptedAtVersion": 1,
                            }
                        ],
                        "previewExposures": [
                            {
                                "name": "$$canvas-image-preview",
                                "sourceNodeId": "3",
                                "sourcePreviewName": "$$canvas-image-preview",
                            }
                        ],
                    },
                    "widgets_values": [
                        "Hatsune Miku with red outfit.\n"
                        "<lora:76N0PGDVMCA64NA75C2NW7V600:1>",
                        1024,
                        1024,
                        295206784373826,
                        8,
                        "ZImageTurbo/z_image_turbo_bf16.safetensors",
                        "qwen_3_4b.safetensors",
                        "ae.safetensors",
                    ],
                },
            ],
            "links": [[62, 57, 0, 9, 0, "IMAGE"]],
            "groups": [],
            "definitions": {
                "subgraphs": [
                    {
                        "id": "f2fdebf6-dfaf-43b6-9eb2-7f70613cfdc1",
                        "version": 1,
                        "state": {
                            "lastGroupId": 4,
                            "lastNodeId": 66,
                            "lastLinkId": 99,
                            "lastRerouteId": 0,
                        },
                        "revision": 0,
                        "config": {},
                        "name": "Text to Image (Z-Image-Turbo)",
                        "inputNode": {"id": -10, "bounding": [-560, 480, 128, 208]},
                        "outputNode": {"id": -20, "bounding": [1670, 320, 128, 68]},
                        "inputs": [
                            {
                                "id": "fb178669-e742-4a53-8a69-7df59834dfd8",
                                "name": "text",
                                "type": "STRING",
                                "linkIds": [98],
                                "label": "prompt",
                                "pos": [-456, 504],
                            },
                            {
                                "id": "dd780b3c-23e9-46ff-8469-156008f42e5a",
                                "name": "width",
                                "type": "INT",
                                "linkIds": [35],
                                "pos": [-456, 524],
                            },
                            {
                                "id": "7b08d546-6bb0-4ef9-82e9-ffae5e1ee6bc",
                                "name": "height",
                                "type": "INT",
                                "linkIds": [36],
                                "pos": [-456, 544],
                            },
                            {
                                "id": "f77677f7-6bf6-4c19-a71f-c4a553d5981e",
                                "name": "seed",
                                "type": "INT",
                                "linkIds": [71],
                                "pos": [-456, 564],
                            },
                            {
                                "id": "ef9a9fb1-5983-4bc9-a60b-cf5aec48bff1",
                                "name": "steps",
                                "type": "INT",
                                "linkIds": [72],
                                "pos": [-456, 584],
                            },
                            {
                                "id": "a20a1b30-785f-4a04-bb6d-3d61adab9764",
                                "name": "unet_name",
                                "type": "COMBO",
                                "linkIds": [73],
                                "pos": [-456, 604],
                            },
                            {
                                "id": "4af8fc2b-4655-4086-8240-45f8cb38c6f6",
                                "name": "clip_name",
                                "type": "COMBO",
                                "linkIds": [74],
                                "pos": [-456, 624],
                            },
                            {
                                "id": "4d518693-2807-439c-9cb6-cffd23ccba2c",
                                "name": "vae_name",
                                "type": "COMBO",
                                "linkIds": [75],
                                "pos": [-456, 644],
                            },
                        ],
                        "outputs": [
                            {
                                "id": "1fa72a21-ce00-4952-814e-1f2ffbe87d1d",
                                "name": "IMAGE",
                                "type": "IMAGE",
                                "linkIds": [16],
                                "localized_name": "IMAGE",
                                "pos": [1694, 344],
                            }
                        ],
                        "widgets": [],
                        "nodes": [
                            {
                                "id": 29,
                                "type": "VAELoader",
                                "pos": [-334.74787838765036, 642.2061579161602],
                                "size": [270, 106.64999389648438],
                                "flags": {},
                                "order": 6,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "vae_name",
                                        "name": "vae_name",
                                        "type": "COMBO",
                                        "widget": {"name": "vae_name"},
                                        "link": 75,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "VAE",
                                        "name": "VAE",
                                        "type": "VAE",
                                        "links": [27],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "VAELoader",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.3.73",
                                    "models": [
                                        {
                                            "name": "ae.safetensors",
                                            "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
                                            "directory": "vae",
                                        }
                                    ],
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                },
                                "widgets_values": ["ae.safetensors"],
                            },
                            {
                                "id": 33,
                                "type": "ConditioningZeroOut",
                                "pos": [629.9998362150791, 959.999861458308],
                                "size": [225, 72],
                                "flags": {},
                                "order": 8,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "conditioning",
                                        "name": "conditioning",
                                        "type": "CONDITIONING",
                                        "link": 32,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "CONDITIONING",
                                        "name": "CONDITIONING",
                                        "type": "CONDITIONING",
                                        "links": [33],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "ConditioningZeroOut",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.3.73",
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                },
                                "widgets_values": [],
                            },
                            {
                                "id": 8,
                                "type": "VAEDecode",
                                "pos": [1319.9997837897204, 229.99998088352976],
                                "size": [225, 96],
                                "flags": {},
                                "order": 1,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "samples",
                                        "name": "samples",
                                        "type": "LATENT",
                                        "link": 14,
                                    },
                                    {
                                        "localized_name": "vae",
                                        "name": "vae",
                                        "type": "VAE",
                                        "link": 27,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "IMAGE",
                                        "name": "IMAGE",
                                        "type": "IMAGE",
                                        "slot_index": 0,
                                        "links": [16],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "VAEDecode",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.3.64",
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                },
                                "widgets_values": [],
                            },
                            {
                                "id": 11,
                                "type": "ModelSamplingAuraFlow",
                                "pos": [949.9997988929108, 229.99998088352976],
                                "size": [310, 104],
                                "flags": {},
                                "order": 2,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "model",
                                        "name": "model",
                                        "type": "MODEL",
                                        "link": 96,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "MODEL",
                                        "name": "MODEL",
                                        "type": "MODEL",
                                        "slot_index": 0,
                                        "links": [13],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "ModelSamplingAuraFlow",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.3.64",
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                },
                                "widgets_values": [3],
                            },
                            {
                                "id": 3,
                                "type": "KSampler",
                                "pos": [949.9997988929108, 399.9999517059391],
                                "size": [320, 341.3333435058594],
                                "flags": {},
                                "order": 0,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "model",
                                        "name": "model",
                                        "type": "MODEL",
                                        "link": 13,
                                    },
                                    {
                                        "localized_name": "positive",
                                        "name": "positive",
                                        "type": "CONDITIONING",
                                        "link": 30,
                                    },
                                    {
                                        "localized_name": "negative",
                                        "name": "negative",
                                        "type": "CONDITIONING",
                                        "link": 33,
                                    },
                                    {
                                        "localized_name": "latent_image",
                                        "name": "latent_image",
                                        "type": "LATENT",
                                        "link": 17,
                                    },
                                    {
                                        "localized_name": "seed",
                                        "name": "seed",
                                        "type": "INT",
                                        "widget": {"name": "seed"},
                                        "link": 71,
                                    },
                                    {
                                        "localized_name": "steps",
                                        "name": "steps",
                                        "type": "INT",
                                        "widget": {"name": "steps"},
                                        "link": 72,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "LATENT",
                                        "name": "LATENT",
                                        "type": "LATENT",
                                        "slot_index": 0,
                                        "links": [14],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "KSampler",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.3.64",
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                },
                                "widgets_values": [
                                    504827585819016,
                                    "randomize",
                                    8,
                                    1,
                                    "res_multistep",
                                    "simple",
                                    1,
                                ],
                            },
                            {
                                "id": 13,
                                "type": "EmptySD3LatentImage",
                                "pos": [-324.74781410264677, 882.2061548583707],
                                "size": [260, 168],
                                "flags": {},
                                "order": 3,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "width",
                                        "name": "width",
                                        "type": "INT",
                                        "widget": {"name": "width"},
                                        "link": 35,
                                    },
                                    {
                                        "localized_name": "height",
                                        "name": "height",
                                        "type": "INT",
                                        "widget": {"name": "height"},
                                        "link": 36,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "LATENT",
                                        "name": "LATENT",
                                        "type": "LATENT",
                                        "slot_index": 0,
                                        "links": [17],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "EmptySD3LatentImage",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.3.64",
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                },
                                "widgets_values": [1024, 1024, 1],
                            },
                            {
                                "id": 28,
                                "type": "UNETLoader",
                                "pos": [-334.74787838765036, 222.2062256018836],
                                "size": [270, 108],
                                "flags": {},
                                "order": 5,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "unet_name",
                                        "name": "unet_name",
                                        "type": "COMBO",
                                        "widget": {"name": "unet_name"},
                                        "link": 73,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "MODEL",
                                        "name": "MODEL",
                                        "type": "MODEL",
                                        "links": [94],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "UNETLoader",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.3.73",
                                    "models": [
                                        {
                                            "name": "z_image_turbo_bf16.safetensors",
                                            "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors",
                                            "directory": "diffusion_models",
                                        }
                                    ],
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                },
                                "widgets_values": [
                                    "ZImageTurbo/z_image_turbo_bf16.safetensors",
                                    "default",
                                ],
                            },
                            {
                                "id": 30,
                                "type": "CLIPLoader",
                                "pos": [-334.74787838765036, 414.870256108062],
                                "size": [270, 141.3333282470703],
                                "flags": {},
                                "order": 7,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "clip_name",
                                        "name": "clip_name",
                                        "type": "COMBO",
                                        "widget": {"name": "clip_name"},
                                        "link": 74,
                                    }
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "CLIP",
                                        "name": "CLIP",
                                        "type": "CLIP",
                                        "links": [95],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "CLIPLoader",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.3.73",
                                    "models": [
                                        {
                                            "name": "qwen_3_4b.safetensors",
                                            "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors",
                                            "directory": "text_encoders",
                                        }
                                    ],
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                },
                                "widgets_values": [
                                    "qwen_3_4b.safetensors",
                                    "lumina2",
                                    "default",
                                ],
                            },
                            {
                                "id": 66,
                                "type": "LoraTagLoader",
                                "pos": [414.61777626066913, -80.23113444711362],
                                "size": [400, 200],
                                "flags": {},
                                "order": 9,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "model",
                                        "name": "model",
                                        "type": "MODEL",
                                        "link": 94,
                                    },
                                    {
                                        "localized_name": "clip",
                                        "name": "clip",
                                        "type": "CLIP",
                                        "link": 95,
                                    },
                                    {
                                        "localized_name": "text",
                                        "name": "text",
                                        "type": "STRING",
                                        "widget": {"name": "text"},
                                        "link": 98,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "MODEL",
                                        "name": "MODEL",
                                        "type": "MODEL",
                                        "links": [96],
                                    },
                                    {
                                        "localized_name": "CLIP",
                                        "name": "CLIP",
                                        "type": "CLIP",
                                        "links": [97],
                                    },
                                    {
                                        "localized_name": "STRING",
                                        "name": "STRING",
                                        "type": "STRING",
                                        "links": [99],
                                    },
                                ],
                                "properties": {"Node name for S&R": "LoraTagLoader"},
                                "widgets_values": [
                                    "remove "
                                    "bicycle.\n"
                                    "<lora:76N0PGDVMCA64NA75C2NW7V600:1>"
                                ],
                            },
                            {
                                "id": 27,
                                "type": "CLIPTextEncode",
                                "pos": [412.8800991182144, 231.2876937533725],
                                "size": [411.353279595143, 88],
                                "flags": {"collapsed": False},
                                "order": 4,
                                "mode": 0,
                                "inputs": [
                                    {
                                        "localized_name": "clip",
                                        "name": "clip",
                                        "type": "CLIP",
                                        "link": 97,
                                    },
                                    {
                                        "localized_name": "text",
                                        "name": "text",
                                        "type": "STRING",
                                        "widget": {"name": "text"},
                                        "link": 99,
                                    },
                                ],
                                "outputs": [
                                    {
                                        "localized_name": "CONDITIONING",
                                        "name": "CONDITIONING",
                                        "type": "CONDITIONING",
                                        "links": [30, 32],
                                    }
                                ],
                                "properties": {
                                    "Node name for S&R": "CLIPTextEncode",
                                    "cnr_id": "comfy-core",
                                    "ver": "0.3.73",
                                    "enableTabs": False,
                                    "tabWidth": 65,
                                    "tabXOffset": 10,
                                    "hasSecondTab": False,
                                    "secondTabText": "Send Back",
                                    "secondTabOffset": 80,
                                    "secondTabWidth": 65,
                                },
                                "widgets_values": [""],
                            },
                        ],
                        "groups": [],
                        "links": [
                            {
                                "id": 32,
                                "origin_id": 27,
                                "origin_slot": 0,
                                "target_id": 33,
                                "target_slot": 0,
                                "type": "CONDITIONING",
                            },
                            {
                                "id": 14,
                                "origin_id": 3,
                                "origin_slot": 0,
                                "target_id": 8,
                                "target_slot": 0,
                                "type": "LATENT",
                            },
                            {
                                "id": 27,
                                "origin_id": 29,
                                "origin_slot": 0,
                                "target_id": 8,
                                "target_slot": 1,
                                "type": "VAE",
                            },
                            {
                                "id": 13,
                                "origin_id": 11,
                                "origin_slot": 0,
                                "target_id": 3,
                                "target_slot": 0,
                                "type": "MODEL",
                            },
                            {
                                "id": 30,
                                "origin_id": 27,
                                "origin_slot": 0,
                                "target_id": 3,
                                "target_slot": 1,
                                "type": "CONDITIONING",
                            },
                            {
                                "id": 33,
                                "origin_id": 33,
                                "origin_slot": 0,
                                "target_id": 3,
                                "target_slot": 2,
                                "type": "CONDITIONING",
                            },
                            {
                                "id": 17,
                                "origin_id": 13,
                                "origin_slot": 0,
                                "target_id": 3,
                                "target_slot": 3,
                                "type": "LATENT",
                            },
                            {
                                "id": 16,
                                "origin_id": 8,
                                "origin_slot": 0,
                                "target_id": -20,
                                "target_slot": 0,
                                "type": "IMAGE",
                            },
                            {
                                "id": 35,
                                "origin_id": -10,
                                "origin_slot": 1,
                                "target_id": 13,
                                "target_slot": 0,
                                "type": "INT",
                            },
                            {
                                "id": 36,
                                "origin_id": -10,
                                "origin_slot": 2,
                                "target_id": 13,
                                "target_slot": 1,
                                "type": "INT",
                            },
                            {
                                "id": 71,
                                "origin_id": -10,
                                "origin_slot": 3,
                                "target_id": 3,
                                "target_slot": 4,
                                "type": "INT",
                            },
                            {
                                "id": 72,
                                "origin_id": -10,
                                "origin_slot": 4,
                                "target_id": 3,
                                "target_slot": 5,
                                "type": "INT",
                            },
                            {
                                "id": 73,
                                "origin_id": -10,
                                "origin_slot": 5,
                                "target_id": 28,
                                "target_slot": 0,
                                "type": "COMBO",
                            },
                            {
                                "id": 74,
                                "origin_id": -10,
                                "origin_slot": 6,
                                "target_id": 30,
                                "target_slot": 0,
                                "type": "COMBO",
                            },
                            {
                                "id": 75,
                                "origin_id": -10,
                                "origin_slot": 7,
                                "target_id": 29,
                                "target_slot": 0,
                                "type": "COMBO",
                            },
                            {
                                "id": 94,
                                "origin_id": 28,
                                "origin_slot": 0,
                                "target_id": 66,
                                "target_slot": 0,
                                "type": "MODEL",
                            },
                            {
                                "id": 95,
                                "origin_id": 30,
                                "origin_slot": 0,
                                "target_id": 66,
                                "target_slot": 1,
                                "type": "CLIP",
                            },
                            {
                                "id": 96,
                                "origin_id": 66,
                                "origin_slot": 0,
                                "target_id": 11,
                                "target_slot": 0,
                                "type": "MODEL",
                            },
                            {
                                "id": 97,
                                "origin_id": 66,
                                "origin_slot": 1,
                                "target_id": 27,
                                "target_slot": 0,
                                "type": "CLIP",
                            },
                            {
                                "id": 98,
                                "origin_id": -10,
                                "origin_slot": 0,
                                "target_id": 66,
                                "target_slot": 2,
                                "type": "STRING",
                            },
                            {
                                "id": 99,
                                "origin_id": 66,
                                "origin_slot": 2,
                                "target_id": 27,
                                "target_slot": 1,
                                "type": "STRING",
                            },
                        ],
                        "extra": {"workflowRendererVersion": "LG"},
                    }
                ]
            },
            "config": {},
            "extra": {
                "frontendVersion": "1.46.3",
                "workflowRendererVersion": "LG",
                "VHS_latentpreview": False,
                "VHS_latentpreviewrate": 0,
                "VHS_MetadataImage": True,
                "VHS_KeepIntermediate": True,
                "ds": {
                    "scale": 1.1976443661971836,
                    "offset": [152.48722241135061, 18.46349728899659],
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
    seed: int,
    width: int,
    height: int,
    unload_models: bool | None = None,
):
    bootstrap_comfyui_runtime()
    add_extra_model_paths()
    import_custom_nodes()

    from nodes import (
        CLIPLoader,
        CLIPTextEncode,
        ConditioningZeroOut,
        KSampler,
        NODE_CLASS_MAPPINGS,
        SaveImage,
        UNETLoader,
        VAEDecode,
        VAELoader,
    )
    import folder_paths
    import torch

    try:
        with torch.inference_mode():
            vaeloader = VAELoader()
            vaeloader_57_29 = vaeloader.load_vae(vae_name="ae.safetensors")
            emptysd3latentimage = NODE_CLASS_MAPPINGS["EmptySD3LatentImage"]()
            emptysd3latentimage_57_13 = emptysd3latentimage.EXECUTE_NORMALIZED(
                width=width, height=height, batch_size=1
            )
            unetloader = UNETLoader()
            unetloader_57_28 = unetloader.load_unet(
                unet_name="ZImageTurbo/z_image_turbo_bf16.safetensors",
                weight_dtype="default",
            )
            cliploader = CLIPLoader()
            cliploader_57_30 = cliploader.load_clip(
                clip_name="qwen_3_4b.safetensors", type="lumina2", device="default"
            )
            loratagloader = NODE_CLASS_MAPPINGS["LoraTagLoader"]()
            loratagloader_57_66 = loratagloader.load_lora(
                text=prompt_text,
                model=get_value_at_index(unetloader_57_28, 0),
                clip=get_value_at_index(cliploader_57_30, 0),
            )
            cliptextencode = CLIPTextEncode()
            cliptextencode_57_27 = cliptextencode.encode(
                text=get_value_at_index(loratagloader_57_66, 2),
                clip=get_value_at_index(loratagloader_57_66, 1),
            )
            modelsamplingauraflow = NODE_CLASS_MAPPINGS["ModelSamplingAuraFlow"]()
            conditioningzeroout = ConditioningZeroOut()
            ksampler = KSampler()
            vaedecode = VAEDecode()
            saveimage = SaveImage()
            for q in range(1):
                modelsamplingauraflow_57_11 = modelsamplingauraflow.patch_aura(
                    shift=3, model=get_value_at_index(loratagloader_57_66, 0)
                )
                conditioningzeroout_57_33 = conditioningzeroout.zero_out(
                    conditioning=get_value_at_index(cliptextencode_57_27, 0)
                )
                node_57_3_seed = prompt["57:3"]["inputs"]["seed"] = seed
                ksampler_57_3 = ksampler.sample(
                    seed=node_57_3_seed,
                    steps=8,
                    cfg=1,
                    sampler_name="res_multistep",
                    scheduler="simple",
                    denoise=1,
                    model=get_value_at_index(modelsamplingauraflow_57_11, 0),
                    positive=get_value_at_index(cliptextencode_57_27, 0),
                    negative=get_value_at_index(conditioningzeroout_57_33, 0),
                    latent_image=get_value_at_index(emptysd3latentimage_57_13, 0),
                )
                vaedecode_57_8 = vaedecode.decode(
                    samples=get_value_at_index(ksampler_57_3, 0),
                    vae=get_value_at_index(vaeloader_57_29, 0),
                )
                saveimage_9 = saveimage.save_images(
                    filename_prefix="zit",
                    images=get_value_at_index(vaedecode_57_8, 0),
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                )
                output_dir = folder_paths.get_output_directory()
                import os
                import glob as glob_module

                pattern = os.path.join(output_dir, "zit_*.png")
                files = sorted(glob_module.glob(pattern), key=os.path.getmtime)
                if files:
                    return files[-1]
                return None
    finally:
        cleanup_comfyui_runtime(unload_models=unload_models)
