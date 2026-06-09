import os
import subprocess
import json


def get_video_info(video_path: str) -> str:
    if not video_path or not os.path.isfile(video_path):
        return ""

    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "v:0",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return ""
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]

        fps_str = stream.get("r_frame_rate", "0/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        else:
            fps = float(fps_str)

        width = stream.get("width", 0)
        height = stream.get("height", 0)
        nb_frames = stream.get("nb_frames", "N/A")

        return f"FPS: {fps:.2f} | Frames: {nb_frames} | {width}x{height}"
    except Exception:
        return ""
