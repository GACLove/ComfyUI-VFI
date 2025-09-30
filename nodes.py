"""ComfyUI nodes for Video Frame Interpolation using RIFE"""

import os
import subprocess
import sys

import torch

from .rife.rife_comfyui_wrapper import RIFEWrapper

try:
    import comfy.utils
    import folder_paths
except ImportError:
    folder_paths = None
    comfy = None


MODEL_CACHE = {}


class RIFEInterpolation:
    """
    ComfyUI node for RIFE (Real-Time Intermediate Flow Estimation) video frame interpolation.
    Takes a sequence of images and interpolates frames to achieve a target frame rate.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_fps": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.1,
                        "display": "number",
                        "tooltip": "Source video frame rate",
                    },
                ),
                "target_fps": (
                    "FLOAT",
                    {
                        "default": 60.0,
                        "min": 1.0,
                        "max": 240.0,
                        "step": 0.1,
                        "display": "number",
                        "tooltip": "Target frame rate after interpolation",
                    },
                ),
                "scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.25,
                        "max": 4.0,
                        "step": 0.25,
                        "display": "number",
                        "tooltip": "Processing scale factor. Lower values process faster but may reduce quality",
                    },
                ),
            },
            "optional": {
                "model_name": (
                    ["flownet.pkl"],
                    {"default": "flownet.pkl", "tooltip": "RIFE model to use for interpolation"},
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 32,
                        "step": 1,
                        "display": "number",
                        "tooltip": "Number of frames to process in parallel. Higher values are faster but use more VRAM",
                    },
                ),
                "use_fp16": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use half precision (FP16) for faster inference and lower VRAM usage. Requires CUDA GPU",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "interpolate"

    CATEGORY = "image/animation"

    DESCRIPTION = "Interpolate video frames using RIFE (Real-Time Intermediate Flow Estimation) to increase frame rate"

    def interpolate(
        self, images, source_fps, target_fps, scale, model_name="flownet.pkl", batch_size=8, use_fp16=False
    ):
        # Validate inputs
        if images is None or len(images) == 0:
            raise ValueError("No images provided")

        if len(images.shape) != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected image tensor shape [N, H, W, 3], got {images.shape}")

        if source_fps <= 0 or target_fps <= 0:
            raise ValueError("Frame rates must be positive")

        if scale <= 0:
            raise ValueError("Scale must be positive")

        # If source and target fps are the same, return original
        if abs(source_fps - target_fps) < 0.01:
            return (images,)

        # Get or load model
        model = self._get_or_load_model(model_name, use_fp16=use_fp16)

        duration = len(images) / source_fps
        total_target_frames = int(duration * target_fps)

        pbar = None
        if comfy and hasattr(comfy, "utils"):
            pbar = comfy.utils.ProgressBar(total_target_frames)

        def progress_callback(current, total):
            if pbar:
                pbar.update_absolute(current, total)

        # Use autocast context for mixed precision
        autocast_enabled = use_fp16 and torch.cuda.is_available()
        autocast_context = torch.amp.autocast("cuda") if autocast_enabled else torch.nullcontext()

        try:
            with autocast_context:
                interpolated_images = model.interpolate_frames(
                    images=images,
                    source_fps=source_fps,
                    target_fps=target_fps,
                    scale=scale,
                    progress_callback=progress_callback,
                    batch_size=batch_size,
                )

            return (interpolated_images,)

        except Exception as e:
            raise RuntimeError(f"Frame interpolation failed: {str(e)}")

    def _get_or_load_model(self, model_name, use_fp16=False):
        """Load model from cache or disk"""
        global MODEL_CACHE

        # Create cache key with fp16 flag
        cache_key = f"{model_name}_fp16" if use_fp16 else model_name

        if cache_key in MODEL_CACHE:
            return MODEL_CACHE[cache_key]

        # Look for model in multiple locations
        model_paths = [
            os.path.join(os.path.dirname(__file__), "rife", "train_log", model_name),
            os.path.join(os.path.dirname(__file__), "models", model_name),
        ]

        # Add ComfyUI model directory if available
        if folder_paths and hasattr(folder_paths, "models_dir"):
            model_paths.insert(1, os.path.join(folder_paths.models_dir, "rife", model_name))

        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            # Try to download the model automatically
            print(f"RIFE model '{model_name}' not found. Attempting to download...")

            # Default download location
            download_target = os.path.join(os.path.dirname(__file__), "rife", "train_log")

            try:
                # Run the download script
                download_script = os.path.join(os.path.dirname(__file__), "rife", "download_rife.py")

                if os.path.exists(download_script):
                    result = subprocess.run(
                        [sys.executable, download_script, download_target], capture_output=True, text=True
                    )

                    if result.returncode == 0:
                        print("Model downloaded successfully!")
                        # Check if model now exists
                        model_path = os.path.join(download_target, model_name)
                        if not os.path.exists(model_path):
                            raise FileNotFoundError(
                                f"Model download completed but '{model_name}' not found at expected location."
                            )
                    else:
                        raise RuntimeError(f"Model download failed: {result.stderr}")
                else:
                    raise FileNotFoundError(
                        f"Download script not found at {download_script}. "
                        f"Please manually download the model and place it in one of these locations:\n"
                        + "\n".join(f"  - {p}" for p in model_paths)
                    )

            except Exception as e:
                raise RuntimeError(
                    f"Failed to automatically download RIFE model: {str(e)}\n"
                    f"Please manually download the model and place it in one of these locations:\n"
                    + "\n".join(f"  - {p}" for p in model_paths)
                )

        # Load model
        print(f"Loading RIFE model from: {model_path}")
        model = RIFEWrapper(model_path, use_fp16=use_fp16)
        MODEL_CACHE[cache_key] = model

        return model

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


class CalculateLoadedFPS:
    """
    计算加载后的FPS，根据原始FPS和每n帧选择一帧的参数
    Calculate loaded FPS based on source FPS and select_every_nth parameter
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_fps": (
                    "FLOAT",
                    {
                        "default": 24,
                        "min": 0.1,
                        "max": 160.0,
                        "step": 0.1,
                        "display": "number",
                        "tooltip": "Source video frame rate",
                    },
                ),
                "select_every_nth": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "number",
                        "tooltip": "Select every Nth frame (from VideoHelperSuite)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("loaded_fps",)

    FUNCTION = "calculate_fps"

    CATEGORY = "image/animation"

    DESCRIPTION = "Calculate loaded FPS after frame selection (source_fps / select_every_nth)"

    def calculate_fps(self, source_fps, select_every_nth):
        # 验证输入
        if source_fps <= 0:
            raise ValueError("source_fps must be positive")
        if select_every_nth <= 0:
            raise ValueError("select_every_nth must be positive")

        # 计算加载后的FPS
        loaded_fps = source_fps / select_every_nth
        return (loaded_fps,)


# ComfyUI node mappings
NODE_CLASS_MAPPINGS = {
    "RIFEInterpolation": RIFEInterpolation,
    "CalculateLoadedFPS": CalculateLoadedFPS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RIFEInterpolation": "RIFE Frame Interpolation",
    "CalculateLoadedFPS": "Calculate Loaded FPS",
}
