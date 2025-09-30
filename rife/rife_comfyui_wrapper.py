import os
from typing import List, Optional, Tuple

import torch
from torch.nn import functional as F

from .train_log.RIFE_HDv3 import Model


class RIFEWrapper:
    """Wrapper for RIFE model to work with ComfyUI Image tensors"""

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, model_path, device: Optional[torch.device] = None, use_fp16: bool = False):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and torch.cuda.is_available()

        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)

        self.model = Model()
        self.model.load_model(model_path, -1)
        self.model.eval()
        self.model.device()

        # Convert to fp16 if requested
        if self.use_fp16:
            self.model.flownet = self.model.flownet.half()

    def interpolate_frames(
        self,
        images: torch.Tensor,
        source_fps: float,
        target_fps: float,
        scale: float = 1.0,
        progress_callback=None,
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        Interpolate frames from source FPS to target FPS

        Args:
            images: ComfyUI Image tensor [N, H, W, C] in range [0, 1]
            source_fps: Source frame rate
            target_fps: Target frame rate
            scale: Scale factor for processing
            progress_callback: Optional callback function that accepts (current, total) parameters
            batch_size: Number of frames to process in parallel (default: 8)

        Returns:
            Interpolated ComfyUI Image tensor [M, H, W, C] in range [0, 1]
        """

        assert images.dim() == 4 and images.shape[-1] == 3, "Input must be [N, H, W, C] with C=3"

        if source_fps == target_fps:
            return images

        total_source_frames = images.shape[0]
        height, width = images.shape[1:3]

        # Calculate padding
        tmp = max(128, int(128 / scale))
        ph = ((height - 1) // tmp + 1) * tmp
        pw = ((width - 1) // tmp + 1) * tmp
        padding = (0, pw - width, 0, ph - height)

        # Calculate frame positions
        frame_positions = self._calculate_target_frame_positions(source_fps, target_fps, total_source_frames)

        # Pre-allocate output on CPU (NOT GPU to avoid OOM)
        output_frames = []

        # Build interpolation job list
        interp_job_list = []
        output_index_map = {}  # Maps job_idx -> output position

        for out_idx, (source_idx1, source_idx2, interp_factor) in enumerate(frame_positions):
            if interp_factor == 0.0 or source_idx1 == source_idx2:
                # Direct copy, no interpolation needed
                output_frames.append(images[source_idx1])
            else:
                # Need interpolation - add placeholder
                output_frames.append(None)
                job_idx = len(interp_job_list)
                interp_job_list.append((source_idx1, source_idx2, interp_factor))
                output_index_map[job_idx] = out_idx

        # Process interpolations in batches with streaming
        num_jobs = len(interp_job_list)
        gpu_dtype = torch.float16 if self.use_fp16 else torch.float32

        with torch.inference_mode():
            for batch_start in range(0, num_jobs, batch_size):
                batch_end = min(batch_start + batch_size, num_jobs)
                current_batch_size = batch_end - batch_start

                # Collect unique source frames needed for this batch
                source_frames_needed = set()
                for job_idx in range(batch_start, batch_end):
                    source_idx1, source_idx2, _ = interp_job_list[job_idx]
                    source_frames_needed.add(source_idx1)
                    source_frames_needed.add(source_idx2)

                # Load only required source frames to GPU
                source_cache = {}
                for src_idx in source_frames_needed:
                    source_cache[src_idx] = images[src_idx].to(device=self.device, dtype=gpu_dtype)

                # Prepare batch tensors on GPU
                batch_I0 = torch.empty((current_batch_size, 3, ph, pw), dtype=gpu_dtype, device=self.device)
                batch_I1 = torch.empty((current_batch_size, 3, ph, pw), dtype=gpu_dtype, device=self.device)
                timesteps = []

                for i, job_idx in enumerate(range(batch_start, batch_end)):
                    source_idx1, source_idx2, interp_factor = interp_job_list[job_idx]

                    # Get frames from cache (already on GPU)
                    I0 = source_cache[source_idx1].permute(2, 0, 1).unsqueeze(0)
                    I1 = source_cache[source_idx2].permute(2, 0, 1).unsqueeze(0)

                    # Pad
                    batch_I0[i] = F.pad(I0, padding)[0]
                    batch_I1[i] = F.pad(I1, padding)[0]
                    timesteps.append(interp_factor)

                # Batch inference
                interpolated_batch = self.model.inference_batch(batch_I0, batch_I1, timesteps, scale=scale)

                # Transfer results to CPU and store
                for i, job_idx in enumerate(range(batch_start, batch_end)):
                    output_idx = output_index_map[job_idx]
                    result = interpolated_batch[i, :, :height, :width].permute(1, 2, 0).cpu().to(torch.float32)
                    output_frames[output_idx] = result

                # Update progress
                if progress_callback:
                    progress_callback(batch_end, num_jobs)

                # Cleanup batch memory immediately
                del batch_I0, batch_I1, interpolated_batch, source_cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Stack all output frames
        result = torch.stack(output_frames, dim=0)

        return result

    def _calculate_target_frame_positions(
        self, source_fps: float, target_fps: float, total_source_frames: int
    ) -> List[Tuple[int, int, float]]:
        """
        Calculate which frames need to be generated for the target frame rate.

        Returns:
            List of (source_frame_index1, source_frame_index2, interpolation_factor) tuples
        """
        frame_positions = []

        # Calculate the time duration of the video
        duration = total_source_frames / source_fps

        # Calculate number of target frames
        total_target_frames = int(duration * target_fps)

        for target_idx in range(total_target_frames):
            # Calculate the time position of this target frame
            target_time = target_idx / target_fps

            # Calculate the corresponding position in source frames
            source_position = target_time * source_fps

            # Find the two source frames to interpolate between
            source_idx1 = int(source_position)
            source_idx2 = min(source_idx1 + 1, total_source_frames - 1)

            # Calculate interpolation factor (0 means use frame1, 1 means use frame2)
            if source_idx1 == source_idx2:
                interpolation_factor = 0.0
            else:
                interpolation_factor = source_position - source_idx1

            frame_positions.append((source_idx1, source_idx2, interpolation_factor))

        return frame_positions
