import os
from typing import List, Optional, Tuple

import torch
from torch.nn import functional as F


class RIFEWrapper:
    """Wrapper for RIFE model to work with ComfyUI Image tensors"""

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, model_path, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup torch for optimal performance
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            # Enable memory efficient attention if available
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)

        # Load model
        from .train_log.RIFE_HDv3 import Model

        self.model = Model()
        self.model.load_model(model_path, -1)
        self.model.eval()
        self.model.device()

    def interpolate_frames(
        self,
        images: torch.Tensor,
        source_fps: float,
        target_fps: float,
        scale: float = 1.0,
        progress_callback=None,
    ) -> torch.Tensor:
        """
        Interpolate frames from source FPS to target FPS

        Args:
            images: ComfyUI Image tensor [N, H, W, C] in range [0, 1]
            source_fps: Source frame rate
            target_fps: Target frame rate
            scale: Scale factor for processing
            progress_callback: Optional callback function that accepts (current, total) parameters

        Returns:
            Interpolated ComfyUI Image tensor [M, H, W, C] in range [0, 1]
        """
        # Validate input
        assert images.dim() == 4 and images.shape[-1] == 3, "Input must be [N, H, W, C] with C=3"

        if source_fps == target_fps:
            return images

        total_source_frames = images.shape[0]
        height, width = images.shape[1:3]

        # Calculate padding for model
        tmp = max(128, int(128 / scale))
        ph = ((height - 1) // tmp + 1) * tmp
        pw = ((width - 1) // tmp + 1) * tmp
        padding = (0, pw - width, 0, ph - height)

        # Calculate target frame positions
        frame_positions = self._calculate_target_frame_positions(source_fps, target_fps, total_source_frames)

        # 内存优化：分离源帧和插值帧，只为插值帧分配内存
        total_frames = len(frame_positions)
        # 统计需要插值的帧数量
        interp_indices = []
        for idx, (source_idx1, source_idx2, interp_factor) in enumerate(frame_positions):
            if interp_factor != 0.0 and source_idx1 != source_idx2:
                interp_indices.append(idx)

        # 只为插值帧分配内存
        num_interp_frames = len(interp_indices)
        if num_interp_frames > 0:
            interpolated_frames = torch.empty((num_interp_frames, height, width, 3), dtype=images.dtype, device="cpu")
        else:
            interpolated_frames = torch.empty((0, height, width, 3), dtype=images.dtype, device="cpu")

        # Process frames with optimized memory management and batching
        batch_size = 4  # Process 4 frames at a time for better GPU utilization
        interp_batch = []

        with torch.inference_mode():
            for idx, (source_idx1, source_idx2, interp_factor) in enumerate(frame_positions):
                if interp_factor == 0.0 or source_idx1 == source_idx2:
                    # No interpolation needed, will handle in final assembly
                    pass
                else:
                    # Collect frames for batch processing
                    frame1 = images[source_idx1]
                    frame2 = images[source_idx2]
                    interp_batch.append((frame1, frame2, interp_factor, idx))

                    # Process batch when it reaches the target size
                    if len(interp_batch) >= batch_size:
                        self._process_interpolation_batch_optimized(
                            interp_batch, interpolated_frames, interp_indices, padding, scale, height, width
                        )
                        interp_batch = []

                if progress_callback:
                    progress_callback(idx + 1, total_frames)

            # Process remaining frames in the batch
            if interp_batch:
                self._process_interpolation_batch_optimized(
                    interp_batch, interpolated_frames, interp_indices, padding, scale, height, width
                )

        # 组装最终结果：源帧直接引用，插值帧从预分配内存中获取
        return self._assemble_output_frames(images, interpolated_frames, frame_positions, interp_indices)

    def _batch_interpolate_frames(self, frame_batch, padding, scale):
        """Batch process multiple frame interpolations for better GPU utilization"""
        batch_size = len(frame_batch)
        if batch_size == 0:
            return []

        # Prepare batch tensors
        batch_I0 = []
        batch_I1 = []
        timesteps = []

        for frame1, frame2, interp_factor in frame_batch:
            I0 = frame1.permute(2, 0, 1).unsqueeze(0).to(self.device)
            I1 = frame2.permute(2, 0, 1).unsqueeze(0).to(self.device)

            # Pad images
            I0 = F.pad(I0, padding)
            I1 = F.pad(I1, padding)

            batch_I0.append(I0)
            batch_I1.append(I1)
            timesteps.append(interp_factor)

        # Stack batch
        batch_I0 = torch.cat(batch_I0, dim=0)
        batch_I1 = torch.cat(batch_I1, dim=0)

        # Batch inference
        results = []
        for i in range(batch_size):
            interpolated = self.model.inference(
                batch_I0[i : i + 1], batch_I1[i : i + 1], timestep=timesteps[i], scale=scale
            )
            results.append(interpolated)

        # Cleanup
        del batch_I0, batch_I1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def _process_interpolation_batch_optimized(
        self, interp_batch, interpolated_frames, interp_indices, padding, scale, height, width
    ):
        """Process a batch of frame interpolations with memory optimization"""
        if not interp_batch:
            return

        # Prepare frames for batch processing
        frame_data = [(f1, f2, factor) for f1, f2, factor, _ in interp_batch]
        batch_indices = [idx for _, _, _, idx in interp_batch]

        # Batch interpolate
        interpolated_results = self._batch_interpolate_frames(frame_data, padding, scale)

        # Store results in pre-allocated memory
        for i, (interpolated, original_idx) in enumerate(zip(interpolated_results, batch_indices)):
            # Find the position in interpolated_frames array
            interp_pos = interp_indices.index(original_idx)
            interpolated_frames[interp_pos] = interpolated[0, :, :height, :width].permute(1, 2, 0).cpu()

        # Cleanup
        del interpolated_results, frame_data, batch_indices

    def _assemble_output_frames(self, images, interpolated_frames, frame_positions, interp_indices):
        """组装最终输出帧：源帧直接引用，插值帧从预分配内存中获取"""
        height, width = images.shape[1:3]

        # 创建输出tensor列表而不是预分配大块内存
        output_frames = []

        interp_idx = 0
        for idx, (source_idx1, source_idx2, interp_factor) in enumerate(frame_positions):
            if interp_factor == 0.0 or source_idx1 == source_idx2:
                # 使用源帧，直接引用输入图像
                output_frames.append(images[source_idx1])
            else:
                # 使用插值帧
                output_frames.append(interpolated_frames[interp_idx])
                interp_idx += 1

        # 将列表转换为tensor
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
