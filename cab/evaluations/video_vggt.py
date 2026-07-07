from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


class VGGTMetric:
    """VGGT-based video metric wrapper.

    This wrapper expects VGGT code to be importable from cab.models.vggt or from
    the Python environment. It returns a single per-video tensor so the current
    DDP aggregation path can consume it directly.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        score: str = "camera_center_error_mean",
        mode: str = "crop",
        zero_mean: bool | None = None,
    ):
        self.ckpt_path = ckpt_path
        self.device = device
        self.score = score
        self.mode = mode
        self.zero_mean = zero_mean
        self.model, self.load_and_preprocess_images, self.pose_to_camera = self._load_vggt()
        self.model = self.model.to(device)
        state = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, x_input, x_recon, **kwargs):
        if x_input.ndim != 5 or x_recon.ndim != 5:
            raise ValueError("VGGTMetric expects tensors shaped (B, C, T, H, W)")
        if self.score != "camera_center_error_mean":
            raise NotImplementedError(f"Unsupported VGGT score: {self.score}")

        values = []
        for src, rec in zip(x_input, x_recon):
            pred_src = self._predict(src)
            pred_rec = self._predict(rec)
            values.append(self._camera_center_error(pred_src["extrinsic"], pred_rec["extrinsic"]))
        return torch.stack(values).to(device=x_input.device)

    def _predict(self, video: torch.Tensor) -> dict:
        video = video.detach().clamp(0.0, 1.0).to(self.device)
        images = video.permute(1, 0, 2, 3).unsqueeze(0)
        images = self._prepare_images(images)
        with torch.inference_mode():
            preds = self.model(images)
        extrinsic, intrinsic = self.pose_to_camera(preds["pose_enc"], images.shape[-2:])
        return {
            "extrinsic": extrinsic[0],
            "intrinsic": intrinsic[0],
        }

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        patch_size = 14
        height, width = images.shape[-2:]
        if height < patch_size or width < patch_size:
            raise ValueError(f"VGGT input is too small: {height}x{width}")
        if height % patch_size == 0 and width % patch_size == 0:
            return images
        if self.mode == "crop":
            target_h = height // patch_size * patch_size
            target_w = width // patch_size * patch_size
            top = (height - target_h) // 2
            left = (width - target_w) // 2
            return images[..., top : top + target_h, left : left + target_w]
        if self.mode == "pad":
            target_h = (height + patch_size - 1) // patch_size * patch_size
            target_w = (width + patch_size - 1) // patch_size * patch_size
            pad_h = target_h - height
            pad_w = target_w - width
            return F.pad(
                images,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            )
        raise ValueError(f"Unsupported VGGT preprocessing mode: {self.mode}")

    def _camera_center_error(self, ext_ref: torch.Tensor, ext_test: torch.Tensor) -> torch.Tensor:
        centers_ref = self._camera_centers(ext_ref)
        centers_test = self._camera_centers(ext_test)
        return torch.linalg.norm(centers_test - centers_ref, dim=1).mean()

    @staticmethod
    def _camera_centers(extrinsic: torch.Tensor) -> torch.Tensor:
        r = extrinsic[:, :3, :3]
        t = extrinsic[:, :3, 3]
        return -torch.einsum("nij,nj->ni", r.transpose(1, 2), t)

    def _load_vggt(self):
        try:
            from cab.models.vggt.models.vggt import VGGT
            from cab.models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
        except ModuleNotFoundError:
            try:
                from vggt.models.vggt import VGGT
                from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "VGGT code is not available. Place it under cab/models/vggt "
                    "or install it so `import vggt` works."
                ) from exc

        if not Path(self.ckpt_path).exists():
            raise FileNotFoundError(f"VGGT checkpoint not found: {self.ckpt_path}")
        return VGGT(), None, pose_encoding_to_extri_intri


class VGGTVideoMetric(VGGTMetric):
    """Pairwise video-file VGGT metric.

    The pairwise evaluator passes decoded uint8 frame arrays, while VGGTMetric
    consumes batched tensors. This adapter keeps config compatibility with the
    video-pair path.
    """

    def __init__(self, *args, frame_count: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_count = frame_count

    def __call__(self, reference_frames, distorted_frames):
        import numpy as np

        reference = self._frames_to_tensor(reference_frames)
        distorted = self._frames_to_tensor(distorted_frames)
        value = super().__call__(reference, distorted)
        return {"vggt": float(value.detach().reshape(-1).mean().cpu().item())}

    def _frames_to_tensor(self, frames):
        import numpy as np

        if not frames:
            raise ValueError("VGGTVideoMetric received no frames")
        selected = self._select_frames(frames)
        array = np.stack(selected, axis=0)
        tensor = torch.from_numpy(array).permute(3, 0, 1, 2).unsqueeze(0)
        return tensor.to(device=self.device, dtype=torch.float32).div_(255.0)

    def _select_frames(self, frames):
        import numpy as np

        if self.frame_count is None or len(frames) <= self.frame_count:
            return list(frames)
        indices = np.linspace(0, len(frames) - 1, int(self.frame_count))
        return [frames[int(round(index))] for index in indices]
