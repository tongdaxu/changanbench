from __future__ import annotations

from pathlib import Path

import torch


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
    ):
        self.ckpt_path = ckpt_path
        self.device = device
        self.score = score
        self.mode = mode
        self.model, self.load_and_preprocess_images, self.pose_to_camera = self._load_vggt()
        self.model = self.model.to(device)
        state = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, x_input, x_recon):
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
        with torch.inference_mode():
            preds = self.model(images)
        extrinsic, intrinsic = self.pose_to_camera(preds["pose_enc"], images.shape[-2:])
        return {
            "extrinsic": extrinsic[0],
            "intrinsic": intrinsic[0],
        }

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
