from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


class VGGTVideoMetric:
    """Compare VGGT predictions between reference and distorted video frames."""

    names = (
        "vggt_camera_center_error_mean",
        "vggt_camera_rotation_error_deg_mean",
        "vggt_depth_absrel",
        "vggt_point_l2_mean",
    )

    def __init__(
        self,
        *,
        ckpt_path: str,
        device: str = "cuda",
        frame_count: int = 32,
        mode: str = "crop",
    ) -> None:
        if frame_count <= 0:
            raise ValueError("frame_count must be greater than 0")
        self.ckpt_path = str(ckpt_path)
        self.device = device
        self.frame_count = int(frame_count)
        self.mode = mode
        self._torch = None
        self._model = None
        self._load_and_preprocess_images = None
        self._pose_to_camera = None

    def __call__(self, reference_frames: Sequence[np.ndarray], distorted_frames: Sequence[np.ndarray]) -> dict[str, float]:
        if not reference_frames or not distorted_frames:
            raise ValueError("VGGTVideoMetric requires non-empty frame sequences")
        count = min(len(reference_frames), len(distorted_frames))
        if count <= 0:
            raise ValueError("No aligned frames available for VGGTVideoMetric")

        indices = self._sample_indices(count)
        ref_sample = [reference_frames[i] for i in indices]
        dist_sample = [distorted_frames[i] for i in indices]

        self._ensure_model()
        ref_pred = self._predict(ref_sample)
        dist_pred = self._predict(dist_sample)
        return self._compute_metrics(ref_pred, dist_pred)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        import torch

        try:
            from vggt.models.vggt import VGGT
            from vggt.utils.load_fn import load_and_preprocess_images
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "VGGT is not importable. Install VGGT or add its source root to PYTHONPATH."
            ) from exc

        ckpt = Path(self.ckpt_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"VGGT checkpoint not found: {ckpt}")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for VGGTVideoMetric but torch.cuda.is_available() is False")

        model = VGGT()
        state = torch.load(str(ckpt), map_location=self.device)
        model.load_state_dict(state)
        model = model.to(self.device)
        model.eval()

        self._torch = torch
        self._model = model
        self._load_and_preprocess_images = load_and_preprocess_images
        self._pose_to_camera = pose_encoding_to_extri_intri

    def _predict(self, frames: Sequence[np.ndarray]) -> dict:
        torch = self._torch
        assert torch is not None
        with tempfile.TemporaryDirectory(prefix="changan_vggt_") as tmp:
            frame_paths = self._write_frames(frames, Path(tmp))
            images = self._load_and_preprocess_images(frame_paths, mode=self.mode).to(self.device)

            with torch.inference_mode():
                if self.device.startswith("cuda"):
                    major = torch.cuda.get_device_capability()[0]
                    dtype = torch.bfloat16 if major >= 8 else torch.float16
                    with torch.cuda.amp.autocast(dtype=dtype):
                        preds = self._model(images)
                else:
                    preds = self._model(images)

            extrinsic, intrinsic = self._pose_to_camera(preds["pose_enc"], images.shape[-2:])
            out = {
                "extrinsic": extrinsic[0].detach().float().cpu(),
                "intrinsic": intrinsic[0].detach().float().cpu(),
                "depth": preds["depth"][0].detach().float().cpu(),
                "depth_conf": preds["depth_conf"][0].detach().float().cpu(),
            }
            if "world_points" in preds:
                out["world_points"] = preds["world_points"][0].detach().float().cpu()
            del preds, images, extrinsic, intrinsic
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            return out

    def _compute_metrics(self, ref: dict, dist: dict) -> dict[str, float]:
        center_error = self._camera_center_error(ref["extrinsic"], dist["extrinsic"])
        rotation_error = self._rotation_error_deg(ref["extrinsic"], dist["extrinsic"])
        depth_absrel = self._depth_absrel(ref["depth"], dist["depth"])
        point_l2 = self._point_l2(ref.get("world_points"), dist.get("world_points"))
        return {
            "vggt_camera_center_error_mean": center_error,
            "vggt_camera_rotation_error_deg_mean": rotation_error,
            "vggt_depth_absrel": depth_absrel,
            "vggt_point_l2_mean": point_l2,
        }

    def _sample_indices(self, count: int) -> list[int]:
        target = min(self.frame_count, count)
        if target == 1:
            return [count // 2]
        values = np.linspace(0, count - 1, target)
        return [int(round(v)) for v in values]

    @staticmethod
    def _write_frames(frames: Sequence[np.ndarray], folder: Path) -> list[str]:
        paths = []
        for index, frame in enumerate(frames, start=1):
            array = np.asarray(frame)
            if array.dtype != np.uint8:
                array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"Expected RGB frame with shape HxWx3, got {array.shape}")
            path = folder / f"frame_{index:05d}.png"
            Image.fromarray(array, mode="RGB").save(path)
            paths.append(str(path))
        return paths

    @staticmethod
    def _camera_centers(extrinsic):
        import torch

        r = extrinsic[:, :3, :3]
        t = extrinsic[:, :3, 3]
        return -torch.einsum("nij,nj->ni", r.transpose(1, 2), t)

    @classmethod
    def _camera_center_error(cls, ext_ref, ext_dist) -> float:
        import torch

        centers_ref = cls._camera_centers(ext_ref)
        centers_dist = cls._camera_centers(ext_dist)
        value = torch.linalg.norm(centers_dist - centers_ref, dim=1).mean()
        return float(value.item())

    @staticmethod
    def _rotation_error_deg(ext_ref, ext_dist) -> float:
        import torch

        r_ref = ext_ref[:, :3, :3]
        r_dist = ext_dist[:, :3, :3]
        r_rel = torch.matmul(r_dist, r_ref.transpose(1, 2))
        trace = torch.diagonal(r_rel, dim1=1, dim2=2).sum(dim=1)
        cos = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
        return float(torch.rad2deg(torch.arccos(cos)).mean().item())

    @staticmethod
    def _depth_absrel(ref_depth, dist_depth) -> float:
        import torch

        ref = ref_depth.squeeze(-1)
        dist = dist_depth.squeeze(-1)
        valid = torch.isfinite(ref) & torch.isfinite(dist) & (ref > 1e-6) & (dist > 1e-6)
        if not bool(valid.any()):
            return math.nan
        diff = torch.abs(dist[valid] - ref[valid]) / torch.clamp(ref[valid], min=1e-6)
        return float(diff.mean().item())

    @staticmethod
    def _point_l2(ref_points, dist_points) -> float:
        if ref_points is None or dist_points is None:
            return math.nan
        import torch

        valid = torch.isfinite(ref_points).all(dim=-1) & torch.isfinite(dist_points).all(dim=-1)
        if not bool(valid.any()):
            return math.nan
        dist = torch.linalg.norm(dist_points[valid] - ref_points[valid], dim=-1)
        return float(dist.mean().item())
