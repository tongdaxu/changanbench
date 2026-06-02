from __future__ import annotations

import importlib.util
import importlib
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_METRICS = ("psnr", "ssim", "msssim", "lpips")
FRAME_METRICS = ("psnr", "ssim", "msssim", "lpips")
VIDEO_METRICS = ("fid", "fvd", "vggt")
ALL_METRICS = (*FRAME_METRICS, "fid", "fvd")
SUPPORTED_METRICS = (*FRAME_METRICS, *VIDEO_METRICS)


def normalize_metrics(metrics: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for metric in metrics:
        parts = [part.strip().lower() for part in str(metric).split(",")]
        for part in parts:
            if not part:
                continue
            if part == "all":
                candidates = ALL_METRICS
            else:
                candidates = (_normalize_metric_name(part),)

            for candidate in candidates:
                if candidate not in SUPPORTED_METRICS:
                    supported = ", ".join(SUPPORTED_METRICS)
                    raise ValueError(f"Unsupported metric {part!r}; choose from {supported}")
                if candidate not in normalized:
                    normalized.append(candidate)

    if not normalized:
        raise ValueError("At least one metric is required")
    return tuple(normalized)


def check_dependencies(metrics: Sequence[str]) -> None:
    normalized = _normalize_metric_names(metrics)
    required = {"av", "numpy", "torch"}
    if "ssim" in normalized or "msssim" in normalized:
        required.add("pytorch_msssim")
    if "lpips" in normalized:
        required.add("lpips")
    if "fid" in normalized:
        required.update({"scipy", "torchvision"})
    if "fvd" in normalized:
        required.update({"scipy", "requests"})

    missing = sorted(module for module in required if importlib.util.find_spec(module) is None)
    if missing:
        modules = ", ".join(missing)
        raise SystemExit(
            f"Missing dependencies: {modules}. Activate the evaluation environment first."
        )


def metrics_from_config(
    config: Any,
    metrics_name: Sequence[str],
    *,
    fallback_zero_mean: bool,
    fallback_lpips_network: str,
) -> tuple[list[str], bool, str]:
    metric_names: list[str] = []
    zero_mean_values: list[bool] = []
    lpips_network = fallback_lpips_network

    for metric_name in metrics_name:
        entry = _to_plain_container(config.get(metric_name, {}))
        metric_type = str(entry.get("type", ""))
        params = entry.get("params", {}) or {}

        metric_names.extend(_metric_names_from_entry(str(metric_name), metric_type))
        if "zero_mean" in params:
            zero_mean_values.append(str_to_bool(params["zero_mean"]))
        if "network_type" in params:
            lpips_network = str(params["network_type"])

    zero_mean = fallback_zero_mean
    if zero_mean_values:
        zero_mean = zero_mean_values[0]
        if any(value != zero_mean for value in zero_mean_values):
            print(
                "Warning: metric config has mixed zero_mean values; "
                f"using {zero_mean} for frame-by-frame video evaluation.",
                flush=True,
            )
    return metric_names, zero_mean, lpips_network


def fvd_options_from_config(
    config: Any,
    metrics_name: Sequence[str],
    *,
    base_dir: Path,
) -> tuple[int, int, str | None]:
    clip_length = 16
    clip_stride = 16
    model_path = None

    for metric_name in metrics_name:
        entry = _to_plain_container(config.get(metric_name, {}))
        if not isinstance(entry, dict):
            continue

        metric_type = str(entry.get("type", ""))
        if "fvd" not in _metric_names_from_entry(str(metric_name), metric_type):
            continue

        params = entry.get("params", {}) or {}
        if not isinstance(params, dict):
            continue

        clip_length = _int_param(
            params,
            ("clip_length", "fvd_clip_length"),
            default=clip_length,
        )
        clip_stride = _int_param(
            params,
            ("clip_stride", "fvd_clip_stride"),
            default=clip_stride,
        )
        candidate = _first_param(
            params,
            ("model_path", "fvd_model_path", "i3d_model_path"),
        )
        if candidate not in (None, ""):
            model_path = _resolve_source(str(candidate), base_dir)

    return clip_length, clip_stride, model_path


def vggt_metric_from_config(
    config: Any,
    metrics_name: Sequence[str],
    *,
    base_dir: Path,
):
    for metric_name in metrics_name:
        entry = _to_plain_container(config.get(metric_name, {}))
        if not isinstance(entry, dict):
            continue

        metric_type = str(entry.get("type", ""))
        if "vggt" not in _metric_names_from_entry(str(metric_name), metric_type):
            continue

        class_path = metric_type or "changan_video.evaluations.vggt.VGGTVideoMetric"
        params = dict(entry.get("params", {}) or {})
        ckpt_path = params.get("ckpt_path")
        if ckpt_path not in (None, ""):
            params["ckpt_path"] = _resolve_source(str(ckpt_path), base_dir)
        module_name, class_name = class_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(**params)
    return None


def video_path_from_config(
    entry: Any,
    *,
    entry_name: str,
    role: str,
    base_dir: Path,
) -> str:
    data = _to_plain_container(entry)
    if not isinstance(data, dict):
        raise ValueError(f"Config entry {entry_name!r} must be a mapping")

    if role == "reference":
        candidates = ["reference", "ref", "path", "source", "video", "root"]
    else:
        candidates = [
            "distorted",
            "recon",
            "reconstructed",
            "path",
            "source",
            "video",
            "output",
            "root",
        ]

    params = data.get("params", {}) or {}
    for key in candidates:
        if key in data:
            return _resolve_source(str(data[key]), base_dir)
        if isinstance(params, dict) and key in params:
            return _resolve_source(str(params[key]), base_dir)

    sample = _first_param(data, ("xiph_sample", "xiph", "sample"))
    if sample is None and isinstance(params, dict):
        sample = _first_param(params, ("xiph_sample", "xiph", "sample"))
    if sample not in (None, ""):
        return _resolve_xiph_sample(str(sample))

    raise ValueError(
        f"Config entry {entry_name!r} needs a video path. "
        "Use one of: path, source, video, reference/ref, distorted/recon/reconstructed, "
        "xiph_sample/xiph/sample."
    )


def progress_callback(progress_every: int):
    if progress_every <= 0:
        return None
    return lambda n: print(f"evaluated {n} frames", flush=True) if (
        n % progress_every == 0
    ) else None


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _normalize_metric_names(metrics: Sequence[str]) -> set[str]:
    names: set[str] = set()
    for metric in metrics:
        for part in str(metric).split(","):
            name = part.strip().lower()
            if not name:
                continue
            if name == "all":
                names.update(ALL_METRICS)
            elif name in {"ms-ssim", "ms_ssim"}:
                names.add("msssim")
            else:
                names.add(name)
    return names


def _metric_names_from_entry(metric_name: str, metric_type: str) -> list[str]:
    text = f"{metric_name} {metric_type}".lower()
    if "all" == metric_name.lower():
        return list(ALL_METRICS)
    if "get_ssim_and_msssim" in text:
        return ["ssim", "msssim"]
    if "ms-ssim" in text or "ms_ssim" in text or "msssim" in text:
        return ["msssim"]
    for name in ("ssim", "lpips", "fid", "fvd", "vggt", "psnr"):
        if name in text:
            return [name]
    return [metric_name]


def _to_plain_container(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


def _resolve_source(source: str, base_dir: Path) -> str:
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", source):
        return source
    path = Path(source)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _resolve_xiph_sample(sample_name: str) -> str:
    from changan_video.dataset.xiph_dataset import XIPH_SAMPLES

    if sample_name in XIPH_SAMPLES:
        return XIPH_SAMPLES[sample_name].url
    known = ", ".join(sorted(XIPH_SAMPLES))
    raise ValueError(f"Unknown Xiph sample {sample_name!r}. Known samples: {known}")


def _normalize_metric_name(metric: str) -> str:
    aliases = {
        "ms-ssim": "msssim",
        "ms_ssim": "msssim",
        "msssim": "msssim",
    }
    return aliases.get(metric, metric)


def _first_param(params: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in params:
            return params[key]
    return None


def _int_param(params: dict[str, Any], keys: Sequence[str], *, default: int) -> int:
    value = _first_param(params, keys)
    if value in (None, ""):
        return default
    return int(value)
