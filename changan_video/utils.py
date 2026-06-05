from __future__ import annotations

import importlib
from typing import Any


_ALIASES = {
    "cab.dataset.video_data.": "changan_video.dataset.video_data.",
    "cab.codec.ffmpeg_video.": "changan_video.codec.ffmpeg_video.",
    "cab.evaluations.": "changan_video.evaluations.",
}


def instantiate_from_config(config):
    """Instantiate an object from config without importing cab modules."""

    target = _alias_target(config["type"])
    func_or_class = get_obj_from_str(target)
    params = _to_plain_container(config.get("params", {})) or {}

    if callable(func_or_class) and not isinstance(func_or_class, type):
        return FunctionWrapper(func_or_class, params)

    return func_or_class(**params)


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    module_imp = importlib.import_module(module)
    if reload:
        importlib.reload(module_imp)
    return getattr(module_imp, cls)


def inject_codec_zero_means(cfg: dict[str, Any], codec_name: str) -> dict[str, Any]:
    codec_def = cfg.get(codec_name)
    if not codec_def:
        raise KeyError(f"codec {codec_name!r} not found in config")

    codec_params = codec_def.get("params", {}) or {}
    dataset_z = codec_params.get("dataset_zero_mean")
    metrics_z = codec_params.get("metrics_zero_mean")

    for ds_name in cfg.get("datasets", []):
        ds_def = cfg.get(ds_name)
        if not ds_def:
            continue
        ds_params = ds_def.setdefault("params", {})
        if dataset_z is not None:
            ds_params["zero_mean"] = dataset_z

    for metric_name in cfg.get("metrics", []):
        metric_def = cfg.get(metric_name)
        if not metric_def:
            continue
        metric_params = metric_def.setdefault("params", {})
        if metrics_z is not None:
            metric_params["zero_mean"] = metrics_z

    return cfg


class FunctionWrapper:
    def __init__(self, func, params: dict[str, Any]):
        self.func = func
        self.params = params

    def __call__(self, x, y, **kwargs):
        params = dict(self.params)
        params.update(kwargs)
        return self.func(x, y, **params)


def _alias_target(target: str) -> str:
    for old, new in _ALIASES.items():
        if target.startswith(old):
            return new + target[len(old):]
    return target


def _to_plain_container(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value
