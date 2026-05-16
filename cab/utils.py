import importlib
from typing import Dict, Any

def instantiate_from_config(config):
    """Instantiate object from config, handling both classes and functions"""
    func_or_class = get_obj_from_str(config["type"])
    params = config.get("params", {})
    
    # 如果是函数，直接返回包装后的版本
    if callable(func_or_class) and not isinstance(func_or_class, type):
        class FunctionWrapper:
            def __call__(self, x, y, **kwargs):
                return func_or_class(x, y, **kwargs)
        return FunctionWrapper()
    
    # 如果是类，进行实例化
    return func_or_class(**params)


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def inject_codec_zero_means(cfg: Dict[str, Any], codec_name: str) -> Dict[str, Any]:

    codec_def = cfg.get(codec_name)
    if not codec_def:
        raise KeyError(f"codec '{codec_name}' not found in config")
    codec_params = codec_def.get("params", {})
    dataset_z = codec_params.get("dataset_zero_mean")
    metrics_z = codec_params.get("metrics_zero_mean")

    for ds_name in cfg.get("datasets", []):
        ds_def = cfg.get(ds_name)
        if not ds_def:
            continue
        ds_params = ds_def.setdefault("params", {})
        if dataset_z is not None:
            ds_params["zero_mean"] = dataset_z

    for m_name in cfg.get("metrics", []):
        m_def = cfg.get(m_name)
        if not m_def:
            continue
        m_params = m_def.setdefault("params", {})
        if metrics_z is not None:
            m_params["zero_mean"] = metrics_z

    return cfg