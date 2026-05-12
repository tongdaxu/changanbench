import importlib

def instantiate_from_config(config):
    """Instantiate object from config, handling both classes and functions"""
    func_or_class = get_obj_from_str(config["type"])
    params = config.get("params", {})
    
    # 如果是函数，直接返回包装后的版本
    if callable(func_or_class) and not isinstance(func_or_class, type):
        class FunctionWrapper:
            def __call__(self, x, y):
                return func_or_class(x, y)
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
