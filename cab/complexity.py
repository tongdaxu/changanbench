import time
import torch
import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None


def count_params(module: nn.Module):
    return sum(p.numel() for p in module.parameters())


def count_trainable_params(module: nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def cpu_time_ms(fn, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)

    return sum(times) / len(times)


def cuda_time_ms(fn, warmup=3, repeat=10):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(repeat):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))

    return sum(times) / len(times)


def measure_time_ms(fn, device, warmup=10, repeat=50):
    device = torch.device(device)

    if device.type == "cuda" and torch.cuda.is_available():
        return cuda_time_ms(fn, warmup=warmup, repeat=repeat)

    return cpu_time_ms(
        fn,
        warmup=max(1, warmup // 2),
        repeat=max(1, repeat // 2),
    )


def safe_flops(module, inputs):
    if FlopCountAnalysis is None:
        return None, {"error": "fvcore is not installed"}

    try:
        module.eval()
        flops = FlopCountAnalysis(module, inputs)
        return int(flops.total()), {
            "unsupported_ops": flops.unsupported_ops(),
            "uncalled_modules": flops.uncalled_modules(),
        }
    except Exception as e:
        return None, {"error": repr(e)}