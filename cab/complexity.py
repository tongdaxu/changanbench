import time
import torch

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None


def params_m(module):
    return sum(p.numel() for p in module.parameters()) / 1e6


def time_ms(fn, device, warmup=5, repeat=20):
    device = torch.device(device)

    for _ in range(warmup):
        fn()

    if device.type == "cuda":
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

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def gflops(module, inputs):
    if FlopCountAnalysis is None:
        return None

    try:
        module.eval()
        flops = FlopCountAnalysis(module, inputs).total()
        return flops / 1e9
    except Exception as e:
        print(f"[FLOPs failed] {e}")
        return None