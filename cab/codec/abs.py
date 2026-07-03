from abc import abstractmethod
from cab.complexity import params_m, gflops

import torch
import torch.nn as nn

class ImageCodecIface(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        """Implement encoding/decoding.
        Args:
            x: (B, 3, H, W)
        Returns:
            xhat: (B, 3, H, W)
            bpp:  (B,) tensor
        """
        raise NotImplementedError
    
    def fake_input(self, image_size=256, batch_size=1, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.randn(1, 3, image_size, image_size, device=device)

    def encode_params_m(self):
        return params_m(self)

    def decode_params_m(self):
        return params_m(self)

    def encode_time_ms(self, x, warmup=5, repeat=20):
        return None

    def decode_time_ms(self, x, warmup=5, repeat=20):
        return None

    def encode_gflops(self, x):
        return gflops(self, x)

    def decode_gflops(self, x):
        return None

class VideoCodecIface(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        """Implement video encoding/decoding.
        Args:
            x: (B, 3, T, H, W)
        Returns:
            xhat: (B, 3, T, H, W)
            bpp:  (B,) tensor, bits per pixel per frame
        """
        raise NotImplementedError

    def fake_input(self, image_size=256, batch_size=1, frames=32, device=None, *args, **kwargs):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        frames = 32 if frames is None else int(frames)
        return torch.randn(
            int(batch_size), 3, frames, int(image_size), int(image_size),
            device=device,
        )

    def encode_params_m(self):
        return params_m(self)

    def decode_params_m(self):
        return params_m(self)

    def flops(self, x, *args, **kwargs):
        return None

    def encode_gflops(self, x):
        return self.flops(x)

    def decode_gflops(self, x):
        return None

    def param_count(self, x, *args, **kwargs):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params

    def encode_time_ms(self, x, warmup=5, repeat=20):
        return None

    def decode_time_ms(self, x, warmup=5, repeat=20):
        return None

    def encode_time(self, x, *args, **kwargs):
        return self.encode_time_ms(x, *args, **kwargs)

    def decode_time(self, x, *args, **kwargs):
        return self.decode_time_ms(x, *args, **kwargs)

    # # complexity: to revise
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total params: {total_params:,}")
    # print(f"Trainable params: {trainable_params:,}")

    # # flops
    # import torch
    # from fvcore.nn import FlopCountAnalysis, parameter_count_table
    # model.eval()
    # dummy = torch.randn(1, 3, 224, 224)
    # flops = FlopCountAnalysis(model, dummy)
    # print("Total FLOPs:", flops.total())
    # print(parameter_count_table(model))

    # # latency
    # import torch
    # import torch.utils.benchmark as benchmark
    # model.eval().cuda()
    # x = torch.randn(1, 3, 224, 224, device='cuda')
    # t = benchmark.Timer(
    # stmt='model(x)',
    # globals={'model': model, 'x': x},
    # )
    # result = t.blocked_autorange(min_run_time=1.0)
    # print(result)
