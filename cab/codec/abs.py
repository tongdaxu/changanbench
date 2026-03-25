from abc import abstractmethod

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