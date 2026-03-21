import torch
import torch.nn as nn

class ImageCodecIface(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError
        # x: (B, 3, H, W)
        # return
        #   xhat: (B, 3, H, W)
        #   bpp: (B)