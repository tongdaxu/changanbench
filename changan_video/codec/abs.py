from __future__ import annotations

from abc import abstractmethod

import torch.nn as nn


class VideoCodecIface(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        """Encode/decode video tensors shaped (B, 3, T, H, W)."""
        raise NotImplementedError
