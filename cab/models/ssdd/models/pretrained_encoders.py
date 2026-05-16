# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from diffusers.models import AutoencoderKL

from ..mutils.torch_utils import freeze_model


class SDVAEPretrainedEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sdvae = [AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")]
        self.patch_size = 8
        # self.z_dim = 4

        freeze_model(self)

    def _apply(self, fn):
        self.sdvae[0] = self.sdvae[0]._apply(fn)
        return super()._apply(fn)

    @torch.no_grad()
    def forward(self, x):
        z = self.sdvae[0].encode(x).latent_dist
        return z
