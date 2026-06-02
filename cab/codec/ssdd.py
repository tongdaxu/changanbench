import torch
import os
import numpy as np
from cab.codec.abs import ImageCodecIface
from cab.models.ssdd import SSDD
import re

class SSDDImageTokenizer(ImageCodecIface):
    def __init__(self, quality, ckpt_path, encoder_spec, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality = quality
        self.ckpt_path = ckpt_path
        self.encoder_spec = encoder_spec
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        match = re.match(r"f(\d+)c(\d+)", self.encoder_spec)
        if match:
            z_dim = int(match.group(2))
        
        self.model = SSDD(
            encoder=self.encoder_spec,
            decoder={"size": "M", "z_dim": z_dim},
            checkpoint=self.ckpt_path,
        ).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):

        x = x.to(self.device, dtype=torch.float)
        # Encode an image into a latent
        z = self.model.encode(x).mode()
        # Reconstruct image form the latent variable
        xhat = self.model.decode(z, steps=1)
        # continuous latent: calculate KL divergence
        posterior = self.model.encode(x)
        kl = posterior.kl()
        bpp = kl / (x.shape[2] * x.shape[3]) / np.log(2)
        
        return xhat, bpp