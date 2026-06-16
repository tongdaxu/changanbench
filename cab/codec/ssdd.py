import torch
import os
import numpy as np
from cab.codec.abs import ImageCodecIface
from cab.models.ssdd import SSDD
import re

from cab.complexity import (
    params_m,
    gflops,
    time_ms,
)

class SSDDEncodeWrapper(torch.nn.Module):
    def __init__(self, ssdd_model):
        super().__init__()
        self.model = ssdd_model

    def forward(self, x):
        return self.model.encode(x).mode()


class SSDDDecodeWrapper(torch.nn.Module):
    def __init__(self, ssdd_model, steps=1):
        super().__init__()
        self.model = ssdd_model
        self.steps = steps

    def forward(self, z):
        return self.model.decode(z, steps=self.steps)


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

        posterior = self.model.encode(x)
        z = posterior.mode()

        xhat = self.model.decode(z, steps=1)

        kl = posterior.kl()
        bpp = kl / (x.shape[2] * x.shape[3]) / np.log(2)

        return xhat, bpp
    
    def encode_params_m(self):
        return params_m(self.model.encoder)

    def decode_params_m(self):
        return params_m(self.model.decoder)

    @torch.no_grad()
    def encode_time_ms(self, x, warmup=3, repeat=10):
        x = x.to(self.device)
        enc = SSDDEncodeWrapper(self.model).to(self.device).eval()
        return time_ms(lambda: enc(x), self.device, warmup, repeat)

    @torch.no_grad()
    def decode_time_ms(self, x, warmup=3, repeat=10, steps=1):
        x = x.to(self.device)

        z = self.model.encode(x).mode()
        dec = SSDDDecodeWrapper(self.model, steps=steps).to(self.device).eval()

        return time_ms(lambda: dec(z), self.device, warmup, repeat)

    def encode_gflops(self, x):
        x = x.to(self.device)
        enc = SSDDEncodeWrapper(self.model).to(self.device).eval()
        return gflops(enc, x)

    @torch.no_grad()
    def decode_gflops(self, x, steps=1):
        x = x.to(self.device)
        z = self.model.encode(x).mode()

        dec = SSDDDecodeWrapper(self.model, steps=steps).to(self.device).eval()
        return gflops(dec, z)