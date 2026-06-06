import torch
import os
import numpy as np
from cab.codec.abs import ImageCodecIface
from cab.models.ssdd import SSDD
import re

from cab.complexity import (
    count_params,
    count_trainable_params,
    measure_time_ms,
    safe_flops,
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


class SSDDFullWrapper(torch.nn.Module):
    def __init__(self, ssdd_model, steps=1):
        super().__init__()
        self.model = ssdd_model
        self.steps = steps

    def forward(self, x):
        z = self.model.encode(x).mode()
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
    
    @torch.no_grad()
    def complexity(self, image_size=256, batch_size=1, steps=1, warmup=10, repeat=50):
        device = self.device
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)

        self.model.eval()

        posterior = self.model.encode(x)
        z = posterior.mode()

        enc = SSDDEncodeWrapper(self.model).to(device).eval()
        dec = SSDDDecodeWrapper(self.model, steps=steps).to(device).eval()
        full = SSDDFullWrapper(self.model, steps=steps).to(device).eval()

        enc_time = measure_time_ms(lambda: enc(x), device, warmup, repeat)
        dec_time = measure_time_ms(lambda: dec(z), device, warmup, repeat)
        full_time = measure_time_ms(lambda: full(x), device, warmup, repeat)

        enc_flops, enc_info = safe_flops(enc, x)
        dec_flops, dec_info = safe_flops(dec, z)
        full_flops, full_info = safe_flops(full, x)

        return {
            "params_total": count_params(self.model),
            "params_trainable": count_trainable_params(self.model),
            "encoder_params": count_params(self.model.encoder),
            "decoder_params": count_params(self.model.decoder),

            "encode_ms_per_image": enc_time / batch_size,
            "decode_ms_per_image": dec_time / batch_size,
            "encode_decode_ms_per_image": full_time / batch_size,

            "encode_flops": enc_flops,
            "decode_flops": dec_flops,
            "encode_decode_flops": full_flops,

            "encode_gflops": None if enc_flops is None else enc_flops / 1e9,
            "decode_gflops": None if dec_flops is None else dec_flops / 1e9,
            "encode_decode_gflops": None if full_flops is None else full_flops / 1e9,

            "fvcore_encode_info": enc_info,
            "fvcore_decode_info": dec_info,
            "fvcore_full_info": full_info,
        }