import torch
import os
import numpy as np
from cab.codec.abs import ImageCodecIface 
from cab.models.infinity.models.bsq_vae.vae import vae_model
from cab.complexity import params_m, time_ms, gflops

class InfinityEncodeWrapper(torch.nn.Module):
    def __init__(self, vae, scale_schedule):
        super().__init__()
        self.vae = vae
        self.scale_schedule = scale_schedule

    def forward(self, x):
        h, z, all_indices, all_bit_indices, residual_norm_per_scale, var_input = self.vae.encode(
            x,
            scale_schedule=self.scale_schedule,
        )
        return z


class InfinityDecodeWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z)


class InfinityImageTokenizer(ImageCodecIface):
    def __init__(self, vqgan_ckpt, codebook_dim, downsample_factor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.codebook_dim = codebook_dim
        self.downsample_factor = downsample_factor
        codebook_size = 2 ** self.codebook_dim
        schedule_mode = "dynamic"
        vae = vae_model(vqgan_ckpt, schedule_mode, self.codebook_dim, codebook_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = vae.to(self.device)
        vae.eval()
        

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):

        x = x.to(self.device, non_blocking=True)
        xhat, _ = self.vae(x)

        bits_per_token = self.codebook_dim
        num_tokens = (x.size(2) // self.downsample_factor) * (x.size(3) // self.downsample_factor)
        total_bits = num_tokens * bits_per_token
        bpp = total_bits / (x.size(2) * x.size(3))
        bpp = torch.tensor([bpp], dtype=torch.float32, device=x.device)
        return xhat, bpp
    
    def fake_input(self, image_size=256, batch_size=1, device=None):
        if device is None:
            device = self.device
        return torch.rand(batch_size, 3, image_size, image_size, device=device) * 2 - 1

    def _scale_schedule(self, x):
        h = x.shape[2] // self.downsample_factor
        w = x.shape[3] // self.downsample_factor

        # Infinity VAE 的 encode 需要 scale_schedule。
        # 对 2D image，通常用 [(1, h, w)]。
        return [(1, h, w)]

    def encode_params_m(self):
        modules = [
            self.vae.encoder,
            self.vae.quantizer,
        ]
        return sum(params_m(m) for m in modules)

    def decode_params_m(self):
        return params_m(self.vae.decoder)

    @torch.no_grad()
    def encode_tokens(self, x):
        x = x.to(self.device, non_blocking=True)
        scale_schedule = self._scale_schedule(x)

        h, z, all_indices, all_bit_indices, residual_norm_per_scale, var_input = self.vae.encode(
            x,
            scale_schedule=scale_schedule,
        )
        return z

    @torch.no_grad()
    def decode_tokens(self, z):
        z = z.to(self.device)
        return self.vae.decode(z)

    @torch.no_grad()
    def encode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, non_blocking=True)
        scale_schedule = self._scale_schedule(x)

        enc = InfinityEncodeWrapper(
            self.vae,
            scale_schedule=scale_schedule,
        ).to(self.device).eval()

        return time_ms(
            lambda: enc(x),
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

    @torch.no_grad()
    def decode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, non_blocking=True)
        z = self.encode_tokens(x).detach()

        dec = InfinityDecodeWrapper(self.vae).to(self.device).eval()

        return time_ms(
            lambda: dec(z),
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

    @torch.no_grad()
    def encode_gflops(self, x):
        x = x.to(self.device, non_blocking=True)
        scale_schedule = self._scale_schedule(x)

        enc = InfinityEncodeWrapper(
            self.vae,
            scale_schedule=scale_schedule,
        ).to(self.device).eval()

        return gflops(enc, x)

    @torch.no_grad()
    def decode_gflops(self, x):
        x = x.to(self.device, non_blocking=True)
        z = self.encode_tokens(x).detach()

        dec = InfinityDecodeWrapper(self.vae).to(self.device).eval()
        return gflops(dec, z)