import torch
import os
import numpy as np
from cab.codec.abs import ImageCodecIface
from omegaconf import OmegaConf
from cab.models.bsq import config as config_utils
import torch.cuda.amp as amp
from cab.complexity import params_m, time_ms, gflops

class BSQEncodeWrapper(torch.nn.Module):
    def __init__(self, bsq_model):
        super().__init__()
        self.model = bsq_model

    def forward(self, x):
        quant, _, _ = self.model.encode(x)
        return quant


class BSQDecodeWrapper(torch.nn.Module):
    def __init__(self, bsq_model):
        super().__init__()
        self.model = bsq_model

    def forward(self, quant):
        return self.model.decode(quant)


class BSQFullWrapper(torch.nn.Module):
    def __init__(self, bsq_model):
        super().__init__()
        self.model = bsq_model

    def forward(self, x):
        xhat, _, _ = self.model(x)
        return xhat

class BSQImageTokenizer(ImageCodecIface):
    def __init__(self, quality, ckpt_path, config_path, bits_per_token, downsample_factor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality = quality
        self.ckpt_path = ckpt_path
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        # support checkpoints saved with DataParallel/DistributedDataParallel ("module." prefix)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            raw_state = checkpoint['state_dict']
        else:
            raw_state = checkpoint
        fixed_state = {}
        for k, v in raw_state.items():
            new_k = k
            if k.startswith("module."):
                new_k = k[len("module."):]
            fixed_state[new_k] = v
        self.bits_per_token = bits_per_token
        self.downsample_factor = downsample_factor
        self.bsq_config = OmegaConf.load(config_path)
        self.bsq_config.model.params.clamp_range = (-1, 1) if self.bsq_config.data.zero_mean else (0, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = config_utils.instantiate_from_config(self.bsq_config.model).to(self.device)
        self.model.load_state_dict(fixed_state)
        assert not self.bsq_config.data.zero_mean or not self.bsq_config.model.params.get('logit_laplace', False), "logit laplace mode is only compatible with the input being [0, 1]"
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):
        if x.dim() != 4:
            raise ValueError(f"BSQ expects input with shape [B, C, H, W], got {tuple(x.shape)}")

        batch_size, channels, height, width = x.shape
        patch_size = 256
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"BSQ input height and width must be multiples of {patch_size}, "
                f"got {height}x{width}"
            )

        grid_h = height // patch_size
        grid_w = width // patch_size
        patches_per_image = grid_h * grid_w
        patches = (
            x.reshape(batch_size, channels, grid_h, patch_size, grid_w, patch_size)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch_size * patches_per_image, channels, patch_size, patch_size)
        )

        with amp.autocast(enabled=not self.bsq_config.optimizer.disable_amp, dtype=torch.bfloat16 if self.bsq_config.optimizer.use_bf16 else torch.float16):
            xhat, _, info = self.model(patches)
        xhat = (
            xhat.to(torch.float32)
            .reshape(batch_size, grid_h, grid_w, channels, patch_size, patch_size)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch_size, channels, height, width)
        )

        tokens_per_patch = (patch_size // self.downsample_factor) ** 2
        bits_per_image = patches_per_image * tokens_per_patch * self.bits_per_token
        bpp_per_image = bits_per_image / (height * width)
        bpp = torch.full(
            (batch_size,),
            float(bpp_per_image),
            dtype=torch.float32,
            device=x.device,
        )

        return xhat, bpp


    def fake_input(self, image_size=256, batch_size=1, device=None):
        if device is None:
            device = self.device

        if self.bsq_config.data.zero_mean:
            return torch.rand(batch_size, 3, image_size, image_size, device=device) * 2 - 1

        return torch.rand(batch_size, 3, image_size, image_size, device=device)

    def encode_params_m(self):
        modules = [
            self.model.encoder,
            self.model.quant_embed,
            self.model.quantize,
        ]
        return sum(params_m(m) for m in modules)

    def decode_params_m(self):
        modules = [
            self.model.post_quant_embed,
            self.model.decoder,
        ]
        return sum(params_m(m) for m in modules)

    @torch.no_grad()
    def encode_tokens(self, x):
        x = x.to(self.device, dtype=torch.float)

        with amp.autocast(
            enabled=not self.bsq_config.optimizer.disable_amp,
            dtype=torch.bfloat16 if self.bsq_config.optimizer.use_bf16 else torch.float16,
        ):
            quant, _, _ = self.model.encode(x)

        return quant

    @torch.no_grad()
    def decode_tokens(self, quant):
        quant = quant.to(self.device)

        with amp.autocast(
            enabled=not self.bsq_config.optimizer.disable_amp,
            dtype=torch.bfloat16 if self.bsq_config.optimizer.use_bf16 else torch.float16,
        ):
            xhat = self.model.decode(quant)

        return xhat

    @torch.no_grad()
    def encode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, dtype=torch.float)
        enc = BSQEncodeWrapper(self.model).to(self.device).eval()

        def fn():
            with amp.autocast(
                enabled=not self.bsq_config.optimizer.disable_amp,
                dtype=torch.bfloat16 if self.bsq_config.optimizer.use_bf16 else torch.float16,
            ):
                return enc(x)

        return time_ms(fn, self.device, warmup=warmup, repeat=repeat)

    @torch.no_grad()
    def decode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, dtype=torch.float)
        quant = self.encode_tokens(x).detach()

        dec = BSQDecodeWrapper(self.model).to(self.device).eval()

        def fn():
            with amp.autocast(
                enabled=not self.bsq_config.optimizer.disable_amp,
                dtype=torch.bfloat16 if self.bsq_config.optimizer.use_bf16 else torch.float16,
            ):
                return dec(quant)

        return time_ms(fn, self.device, warmup=warmup, repeat=repeat)

    @torch.no_grad()
    def encode_gflops(self, x):
        x = x.to(self.device, dtype=torch.float)
        enc = BSQEncodeWrapper(self.model).to(self.device).eval()
        return gflops(enc, x)

    @torch.no_grad()
    def decode_gflops(self, x):
        x = x.to(self.device, dtype=torch.float)
        quant = self.encode_tokens(x).detach()

        dec = BSQDecodeWrapper(self.model).to(self.device).eval()
        return gflops(dec, quant)

    @torch.no_grad()
    def full_gflops(self, x):
        x = x.to(self.device, dtype=torch.float)
        full = BSQFullWrapper(self.model).to(self.device).eval()
        return gflops(full, x)

    @torch.no_grad()
    def full_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, dtype=torch.float)
        full = BSQFullWrapper(self.model).to(self.device).eval()

        def fn():
            with amp.autocast(
                enabled=not self.bsq_config.optimizer.disable_amp,
                dtype=torch.bfloat16 if self.bsq_config.optimizer.use_bf16 else torch.float16,
            ):
                return full(x)

        return time_ms(fn, self.device, warmup=warmup, repeat=repeat)
