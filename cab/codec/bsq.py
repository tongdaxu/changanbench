import torch
import os
import numpy as np
from cab.codec.abs import (
    ImageCodecIface,
    merge_image_patches,
    patch_bits_to_image_bpp,
    split_image_into_patches,
)
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
    PATCH_SIZE = 256

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
        batch, channels, height, width = x.shape
        patches = split_image_into_patches(x, self.PATCH_SIZE)
        patches_per_image = patches.shape[1]
        flat_patches = patches.reshape(
            batch * patches_per_image,
            channels,
            self.PATCH_SIZE,
            self.PATCH_SIZE,
        )

        with amp.autocast(enabled=not self.bsq_config.optimizer.disable_amp, dtype=torch.bfloat16 if self.bsq_config.optimizer.use_bf16 else torch.float16):
            reconstructed_patches, _, info = self.model(flat_patches)
        reconstructed_patches = reconstructed_patches.to(torch.float32)

        indices = info.get("indices") if isinstance(info, dict) else None
        if isinstance(indices, torch.Tensor):
            if indices.shape[0] != flat_patches.shape[0]:
                raise RuntimeError(
                    "BSQ token indices do not preserve the patch batch "
                    f"dimension: got {tuple(indices.shape)} for "
                    f"{flat_patches.shape[0]} patches."
                )
            tokens_per_patch = indices.reshape(indices.shape[0], -1).shape[1]
        else:
            # Compatibility fallback for older BSQ model variants that do not
            # expose indices. This matches the configured encoder token grid.
            tokens_per_patch = (
                self.PATCH_SIZE // self.downsample_factor
            ) ** 2

        patch_bits = torch.full(
            (flat_patches.shape[0],),
            float(tokens_per_patch * self.bits_per_token),
            dtype=torch.float32,
            device=x.device,
        )
        reconstructed_patches = reconstructed_patches.reshape_as(patches)
        xhat = merge_image_patches(
            reconstructed_patches,
            image_size=(height, width),
        )
        bpp = patch_bits_to_image_bpp(
            patch_bits,
            batch_size=batch,
            patches_per_image=patches_per_image,
            image_size=(height, width),
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
