from cab.models.hific_src import model
import torch
import os
import numpy as np
from cab.codec.abs import ImageCodecIface
import compressai
from compressai.zoo import load_state_dict
from cab.models.ELIC.Network import TestModel
from cab.complexity import params_m, time_ms, gflops

class ELICEncodeFLOPsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model.g_a(x)
        z = self.model.h_a(y)

        # 编码端为了得到 y 的概率参数，也需要先得到 z_hat 和 h_s(z_hat)
        z_hat = torch.round(z)
        latent_means, latent_scales = self.model.h_s(z_hat).chunk(2, 1)

        B, C, H, W = y.shape

        y_slices = torch.split(y, self.model.groups[1:], dim=1)

        ctx_params_anchor = torch.zeros(
            B,
            C * 2,
            H,
            W,
            device=x.device,
            dtype=x.dtype,
        )

        ctx_params_anchor_split = torch.split(
            ctx_params_anchor,
            [2 * i for i in self.model.groups[1:]],
            dim=1,
        )

        y_hat_slices = []

        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support = torch.cat([latent_means, latent_scales], dim=1)
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.model.cc_transforms[slice_index - 1](support_slices)
                support_mean, support_scale = support_slices_ch.chunk(2, 1)
                support = torch.cat(
                    [support_mean, support_scale, latent_means, latent_scales],
                    dim=1,
                )
            else:
                support_slices = torch.cat(
                    [y_hat_slices[0], y_hat_slices[slice_index - 1]],
                    dim=1,
                )
                support_slices_ch = self.model.cc_transforms[slice_index - 1](support_slices)
                support_mean, support_scale = support_slices_ch.chunk(2, 1)
                support = torch.cat(
                    [support_mean, support_scale, latent_means, latent_scales],
                    dim=1,
                )

            means_anchor, scales_anchor = self.model.ParamAggregation[slice_index](
                torch.cat([ctx_params_anchor_split[slice_index], support], dim=1)
            ).chunk(2, 1)

            y_anchor_hat = torch.zeros_like(means_anchor)

            masked_context = self.model.context_prediction[slice_index](y_anchor_hat)

            means_non_anchor, scales_non_anchor = self.model.ParamAggregation[slice_index](
                torch.cat([masked_context, support], dim=1)
            ).chunk(2, 1)

            y_slice_hat = torch.zeros_like(means_anchor)
            y_hat_slices.append(y_slice_hat)

        return torch.cat(y_hat_slices, dim=1)
    
class ELICDecodeFLOPsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z_hat):
        latent_means, latent_scales = self.model.h_s(z_hat).chunk(2, 1)

        B, _, H_z, W_z = z_hat.shape
        H, W = H_z * 4, W_z * 4

        ctx_params_anchor = torch.zeros(
            B,
            self.model.M * 2,
            H,
            W,
            device=z_hat.device,
            dtype=z_hat.dtype,
        )

        ctx_params_anchor_split = torch.split(
            ctx_params_anchor,
            [2 * i for i in self.model.groups[1:]],
            dim=1,
        )

        y_hat_slices = []

        for slice_index in range(len(self.model.groups) - 1):
            if slice_index == 0:
                support = torch.cat([latent_means, latent_scales], dim=1)
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.model.cc_transforms[slice_index - 1](support_slices)
                support_mean, support_scale = support_slices_ch.chunk(2, 1)
                support = torch.cat(
                    [support_mean, support_scale, latent_means, latent_scales],
                    dim=1,
                )
            else:
                support_slices = torch.cat(
                    [y_hat_slices[0], y_hat_slices[slice_index - 1]],
                    dim=1,
                )
                support_slices_ch = self.model.cc_transforms[slice_index - 1](support_slices)
                support_mean, support_scale = support_slices_ch.chunk(2, 1)
                support = torch.cat(
                    [support_mean, support_scale, latent_means, latent_scales],
                    dim=1,
                )

            means_anchor, scales_anchor = self.model.ParamAggregation[slice_index](
                torch.cat([ctx_params_anchor_split[slice_index], support], dim=1)
            ).chunk(2, 1)

            y_anchor_decode = torch.zeros_like(means_anchor)

            masked_context = self.model.context_prediction[slice_index](y_anchor_decode)

            means_non_anchor, scales_non_anchor = self.model.ParamAggregation[slice_index](
                torch.cat([masked_context, support], dim=1)
            ).chunk(2, 1)

            y_slice_hat = torch.zeros_like(means_anchor)
            y_hat_slices.append(y_slice_hat)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.model.g_s(y_hat)

        return x_hat

class ELICImageCodec(ImageCodecIface):
    def __init__(self, quality, ckpt_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality = quality
        self.ckpt_path = ckpt_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        compressai.set_entropy_coder(compressai.available_entropy_coders()[0])
        state_dict = load_state_dict(torch.load(self.ckpt_path))
        model_cls = TestModel()
        self.model = model_cls.from_state_dict(state_dict).eval().to(self.device)
        if hasattr(self.model, "update"):
            self.model.update(force=True)

    def fake_input(self, image_size=256, batch_size=1, device=None):
        if device is None:
            device = self.device
        return torch.rand(batch_size, 3, image_size, image_size, device=device)
    
    @torch.no_grad()
    def forward(self, x, *args, **kwargs):

        x = x.to(self.device, dtype=torch.float)
        recon_list = []
        bpp_vals = []
        for i in range(x.shape[0]):
            img = x[i:i+1]
        
            out_enc = self.model.compress(img)
            out_dec = self.model.decompress(out_enc["strings"], out_enc["shape"])
            
            
            xhat = out_dec["x_hat"]
            recon_list.append(xhat)
            bpp_vals.append(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / (img.size(2) * img.size(3)))
        xhat = torch.cat(recon_list, dim=0)
        bpp = torch.tensor(bpp_vals, dtype=torch.float32, device=x.device)
        return xhat, bpp
    
    def encode_params_m(self):
        modules = [
            self.model.g_a,
            self.model.h_a,
            self.model.entropy_bottleneck,
            self.model.h_s,
            self.model.cc_transforms,
            self.model.context_prediction,
            self.model.ParamAggregation,
        ]
        return sum(params_m(m) for m in modules)

    def decode_params_m(self):
        modules = [
            self.model.entropy_bottleneck,
            self.model.h_s,
            self.model.cc_transforms,
            self.model.context_prediction,
            self.model.ParamAggregation,
            self.model.g_s,
        ]
        return sum(params_m(m) for m in modules)

    @torch.no_grad()
    def encode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, dtype=torch.float)

        return time_ms(
            lambda: self.model.compress(x),
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

    @torch.no_grad()
    def decode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, dtype=torch.float)

        out = self.model.compress(x)
        strings = out["strings"]
        shape = out["shape"]

        return time_ms(
            lambda: self.model.decompress(strings, shape),
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

    @torch.no_grad()
    def encode_gflops(self, x):
        x = x.to(self.device, dtype=torch.float)

        enc_wrapper = ELICEncodeFLOPsWrapper(self.model).to(self.device).eval()

        return gflops(enc_wrapper, x)

    @torch.no_grad()
    def decode_gflops(self, x):
        x = x.to(self.device, dtype=torch.float)

        # 用 encoder 生成 z_hat 的 shape，FLOPs 不统计这一步
        y = self.model.g_a(x)
        z = self.model.h_a(y)
        z_hat = torch.round(z)

        dec_wrapper = ELICDecodeFLOPsWrapper(self.model).to(self.device).eval()

        return gflops(dec_wrapper, z_hat)