import torch
import os
import numpy as np
from cab.codec.abs import ImageCodecIface
from cab.models.tcm.models import TCM
from cab.complexity import params_m, time_ms, gflops

class TCMEncodeFLOPsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model.g_a(x)
        y_shape = y.shape[2:]

        z = self.model.h_a(y)
        z_hat = torch.round(z)

        latent_scales = self.model.h_scale_s(z_hat)
        latent_means = self.model.h_mean_s(z_hat)

        y_slices = y.chunk(self.model.num_slices, 1)
        y_hat_slices = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (
                y_hat_slices
                if self.model.max_support_slices < 0
                else y_hat_slices[:self.model.max_support_slices]
            )

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.model.atten_mean[slice_index](mean_support)
            mu = self.model.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.model.atten_scale[slice_index](scale_support)
            scale = self.model.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            y_hat_slice = torch.round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.model.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)

            y_hat_slice = y_hat_slice + lrp
            y_hat_slices.append(y_hat_slice)

        return torch.cat(y_hat_slices, dim=1)

class TCMDecodeFLOPsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z_hat):
        latent_scales = self.model.h_scale_s(z_hat)
        latent_means = self.model.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_hat_slices = []

        for slice_index in range(self.model.num_slices):
            support_slices = (
                y_hat_slices
                if self.model.max_support_slices < 0
                else y_hat_slices[:self.model.max_support_slices]
            )

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.model.atten_mean[slice_index](mean_support)
            mu = self.model.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.model.atten_scale[slice_index](scale_support)
            scale = self.model.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            y_hat_slice = torch.zeros_like(mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.model.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)

            y_hat_slice = y_hat_slice + lrp
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.model.g_s(y_hat)

        return x_hat

class TCMImageCodec(ImageCodecIface):
    def __init__(self, quality, ckpt_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality = quality
        self.ckpt_path = ckpt_path

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dictory = {}
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.model = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=64, M=320).to(self.device)
        for k, v in ckpt["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        self.model.load_state_dict(dictory)

        self.model.eval()
        self.model.update()

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):
        x = x.to(self.device, dtype=torch.float)
        recon_list = []
        bpp_vals = []
        for i in range(x.shape[0]):
            img = x[i:i+1]
            out_enc = self.model.compress(img)
            out_dec = self.model.decompress(out_enc["strings"], out_enc["shape"])
            xhat = out_dec["x_hat"].clamp(0, 1)
            recon_list.append(xhat)
            bpp_vals.append(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / (x.size(2) * x.size(3)))
        xhat = torch.cat(recon_list, dim=0)
        bpp = torch.tensor(bpp_vals, dtype=torch.float32, device=x.device)
        return xhat, bpp
    
    def fake_input(self, image_size=256, batch_size=1, device=None):
        if device is None:
            device = self.device
        return torch.rand(batch_size, 3, image_size, image_size, device=device)

    def encode_params_m(self):
        modules = [
            self.model.g_a,
            self.model.h_a,
            self.model.entropy_bottleneck,
            self.model.h_scale_s,
            self.model.h_mean_s,
            self.model.atten_mean,
            self.model.atten_scale,
            self.model.cc_mean_transforms,
            self.model.cc_scale_transforms,
            self.model.lrp_transforms,
        ]
        return sum(params_m(m) for m in modules)

    def decode_params_m(self):
        modules = [
            self.model.entropy_bottleneck,
            self.model.h_scale_s,
            self.model.h_mean_s,
            self.model.atten_mean,
            self.model.atten_scale,
            self.model.cc_mean_transforms,
            self.model.cc_scale_transforms,
            self.model.lrp_transforms,
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
        wrapper = TCMEncodeFLOPsWrapper(self.model).to(self.device).eval()
        return gflops(wrapper, x)

    @torch.no_grad()
    def decode_gflops(self, x):
        x = x.to(self.device, dtype=torch.float)

        y = self.model.g_a(x)
        z = self.model.h_a(y)
        z_hat = torch.round(z)

        wrapper = TCMDecodeFLOPsWrapper(self.model).to(self.device).eval()
        return gflops(wrapper, z_hat)