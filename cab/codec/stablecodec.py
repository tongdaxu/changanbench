import torch
import os
import math
import numpy as np
from cab.codec.abs import ImageCodecIface
from cab.models.stablecodec_src.StableCodec import StableCodec
from types import SimpleNamespace
from pathlib import Path
from cab.models.stablecodec_src.my_utils.compress_utils import write_body, read_body, filesize
from types import SimpleNamespace
from cab.complexity import params_m, time_ms, gflops

def compress_one_image(net, bin_path, ori_h, ori_w, img_name, x):
    with torch.no_grad():
        output_dict = net.compress(x)
    shape = output_dict["shape"]
    os.makedirs(bin_path, exist_ok=True)
    output = os.path.join(bin_path, img_name)
    with Path(output).open("wb") as f:
        write_body(f, shape, output_dict["strings"])
    size = filesize(output)
    bpp = float(size) * 8 / (ori_h * ori_w)
    return bpp


def decompress_one_image(net, bin_path, ori_h, ori_w, img_name, prompt):
    output = os.path.join(bin_path, img_name)
    with Path(output).open("rb") as f:
        strings, shape = read_body(f)
    with torch.no_grad():
        out_img = net.decompress(strings, shape, prompt)
    out_img = out_img[:, :, 0 : ori_h, 0 : ori_w].float().cpu().detach()
    # out_img = (out_img * 0.5 + 0.5).float().cpu().detach()
    return out_img

def _sum_params_m(modules):
    return sum(params_m(m) for m in modules if m is not None)


class StableCodecEncodeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.codec = model.codec

    def forward(self, x):
        latent2 = self.model.aux_codec((x + 1) / 2).detach()
        lq_latent = (
            self.model.vae.encode(x).latent_dist.mode()
            * self.model.vae.config.scaling_factor
        )

        y = self.codec.g_a(lq_latent, latent2)
        z = self.codec.h_a(y)
        z_hat = torch.round(z)

        B, C, H, W = y.shape
        mask_0, mask_1, mask_2, mask_3 = self.codec.get_mask_four_parts(
            B, C, H, W, device=y.device
        )

        base = self.codec.h_s(z_hat)

        y_hat_slices = []
        for idx, mask in enumerate([mask_0, mask_1, mask_2, mask_3]):
            means, scales = self.codec.adapter_out[idx](
                self.codec.g_c(self.codec.adapter_in[idx](base))
            ).chunk(2, 1)

            y_hat = torch.round(y - means) + means
            y_hat = y_hat * mask

            lrp = self.codec.LRP[idx](
                torch.cat([y_hat, base], dim=1)
            ) * mask

            y_hat = y_hat + 0.5 * torch.tanh(lrp)

            if idx < 3:
                base = base * (1 - mask) + y_hat

            y_hat_slices.append(y_hat)

        return sum(y_hat_slices)


class StableCodecLatentDecodeWrapper(torch.nn.Module):
    def __init__(self, codec):
        super().__init__()
        self.codec = codec

    def forward(self, z_hat):
        B, C, H, W = z_hat.shape

        mask_0, mask_1, mask_2, mask_3 = self.codec.get_mask_four_parts(
            B,
            C * 2,
            H * 4,
            W * 4,
            device=z_hat.device,
        )

        base = self.codec.h_s(z_hat)

        y_hat_slices = []
        for idx, mask in enumerate([mask_0, mask_1, mask_2, mask_3]):
            means, scales = self.codec.adapter_out[idx](
                self.codec.g_c(self.codec.adapter_in[idx](base))
            ).chunk(2, 1)

            y_hat = torch.zeros_like(means[:, : mask.shape[1]]) * mask

            lrp = self.codec.LRP[idx](
                torch.cat([y_hat, base], dim=1)
            ) * mask

            y_hat = y_hat + 0.5 * torch.tanh(lrp)

            if idx < 3:
                base = base * (1 - mask) + y_hat

            y_hat_slices.append(y_hat)

        y_hat = sum(y_hat_slices)

        lq_latent_hat = self.codec.g_s(y_hat)
        res = self.codec.aux(y_hat)

        return lq_latent_hat, res


class StableCodecUNetWrapper(torch.nn.Module):
    def __init__(self, model, pos_caption_enc):
        super().__init__()
        self.model = model
        self.pos_caption_enc = pos_caption_enc

    def forward(self, lq_latent_hat):
        return self.model.unet(
            lq_latent_hat,
            self.model.timesteps,
            encoder_hidden_states=self.pos_caption_enc,
        ).sample


class StableCodecVAEDecodeWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(
            z / self.vae.config.scaling_factor
        ).sample

class StableCodecImageCodec(ImageCodecIface):
    def __init__(self, sd_path, elic_path, codec_path, rec_path, bin_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sd_path = sd_path
        self.elic_path = elic_path
        self.codec_path = codec_path
        self.rec_path = rec_path
        self.bin_path = bin_path
        os.makedirs(self.rec_path, exist_ok=True)
        os.makedirs(self.bin_path, exist_ok=True)
        default_args = dict(
        sd_path=self.sd_path,
        elic_path=self.elic_path,
        codec_path=self.codec_path,
        img_path="",
        rec_path=self.rec_path,
        bin_path=self.bin_path,

        lora_rank_unet=32,
        lora_rank_vae=16,
        vae_decoder_tiled_size=160,
        vae_encoder_tiled_size=1024,
        latent_tiled_size=96,
        latent_tiled_overlap=32,
        pos_prompt="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.",
        seed=None,
        enable_xformers_memory_efficient_attention=False,
        set_grads_to_none=False,
        lambda_rate=0.5,
        color_fix=False,
    )

        default_args.update(kwargs)
        args_ns = SimpleNamespace(**default_args)

        self.model = StableCodec(sd_path=args_ns.sd_path, args=args_ns)
        self.model.cuda().eval()
        self.model.codec.update(force=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_tag_prompt = [1]

        try:
            self.model.set_eval()
        except Exception:
            self.model.eval()

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):        
        recon_list = []
        bpp_vals = []
        pos_tag_prompt = [1]

        batch = x.cuda()
        for i in range(batch.shape[0]):
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
            img_i = batch[i].unsqueeze(0)  # shape (1,C,H,W)
            ori_h, ori_w = img_i.shape[2:]
            fname = f"{i}_r{rank}"

            try:
                rate = compress_one_image(self.model, self.bin_path, ori_h, ori_w, fname, img_i)
                out_img = decompress_one_image(self.model, self.bin_path, ori_h, ori_w, fname, pos_tag_prompt)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    torch.cuda.empty_cache()
                    # fallback: use input as reconstruction and bpp 0
                    out_img = img_i.cpu()
                    rate = 0.0
                else:
                    raise

            # ensure reconstruction on same device as input batch
            out_img = out_img.to(x.device)
            recon_list.append(out_img)
            bpp_vals.append(rate)

        # concat per-image reconstructions into a full-batch tensor
        xhat = torch.cat(recon_list, dim=0)  # shape: (batch, C, H, W)
        bpp = torch.tensor(bpp_vals, dtype=torch.float32, device=x.device)  # shape: (batch,)

        return xhat, bpp
    
    def fake_input(self, image_size=256, batch_size=1, device=None):
        if device is None:
            device = self.device

        return torch.rand(
            batch_size,
            3,
            image_size,
            image_size,
            device=device,
        ) * 2 - 1

    def encode_params_m(self):
        codec = self.model.codec
        vae = self.model.vae

        modules = [
            self.model.aux_codec,
            vae.encoder,
            getattr(vae, "quant_conv", None),
            codec.g_a,
            codec.h_a,
            codec.h_s,
            codec.adapter_in,
            codec.g_c,
            codec.adapter_out,
            codec.LRP,
            codec.entropy_bottleneck,
            codec.gaussian_conditional,
        ]

        return _sum_params_m(modules)

    def decode_params_m(self):
        codec = self.model.codec
        vae = self.model.vae

        modules = [
            codec.h_s,
            codec.adapter_in,
            codec.g_c,
            codec.adapter_out,
            codec.LRP,
            codec.g_s,
            codec.aux,
            codec.entropy_bottleneck,
            codec.gaussian_conditional,
            self.model.unet,
            getattr(vae, "post_quant_conv", None),
            vae.decoder,
        ]

        return _sum_params_m(modules)

    @torch.no_grad()
    def encode_time_ms(self, x, warmup=3, repeat=10):
        x = x[:1].to(self.device, dtype=torch.float)

        return time_ms(
            lambda: self.model.compress(x),
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

    @torch.no_grad()
    def decode_time_ms(self, x, warmup=3, repeat=10):
        x = x[:1].to(self.device, dtype=torch.float)

        output_dict = self.model.compress(x)
        strings = output_dict["strings"]
        shape = output_dict["shape"]

        return time_ms(
            lambda: self.model.decompress(
                strings,
                shape,
                self.pos_tag_prompt,
            ),
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

    @torch.no_grad()
    def encode_gflops(self, x):
        x = x[:1].to(self.device, dtype=torch.float)

        enc = StableCodecEncodeWrapper(
            self.model
        ).to(self.device).eval()

        return gflops(enc, x)

    @torch.no_grad()
    def decode_gflops(self, x):
        x = x[:1].to(self.device, dtype=torch.float)

        codec = self.model.codec

        # 构造真实 z_hat shape
        latent2 = self.model.aux_codec((x + 1) / 2).detach()

        lq_latent = (
            self.model.vae.encode(x).latent_dist.mode()
            * self.model.vae.config.scaling_factor
        )

        y = codec.g_a(lq_latent, latent2)
        z = codec.h_a(y)
        z_hat = torch.round(z).detach()

        # 1. latent codec decode neural FLOPs
        latent_dec = StableCodecLatentDecodeWrapper(
            codec
        ).to(self.device).eval()

        latent_decode_g = gflops(
            latent_dec,
            z_hat,
        )

        latent_out = latent_dec(z_hat)
        lq_latent_hat, res = latent_out

        # 2. UNet one-step FLOPs
        pos_caption_enc = self.model.pos_caption_enc.to(self.device)

        if pos_caption_enc.shape[0] != lq_latent_hat.shape[0]:
            pos_caption_enc = pos_caption_enc.repeat(
                lq_latent_hat.shape[0],
                1,
                1,
            )

        unet_wrapper = StableCodecUNetWrapper(
            self.model,
            pos_caption_enc,
        ).to(self.device).eval()

        unet_g = gflops(
            unet_wrapper,
            lq_latent_hat.detach(),
        )

        model_pred = unet_wrapper(
            lq_latent_hat.detach()
        )

        x_denoised = self.model.sched.step(
            model_pred,
            self.model.timesteps,
            lq_latent_hat[:, :4],
            return_dict=True,
        ).prev_sample + res

        # 3. VAE decoder FLOPs
        vae_dec = StableCodecVAEDecodeWrapper(
            self.model.vae
        ).to(self.device).eval()

        vae_g = gflops(
            vae_dec,
            x_denoised.detach(),
        )

        print(
            "[StableCodec Decode FLOPs] "
            f"latent_decode={latent_decode_g}, "
            f"unet={unet_g}, "
            f"vae_decode={vae_g}"
        )

        if latent_decode_g is None or unet_g is None or vae_g is None:
            return None

        return latent_decode_g + unet_g + vae_g
    
