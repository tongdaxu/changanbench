import torch
import os
import numpy as np
from cab.codec.abs import ImageCodecIface
from cab.models.var.models import build_vae_var
import os.path as osp
from pathlib import Path
from cab.complexity import params_m, time_ms, gflops


def _project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def _resolve_var_weight(filename, local_path=None, token=None, revision=None):
    if local_path:
        path = _project_path(local_path)
        if path.exists():
            return str(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path = None

    from huggingface_hub import hf_hub_download

    kwargs = {"token": token, "revision": revision}
    if path is not None:
        kwargs.update({"local_dir": str(path.parent), "local_dir_use_symlinks": False})
    return hf_hub_download("FoundationVision/var", filename, **kwargs)

class VAREncodeWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        f = self.vae.quant_conv(self.vae.encoder(x))
        f_hat = self.vae.quantize.f_to_idxBl_or_fhat(
            f,
            to_fhat=True,
            v_patch_nums=None,
        )[-1]
        return f_hat


class VARDecodeWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, f_hat):
        return self.vae.decoder(
            self.vae.post_quant_conv(f_hat)
        ).clamp(-1, 1)


class VARFullWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        return self.vae.img_to_reconstructed_img(x, last_one=True)

class VARImageTokenizer(ImageCodecIface):
    def __init__(
        self,
        quality,
        vae_ckpt="/data9-2/BenchmarkData/weights/var/vae_ch160v4096z32.pth",
        model_depth=16,
        hf_token=None,
        hf_revision=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.quality = quality
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MODEL_DEPTH = int(model_depth) #16, 20, 24, 30
        vae_ckpt = _resolve_var_weight(
            "vae_ch160v4096z32.pth",
            vae_ckpt,
            token=hf_token,
            revision=hf_revision,
        )
        
        # build vae, var
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        
        vae, var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
            device=self.device, patch_nums=patch_nums,
            num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
        )

        # load checkpoints
        vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
        self.model = vae.eval()

    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device, dtype=torch.float)
        
        xhat = self.model.img_to_reconstructed_img(x, last_one=True)
        
        patch_size = 16
        num_codebook_entries = 4096 
        bits_per_token = np.log2(num_codebook_entries)

        num_tokens = (x.shape[2] // patch_size) ** 2
        total_bits = num_tokens * bits_per_token

        bpp = total_bits / (x.shape[2] * x.shape[3])
        
        return xhat, torch.tensor([bpp], dtype=torch.float32, device=x.device)

    def fake_input(self, image_size=256, batch_size=1, device=None):
        if device is None:
            device = self.device
        return torch.rand(batch_size, 3, image_size, image_size, device=device) * 2 - 1

    def encode_params_m(self):
        modules = [
            self.model.encoder,
            self.model.quant_conv,
            self.model.quantize,
        ]
        return sum(params_m(m) for m in modules)

    def decode_params_m(self):
        modules = [
            self.model.post_quant_conv,
            self.model.decoder,
        ]
        return sum(params_m(m) for m in modules)

    @torch.no_grad()
    def encode_tokens(self, x):
        x = x.to(self.device, dtype=torch.float)

        f = self.model.quant_conv(self.model.encoder(x))
        f_hat = self.model.quantize.f_to_idxBl_or_fhat(
            f,
            to_fhat=True,
            v_patch_nums=None,
        )[-1]

        return f_hat

    @torch.no_grad()
    def decode_tokens(self, f_hat):
        f_hat = f_hat.to(self.device, dtype=torch.float)
        return self.model.decoder(
            self.model.post_quant_conv(f_hat)
        ).clamp(-1, 1)

    @torch.no_grad()
    def encode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, dtype=torch.float)
        enc = VAREncodeWrapper(self.model).to(self.device).eval()

        return time_ms(
            lambda: enc(x),
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

    @torch.no_grad()
    def decode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, dtype=torch.float)
        f_hat = self.encode_tokens(x).detach()

        dec = VARDecodeWrapper(self.model).to(self.device).eval()

        return time_ms(
            lambda: dec(f_hat),
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

    @torch.no_grad()
    def encode_gflops(self, x):
        x = x.to(self.device, dtype=torch.float)

        enc = VAREncodeWrapper(self.model).to(self.device).eval()

        # 如果直接 trace wrapper 失败，再退化成 encoder + quant_conv
        flops = gflops(enc, x)
        if flops is not None:
            return flops

        from fvcore.nn import FlopCountAnalysis

        encoder_flops = FlopCountAnalysis(self.model.encoder, x).total()
        enc_out = self.model.encoder(x)
        quant_conv_flops = FlopCountAnalysis(self.model.quant_conv, enc_out).total()

        return (encoder_flops + quant_conv_flops) / 1e9

    @torch.no_grad()
    def decode_gflops(self, x):
        x = x.to(self.device, dtype=torch.float)
        f_hat = self.encode_tokens(x).detach()

        dec = VARDecodeWrapper(self.model).to(self.device).eval()

        flops = gflops(dec, f_hat)
        if flops is not None:
            return flops

        from fvcore.nn import FlopCountAnalysis

        post_quant_flops = FlopCountAnalysis(
            self.model.post_quant_conv,
            f_hat,
        ).total()

        decoder_input = self.model.post_quant_conv(f_hat)

        decoder_flops = FlopCountAnalysis(
            self.model.decoder,
            decoder_input,
        ).total()

        return (post_quant_flops + decoder_flops) / 1e9

    @torch.no_grad()
    def full_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, dtype=torch.float)
        full = VARFullWrapper(self.model).to(self.device).eval()

        return time_ms(
            lambda: full(x),
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

    @torch.no_grad()
    def full_gflops(self, x):
        x = x.to(self.device, dtype=torch.float)
        full = VARFullWrapper(self.model).to(self.device).eval()
        return gflops(full, x)
