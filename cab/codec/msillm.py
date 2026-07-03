import torch
import os
import numpy as np
from pathlib import Path
from cab.codec.abs import ImageCodecIface
from cab.complexity import params_m, time_ms
import pickle

def pickle_size_of(obj):
    return len(pickle.dumps(obj))

class MSILLMImageCodec(ImageCodecIface):
    def __init__(self, ckpt_name, torch_home=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ckpt_name = ckpt_name
        if torch_home:
            torch_home_path = Path(torch_home)
            if not torch_home_path.is_absolute():
                torch_home_path = Path(__file__).resolve().parents[2] / torch_home_path
            os.environ["TORCH_HOME"] = str(torch_home_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/NeuralCompression", self.ckpt_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):

        recon_list = []
        bpp_vals = []
        for i in range(x.shape[0]):
            image = x[i].unsqueeze(0)
            # prepare model for compress/decompress on appropriate devices
            self.model.update_tensor_devices('compress')
            compressed = self.model.compress(image, force_cpu=False)
            recon_i = self.model.decompress(compressed, force_cpu=False).clamp(0.0, 1.0)
            self.model.update_tensor_devices('forward')

            recon_list.append(recon_i)  # recon_i shape: (1, C, H, W)

            num_bytes = pickle_size_of(compressed)
            bpp = num_bytes * 8 / (image.shape[0] * image.shape[-2] * image.shape[-1])
            bpp_vals.append(float(bpp))

        # concat per-image reconstructions into a full-batch tensor
        xhat = torch.cat(recon_list, dim=0)  # shape: (batch, C, H, W)
        bpp = torch.tensor(bpp_vals, dtype=torch.float32, device=x.device)  # shape: (batch,)

        return xhat, bpp
    
    def encode_params_m(self):
        modules = [
            self.model.encoder,
            self.model.hyper_analysis,
        ]
        return sum(params_m(m) for m in modules if m is not None)

    def decode_params_m(self):
        modules = [
            self.model.hyper_synthesis_mean,
            self.model.hyper_synthesis_scale,
            self.model.decoder,
        ]
        return sum(params_m(m) for m in modules if m is not None)

    @torch.no_grad()
    def encode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, dtype=torch.float)

        def fn():
            self.model.update_tensor_devices("compress")
            return self.model.compress(x, force_cpu=False)

        out = time_ms(
            fn,
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

        self.model.update_tensor_devices("forward")
        return out
    
    @torch.no_grad()
    def decode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, dtype=torch.float)

        self.model.update_tensor_devices("compress")
        compressed = self.model.compress(x, force_cpu=False)

        def fn():
            return self.model.decompress(
                compressed,
                force_cpu=False,
            ).clamp(0.0, 1.0)

        out = time_ms(
            fn,
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

        self.model.update_tensor_devices("forward")
        return out
    
    @torch.no_grad()
    def encode_gflops(self, x):
        from fvcore.nn import FlopCountAnalysis

        x = x.to(self.device, dtype=torch.float)

        encoder_flops = FlopCountAnalysis(
            self.model.encoder,
            x,
        ).total()

        latent = self.model.encoder(x)

        hyper_analysis_flops = FlopCountAnalysis(
            self.model.hyper_analysis,
            latent,
        ).total()

        return (encoder_flops + hyper_analysis_flops) / 1e9

    @torch.no_grad()
    def decode_gflops(self, x):
        from fvcore.nn import FlopCountAnalysis

        x = x.to(self.device, dtype=torch.float)

        latent = self.model.encoder(x)
        hyper_latent = self.model.hyper_analysis(latent)
        hyper_latent_hat = torch.round(hyper_latent)

        mean_flops = FlopCountAnalysis(
            self.model.hyper_synthesis_mean,
            hyper_latent_hat,
        ).total()

        scale_flops = FlopCountAnalysis(
            self.model.hyper_synthesis_scale,
            hyper_latent_hat,
        ).total()

        latent_hat = torch.round(latent)

        decoder_flops = FlopCountAnalysis(
            self.model.decoder,
            latent_hat,
        ).total()

        return (mean_flops + scale_flops + decoder_flops) / 1e9
