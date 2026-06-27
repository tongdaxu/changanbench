import torch
import os
import numpy as np
from cab.codec.abs import ImageCodecIface
from cab.models.hific_src.model import Model 
from cab.models.hific_src.helpers import utils
from cab.models.hific_src.default_config import ModelModes
from collections import defaultdict
from cab.complexity import params_m, time_ms

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class HiFiCImageCodec(ImageCodecIface):
    def __init__(self, quality, ckpt_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality = quality
        self.ckpt_path = ckpt_path

        self.device = utils.get_device()
        self.logger = utils.logger_setup(logpath=os.path.join('logs'), filepath=os.path.abspath(__file__))
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        model_args = ckpt['args']
        model_args = Struct(**model_args)  

        self.model = Model(
            args=model_args,
            logger=self.logger,
            storage_train=defaultdict(list),
            storage_test=defaultdict(list),
            model_mode=ModelModes.EVALUATION,
            model_type=model_args.model_type,  
        ).to(self.device)
        
        self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        self.model.eval()

        # 构建熵编码概率表，否则 compress/decompress 可能很慢或报错
        if hasattr(self.model, "Hyperprior"):
            hp = self.model.Hyperprior
            if hasattr(hp, "hyperprior_entropy_model"):
                hp.hyperprior_entropy_model.build_tables()

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):

        x = x.to(self.device, dtype=torch.float)
        xhat, bpp = self.model(x)
        bpp = torch.tensor([bpp], dtype=torch.float32, device=x.device)
        return xhat, bpp
    
    def encode_params_m(self):
        modules = [
            self.model.Encoder,
            self.model.Hyperprior.analysis_net,
        ]
        return sum(params_m(m) for m in modules if m is not None)

    def decode_params_m(self):
        modules = [
            self.model.Hyperprior.synthesis_mu,
            self.model.Hyperprior.synthesis_std,
            self.model.Generator,
        ]
        return sum(params_m(m) for m in modules if m is not None)

    @torch.no_grad()
    def encode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, dtype=torch.float)

        return time_ms(
            lambda: self.model.compress(x, silent=True),
            self.device,
            warmup=warmup,
            repeat=repeat,
        )
    
    @torch.no_grad()
    def decode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, dtype=torch.float)

        compression_output = self.model.compress(x, silent=True)

        return time_ms(
            lambda: self.model.decompress(compression_output),
            self.device,
            warmup=warmup,
            repeat=repeat,
        )
    
    @torch.no_grad()
    def encode_gflops(self, x):
        from fvcore.nn import FlopCountAnalysis

        x = x.to(self.device, dtype=torch.float)

        encoder_flops = FlopCountAnalysis(
            self.model.Encoder,
            x,
        ).total()

        y = self.model.Encoder(x)

        n_down = self.model.Hyperprior.analysis_net.n_downsampling_layers
        factor = 2 ** n_down
        y = utils.pad_factor(y, y.size()[2:], factor)

        analysis_flops = FlopCountAnalysis(
            self.model.Hyperprior.analysis_net,
            y,
        ).total()

        return (encoder_flops + analysis_flops) / 1e9
    
    @torch.no_grad()
    def decode_gflops(self, x):
        from fvcore.nn import FlopCountAnalysis

        x = x.to(self.device, dtype=torch.float)

        y = self.model.Encoder(x)

        n_down = self.model.Hyperprior.analysis_net.n_downsampling_layers
        factor = 2 ** n_down
        y = utils.pad_factor(y, y.size()[2:], factor)

        z = self.model.Hyperprior.analysis_net(y)
        z_hat = self.model.Hyperprior._quantize(z, mode="quantize")

        mu_flops = FlopCountAnalysis(
            self.model.Hyperprior.synthesis_mu,
            z_hat,
        ).total()

        std_flops = FlopCountAnalysis(
            self.model.Hyperprior.synthesis_std,
            z_hat,
        ).total()

        latent_means = self.model.Hyperprior.synthesis_mu(z_hat)

        latent_decoded = self.model.Hyperprior.quantize_latents_st(
            y,
            latent_means,
        )

        generator_flops = FlopCountAnalysis(
            self.model.Generator,
            latent_decoded,
        ).total()

        return (mu_flops + std_flops + generator_flops) / 1e9