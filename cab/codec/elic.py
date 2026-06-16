from cab.models.hific_src import model
import torch
import os
import numpy as np
from cab.codec.abs import ImageCodecIface
import compressai
from compressai.zoo import load_state_dict
from cab.models.ELIC.Network import TestModel
from cab.utils.complexity import params_m, time_ms, gflops

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