import torch
import os
import numpy as np
from cab.codec.abs import ImageCodecIface
from cab.models.tcm.models import TCM

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