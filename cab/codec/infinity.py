import torch
import os
import numpy as np
from cab.codec.abs import ImageCodecIface 
from cab.models.infinity.models.bsq_vae.vae import vae_model

class InfinityImageTokenizer(ImageCodecIface):
    def __init__(self, vqgan_ckpt, codebook_dim, downsample_factor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.codebook_dim = codebook_dim
        self.downsample_factor = downsample_factor
        codebook_size = 2 ** self.codebook_dim
        schedule_mode = "dynamic"
        vae = vae_model(vqgan_ckpt, schedule_mode, self.codebook_dim, codebook_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = vae.to(self.device)
        vae.eval()
        

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):

        x = x.to(self.device, non_blocking=True)
        xhat, _ = self.vae(x)

        bits_per_token = self.codebook_dim
        num_tokens = (x.size(2) // self.downsample_factor) * (x.size(3) // self.downsample_factor)
        total_bits = num_tokens * bits_per_token
        bpp = total_bits / (x.size(2) * x.size(3))
        bpp = torch.tensor([bpp], dtype=torch.float32, device=x.device)
        return xhat, bpp