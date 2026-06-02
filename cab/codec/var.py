import torch
import os
import numpy as np
from cab.codec.abs import ImageCodecIface
from cab.models.var.models import build_vae_var
import os.path as osp


class VARImageTokenizer(ImageCodecIface):
    def __init__(self, quality, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality = quality
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MODEL_DEPTH = 16 #16, 20, 24, 30
        hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
        vae_ckpt, var_ckpt = '/NEW_EDS/JJ_Group/lisq/VAR/vae_ch160v4096z32.pth', f'/NEW_EDS/JJ_Group/lisq/VAR/var_d{MODEL_DEPTH}.pth'
        if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
        if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')
        
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

