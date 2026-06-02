import torch
import os
import numpy as np
from cab.codec.abs import ImageCodecIface
import pickle

def pickle_size_of(obj):
    return len(pickle.dumps(obj))

class MSILLMImageCodec(ImageCodecIface):
    def __init__(self, ckpt_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ckpt_name = ckpt_name
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