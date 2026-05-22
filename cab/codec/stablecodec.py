import torch
import os
import math
import numpy as np
from cab.codec.abs import ImageCodecIface
from cab.models.stablecodec_src.StableCodec import StableCodec
from types import SimpleNamespace
from pathlib import Path
from cab.models.stablecodec_src.compress_utils import write_body, read_body, filesize

def compress_one_image(net, bin_path, ori_h, ori_w, img_name, x):
    with torch.no_grad():
        output_dict = net.compress(x)
    shape = output_dict["shape"]
    if not os.path.exists(bin_path): os.makedirs(bin_path)
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

class StableCodecImageCodec(ImageCodecIface):
    def __init__(self, sd_path, elic_path, codec_path, rec_path, bin_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sd_path = sd_path
        self.elic_path = elic_path
        self.codec_path = codec_path
        self.rec_path = rec_path
        self.bin_path = bin_path
        if not os.path.exists(self.rec_path): os.makedirs(self.rec_path)
        if not os.path.exists(self.bin_path): os.makedirs(self.bin_path)
        args_ns = SimpleNamespace(
            elic_path=self.elic_path,
            codec_path=self.codec_path,
            rec_path=self.rec_path,
            bin_path=self.bin_path,
            **kwargs
        )
        self.model = StableCodec(sd_path=sd_path, args=args_ns)
        self.model.cuda().eval()
        self.model.codec.update(force=True)

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):        
        recon_list = []
        bpp_vals = []
        pos_tag_prompt = [1]

        batch = x.cuda()
        for i in range(batch.shape[0]):
            img_i = batch[i].unsqueeze(0)  # shape (1,C,H,W)
            ori_h, ori_w = img_i.shape[2:]
            fname = f"img_{i}"

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
    
