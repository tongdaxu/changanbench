import torch
import os
import numpy as np
from cab.codec.abs import ImageCodecIface
from cab.models.MLICPlusPlus.config.args import test_options
from cab.models.MLICPlusPlus.config.config import model_config
from cab.models.MLICPlusPlus.models import *
from cab.models.MLICPlusPlus.utils.testing import compress_one_image, decompress_one_image

class MLICPlusPlusImageCodec(ImageCodecIface):
    def __init__(self, quality, ckpt_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality = quality
        self.ckpt_path = ckpt_path
        config = model_config()
        
        self.model = MLICPlusPlus(config=config).cuda().eval()
        checkpoint = torch.load(self.ckpt_path, map_location='cpu')
        # support checkpoints saved with DataParallel/DistributedDataParallel ("module." prefix)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            raw_state = checkpoint['state_dict']
        else:
            raw_state = checkpoint
        fixed_state = {}
        for k, v in raw_state.items():
            new_k = k
            if k.startswith("module."):
                new_k = k[len("module."):]
            fixed_state[new_k] = v
        self.model.load_state_dict(fixed_state)
        epoch = checkpoint["epoch"]
        self.save_dir = os.path.join('./experiments', 'codestream', '%02d' % (epoch + 1))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):
        x = x.cuda()
        recon_list = []
        bpp_vals = []
        for i in range(x.shape[0]):
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
            img = x[i:i+1]
            bpp, enc_time = compress_one_image(model=self.model, x=img, stream_path=self.save_dir, H=img.shape[2], W=img.shape[3], img_name=f"{i}_r{rank}")
            xhat, dec_time = decompress_one_image(model=self.model, stream_path=self.save_dir, img_name=f"{i}_r{rank}")
            xhat = xhat.clamp(0, 1)
            if xhat.dim() == 3:  
                xhat = xhat.unsqueeze(0)

            # ensure reconstruction on same device as input batch
            out_img = xhat.to(x.device)
            recon_list.append(out_img)
            bpp_vals.append(bpp)

        # concat per-image reconstructions into a full-batch tensor
        xhat = torch.cat(recon_list, dim=0)  # shape: (batch, C, H, W)
        bpp = torch.tensor(bpp_vals, dtype=torch.float32, device=x.device)  # shape: (batch,)
        return xhat, bpp