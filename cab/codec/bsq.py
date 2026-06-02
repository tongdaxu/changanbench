import torch
import os
import numpy as np
from cab.codec.abs import ImageCodecIface
from omegaconf import OmegaConf
from cab.models.bsq import config as config_utils
import torch.cuda.amp as amp

class BSQImageTokenizer(ImageCodecIface):
    def __init__(self, quality, ckpt_path, config_path, bits_per_token, downsample_factor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality = quality
        self.ckpt_path = ckpt_path
        checkpoint = torch.load(ckpt_path, map_location="cpu")
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
        self.bits_per_token = bits_per_token
        self.downsample_factor = downsample_factor
        self.bsq_config = OmegaConf.load(config_path)
        self.bsq_config.model.params.clamp_range = (-1, 1) if self.bsq_config.data.zero_mean else (0, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = config_utils.instantiate_from_config(self.bsq_config.model).to(self.device)
        self.model.load_state_dict(fixed_state)
        assert not self.bsq_config.data.zero_mean or not self.bsq_config.model.params.get('logit_laplace', False), "logit laplace mode is only compatible with the input being [0, 1]"
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):
        with amp.autocast(enabled=not self.bsq_config.optimizer.disable_amp, dtype=torch.bfloat16 if self.bsq_config.optimizer.use_bf16 else torch.float16):
                        xhat, _, info = self.model(x)
        xhat = xhat.to(torch.float32)
        # Calculate bpp based on the number of tokens and image size
        bits_per_token = self.bits_per_token
        num_tokens = (x.size(2) // self.downsample_factor) * (x.size(3) // self.downsample_factor)
        bpp = (num_tokens * bits_per_token) / (x.size(2) * x.size(3))
        
        return xhat, torch.tensor([bpp], dtype=torch.float32, device=x.device)



