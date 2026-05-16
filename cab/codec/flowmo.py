import torch
import os
from cab.codec.abs import ImageCodecIface
from cab.models.flowmo import train_utils
from omegaconf import OmegaConf

class FlowMoImageCodec(ImageCodecIface):
    def __init__(self, quality, ckpt_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality = quality
        self.ckpt_path = ckpt_path
        self.config = OmegaConf.load("cab/models/flowmo/configs/base.yaml")
        self.config = train_utils.restore_config(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = self.ckpt_path
        
        # build model from config and load weights
        self.model = train_utils.build_model(self.config).to(self.device)
        state_dict = train_utils.load_state_dict(ckpt)[self.config.eval.state_dict_key]
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # read codec-specific params (supports dict-style or attr-style config)
        def _cfg_get(cfg, key, default=None):
            if isinstance(cfg, dict):
                if key in cfg:
                    return cfg[key]
                if 'params' in cfg and isinstance(cfg['params'], dict):
                    return cfg['params'].get(key, default)
                return default
            # attr-style
            if hasattr(cfg, key):
                return getattr(cfg, key)
            if hasattr(cfg, 'params') and isinstance(getattr(cfg, 'params'), dict):
                return cfg.params.get(key, default)
            return default

        self.codebook_size_for_entropy = _cfg_get(self.config, 'codebook_size_for_entropy', kwargs.get('codebook_size_for_entropy'))
        self.code_length = _cfg_get(self.config, 'code_length', kwargs.get('code_length'))
        if self.codebook_size_for_entropy is None or self.code_length is None:
            raise ValueError("FlowMoImageCodec requires 'codebook_size_for_entropy' and 'code_length' in config params")


    @torch.no_grad()
    def forward(self, x, *args, **kwargs):
        x = x.to(self.device)
        dtype_to_use = torch.bfloat16 if train_utils.bfloat16_is_available() else torch.float32
        xhat = self.model.reconstruct(x, dtype=dtype_to_use)
        bpp =self.codebook_size_for_entropy * self.code_length / (x.shape[2] * x.shape[3])
        return xhat, bpp
