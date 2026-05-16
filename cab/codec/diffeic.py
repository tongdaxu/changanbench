import numpy as np
from cab.codec.abs import ImageCodecIface
from cab.models.diffeic.model.diffeic import DiffEIC
from cab.models.diffeic.utils.common import instantiate_from_config, load_state_dict
from omegaconf import OmegaConf
import torch
from cab.models.diffeic.model.spaced_sampler import SpacedSampler
from cab.models.diffeic.model.ddim_sampler import DDIMSampler
from typing import List, Tuple

@torch.no_grad()
def process(
    model: DiffEIC,
    imgs: torch.Tensor, 
    sampler: str,
    steps: int
) -> Tuple[List[np.ndarray], float]:
    """
    Apply DiffEIC model on a batch of images.

    Args:
        model (DiffEIC): Model.
        imgs (np.ndarray): Batch of images, shape [B, H, W, C], RGB, range [0, 1]
        sampler (str): Sampler name.
        steps (int): Sampling steps.

    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 1]).
        bpp
    """
    n_samples = imgs.shape[0]
    if sampler == "ddpm":
        sampler_obj = SpacedSampler(model, var_type="fixed_small")
    else:
        sampler_obj = DDIMSampler(model)
    control = imgs.float().clamp(0, 1).to(model.device)

    height, width = control.size(-2), control.size(-1)
    strings, shape, bpp = model.apply_condition_compress(control, height, width)
    cond = {
        "c_latent": [model.apply_condition_decompress(strings, shape)],
        "c_crossattn": [model.get_learned_conditioning([""] * n_samples)]
    }

    shape = (n_samples, 4, height // 8, width // 8)
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
    if isinstance(sampler_obj, SpacedSampler):
        samples = sampler_obj.sample(
            steps, shape, cond,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=None,
            cond_fn=None, x_T=x_T
        )
    else:
        samples, _ = sampler_obj.sample(
            S=steps, batch_size=shape[0], shape=shape[1:],
            conditioning=cond, unconditional_conditioning=None,
            x_T=x_T, eta=0
        )

    x_samples = model.decode_first_stage(samples)
    rec = ((x_samples + 1) / 2).clamp(0, 1)
    return rec, bpp

class DiffEICImageCodec(ImageCodecIface):
    def __init__(self, quality, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality = quality
        self.config = OmegaConf.load("config/config.yaml")
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
        
        self.ckpt_sd = _cfg_get(self.config, 'ckpt_sd', kwargs.get('ckpt_sd'))
        self.ckpt_lc = _cfg_get(self.config, 'ckpt_lc', kwargs.get('ckpt_lc'))
        
        model: DiffEIC = instantiate_from_config(OmegaConf.load("cab/models/diffeic/diffeic.yaml"))
        ckpt_sd = torch.load(self.ckpt_sd, map_location="cpu")['state_dict']
        ckpt_lc = torch.load(self.ckpt_lc, map_location="cpu")['state_dict']
        ckpt_sd.update(ckpt_lc)
        load_state_dict(model, ckpt_sd, strict=False)
        # update preprocess model
        model.preprocess_model.update(force=True)
        model.freeze()
        self.model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()

        self.steps = _cfg_get(self.config, 'steps', kwargs.get('steps'))
        self.sampler = _cfg_get(self.config, 'sampler', kwargs.get('sampler'))


    @torch.no_grad()
    def forward(self, x, *args, **kwargs):
        xhat, bpp = process(
            self.model, x, steps=self.steps, sampler=self.sampler
        )

        return xhat, bpp