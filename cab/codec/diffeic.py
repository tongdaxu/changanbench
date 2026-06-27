import numpy as np
from cab.codec.abs import ImageCodecIface
from cab.models.diffeic.model.diffeic import DiffEIC
from cab.models.diffeic.utils.common import instantiate_from_config, load_state_dict
from omegaconf import OmegaConf
import torch
from cab.models.diffeic.model.spaced_sampler import SpacedSampler
from cab.models.diffeic.model.ddim_sampler import DDIMSampler
from typing import List, Tuple
from contextlib import contextmanager

from cab.complexity import params_m, time_ms

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False

try:
    from deepspeed.profiling.flops_profiler import get_model_profile
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

@contextmanager
def _disable_checkpoint_and_xformers():
    orig_checkpoint = None
    util_mod = None
    orig_xops = {}
    xops_mod = None

    try:
        import cab.models.diffeic.ldm.modules.diffusionmodules.util as util_mod
        orig_checkpoint = getattr(util_mod, "checkpoint", None)
        if orig_checkpoint is not None:
            util_mod.checkpoint = lambda func, *args, **kwargs: func(*args)
    except Exception:
        orig_checkpoint = None

    try:
        import xformers.ops as xops_mod
        for name in dir(xops_mod):
            if "efficient_attention" in name or "memory_efficient_attention" in name:
                fn = getattr(xops_mod, name)
                if callable(fn):
                    orig_xops[name] = fn

                    def _make_wrapper():
                        def _wrap(q, k, v, *a, **kw):
                            try:
                                return torch.nn.functional.scaled_dot_product_attention(
                                    q, k, v,
                                    attn_mask=None,
                                    dropout_p=0.0,
                                    is_causal=False,
                                )
                            except Exception:
                                dk = q.size(-1)
                                attn = torch.matmul(q, k.transpose(-2, -1)) / (dk ** 0.5)
                                attn = torch.softmax(attn, dim=-1)
                                return torch.matmul(attn, v)
                        return _wrap

                    setattr(xops_mod, name, _make_wrapper())
    except Exception:
        orig_xops = {}

    try:
        yield
    finally:
        try:
            if orig_checkpoint is not None and util_mod is not None:
                util_mod.checkpoint = orig_checkpoint
        except Exception:
            pass

        try:
            if orig_xops and xops_mod is not None:
                for name, fn in orig_xops.items():
                    setattr(xops_mod, name, fn)
        except Exception:
            pass


def safe_flop_analysis(module, *inputs):
    """
    返回 raw FLOPs，不是 GFLOPs。
    与 eval_profiling 保持一致，最后在 encode_gflops/decode_gflops 中 /1e9。
    """
    if not FVCORE_AVAILABLE:
        return None

    try:
        with _disable_checkpoint_and_xformers():
            inp = inputs[0] if len(inputs) == 1 else inputs
            analysis = FlopCountAnalysis(module, inp)
            return analysis.total()
    except Exception as e:
        print(f"[DiffEIC FLOPs failed] {module.__class__.__name__}: {e}")
        return None

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
    
    def fake_input(self, image_size=256, batch_size=1, device=None):
        if device is None:
            device = self.model.device
        return torch.rand(batch_size, 3, image_size, image_size, device=device)

    def encode_params_m(self):
        modules = []

        if hasattr(self.model, "first_stage_model") and hasattr(self.model.first_stage_model, "encoder"):
            modules.append(self.model.first_stage_model.encoder)

        if hasattr(self.model, "preprocess_model"):
            modules.append(self.model.preprocess_model)

        return sum(params_m(m) for m in modules if m is not None)

    def decode_params_m(self):
        modules = []

        if hasattr(self.model, "control_model"):
            modules.append(self.model.control_model)

        if hasattr(self.model, "model") and hasattr(self.model.model, "diffusion_model"):
            modules.append(self.model.model.diffusion_model)

        if hasattr(self.model, "first_stage_model") and hasattr(self.model.first_stage_model, "decoder"):
            modules.append(self.model.first_stage_model.decoder)

        return sum(params_m(m) for m in modules if m is not None)
    
    @torch.no_grad()
    def encode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.model.device, dtype=torch.float).clamp(0, 1)
        height, width = x.shape[-2], x.shape[-1]

        return time_ms(
            lambda: self.model.apply_condition_compress(x, height, width),
            self.model.device,
            warmup=warmup,
            repeat=repeat,
        )

    @torch.no_grad()
    def decode_time_ms(self, x, warmup=1, repeat=3):
        x = x.to(self.model.device, dtype=torch.float).clamp(0, 1)

        height, width = x.shape[-2], x.shape[-1]
        n_samples = x.shape[0]

        strings, shape, _ = self.model.apply_condition_compress(x, height, width)

        if self.sampler == "ddpm":
            sampler_obj = SpacedSampler(self.model, var_type="fixed_small")
        else:
            sampler_obj = DDIMSampler(self.model)

        def fn():
            cond_latent = self.model.apply_condition_decompress(strings, shape)
            cond_crossattn = self.model.get_learned_conditioning([""] * n_samples)

            cond = {
                "c_latent": [cond_latent],
                "c_crossattn": [cond_crossattn],
            }

            latent_shape = (n_samples, 4, height // 8, width // 8)

            x_T = torch.randn(
                latent_shape,
                device=self.model.device,
                dtype=torch.float32,
            )

            if isinstance(sampler_obj, SpacedSampler):
                samples = sampler_obj.sample(
                    self.steps,
                    latent_shape,
                    cond,
                    unconditional_guidance_scale=1.0,
                    unconditional_conditioning=None,
                    cond_fn=None,
                    x_T=x_T,
                )
            else:
                samples, _ = sampler_obj.sample(
                    S=self.steps,
                    batch_size=latent_shape[0],
                    shape=latent_shape[1:],
                    conditioning=cond,
                    unconditional_conditioning=None,
                    x_T=x_T,
                    eta=0,
                )

            x_samples = self.model.decode_first_stage(samples)
            return ((x_samples + 1) / 2).clamp(0, 1)

        return time_ms(
            fn,
            self.model.device,
            warmup=warmup,
            repeat=repeat,
        )
    
    @torch.no_grad()
    def encode_gflops(self, x):
        x = x.to(self.model.device, dtype=torch.float).clamp(0, 1)

        encode_flops = None
        vae_encoder_flops = None
        preprocess_flops = None

        # 1. VAE encoder FLOPs
        try:
            control_normalized = x * 2 - 1

            vae_encoder_flops = safe_flop_analysis(
                self.model.first_stage_model.encoder,
                control_normalized,
            )

            if vae_encoder_flops is not None:
                encode_flops = vae_encoder_flops

        except Exception as e:
            print(f"[DiffEIC] VAE encoder FLOPs failed: {e}")

        # 2. preprocess_model FLOPs
        try:
            ref = self.model.encode_first_stage(x * 2 - 1).mode()

            if hasattr(self.model, "scale_factor"):
                ref = ref * self.model.scale_factor

            class PreprocessWrapper(torch.nn.Module):
                def __init__(self, preprocess_model):
                    super().__init__()
                    self.preprocess_model = preprocess_model

                def forward(self, control, ref):
                    return self.preprocess_model(control, ref)

            preprocess_wrapper = PreprocessWrapper(
                self.model.preprocess_model
            ).to(self.model.device).eval()

            preprocess_flops = safe_flop_analysis(
                preprocess_wrapper,
                (x, ref),
            )

            if preprocess_flops is not None:
                if encode_flops is not None:
                    encode_flops += preprocess_flops
                else:
                    encode_flops = preprocess_flops

        except Exception as e:
            print(f"[DiffEIC] preprocess_model FLOPs failed: {e}")

        print(
            "[DiffEIC Encode FLOPs] "
            f"vae_encoder={None if vae_encoder_flops is None else vae_encoder_flops / 1e9:.3f}G, "
            f"preprocess={None if preprocess_flops is None else preprocess_flops / 1e9:.3f}G"
        )

        if encode_flops is None:
            return None

        return encode_flops / 1e9
    
    @torch.no_grad()
    def decode_gflops(self, x):
        x = x.to(self.model.device, dtype=torch.float).clamp(0, 1)

        n_samples, _, height, width = x.shape
        latent_shape = (n_samples, 4, height // 8, width // 8)

        decode_flops = None
        unet_total_flops = None
        vae_decoder_flops = None

        # 1. Control + UNet single-step FLOPs × steps
        try:
            x_dummy = torch.randn(
                latent_shape,
                device=self.model.device,
                dtype=torch.float32,
            )

            t_dummy = torch.tensor(
                [0],
                device=self.model.device,
                dtype=torch.long,
            )

            strings, shape, _ = self.model.apply_condition_compress(
                x,
                height,
                width,
            )

            cond_latent = self.model.apply_condition_decompress(strings, shape)
            cond_crossattn = self.model.get_learned_conditioning([""] * n_samples)

            class UNetWrapper(torch.nn.Module):
                def __init__(self, model_ref, cond_latent, cond_crossattn):
                    super().__init__()
                    self.model_ref = model_ref
                    self.cond_latent = cond_latent
                    self.cond_crossattn = cond_crossattn

                def forward(self, x_t, t):
                    cond = {
                        "c_latent": [self.cond_latent],
                        "c_crossattn": [self.cond_crossattn],
                    }
                    return self.model_ref.apply_model(x_t, t, cond)

            unet_wrapper = UNetWrapper(
                self.model,
                cond_latent,
                cond_crossattn,
            ).to(self.model.device).eval()

            unet_single_step_flops = None

            if DEEPSPEED_AVAILABLE:
                try:
                    flops, macs, params = get_model_profile(
                        model=unet_wrapper,
                        args=(x_dummy, t_dummy),
                        print_profile=False,
                        detailed=False,
                        warm_up=3,
                        as_string=False,
                        output_file=None,
                        ignore_modules=None,
                    )
                    unet_single_step_flops = flops
                except Exception as e:
                    print(f"[DiffEIC] DeepSpeed UNet FLOPs failed: {e}")

            if unet_single_step_flops is None:
                unet_single_step_flops = safe_flop_analysis(
                    unet_wrapper,
                    (x_dummy, t_dummy),
                )

            if unet_single_step_flops is not None:
                unet_total_flops = unet_single_step_flops * int(self.steps)
                decode_flops = unet_total_flops

        except Exception as e:
            print(f"[DiffEIC] UNet+Control FLOPs failed: {e}")

        # 2. VAE decoder FLOPs
        try:
            latent_dummy = torch.randn(
                latent_shape,
                device=self.model.device,
                dtype=torch.float32,
            )

            vae_decoder_flops = safe_flop_analysis(
                self.model.first_stage_model.decoder,
                latent_dummy,
            )

            if vae_decoder_flops is not None:
                if decode_flops is not None:
                    decode_flops += vae_decoder_flops
                else:
                    decode_flops = vae_decoder_flops

        except Exception as e:
            print(f"[DiffEIC] VAE decoder FLOPs failed: {e}")

        print(
            "[DiffEIC Decode FLOPs] "
            f"unet_total={None if unet_total_flops is None else unet_total_flops / 1e9:.3f}G, "
            f"vae_decoder={None if vae_decoder_flops is None else vae_decoder_flops / 1e9:.3f}G"
        )

        if decode_flops is None:
            return None

        return decode_flops / 1e9