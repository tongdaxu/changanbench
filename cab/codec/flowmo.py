import torch
import os
from cab.codec.abs import ImageCodecIface
from cab.models.flowmo import train_utils
from omegaconf import OmegaConf
from cab.complexity import params_m, time_ms, gflops
from cab.models.flowmo.models import prepare_idxs

class FlowMoEncodeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        code, _ = self.model.encode(x)
        return code


class FlowMoDecoderSingleStepWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z, code, timesteps):
        pred, _ = self.model.decode(z, code, timesteps)
        return pred

class FlowMoImageTokenizer(ImageCodecIface):
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
    
    def fake_input(self, image_size=256, batch_size=1, device=None):
        if device is None:
            device = self.device
        return torch.rand(batch_size, 3, image_size, image_size, device=device) * 2 - 1

    def encode_params_m(self):
        modules = [self.model.encoder]

        if hasattr(self.model, "quantizer"):
            modules.append(self.model.quantizer)

        return sum(params_m(m) for m in modules)

    def decode_params_m(self):
        return params_m(self.model.decoder)

    @torch.no_grad()
    def _encode_and_quantize_code(self, x):
        x = x.to(self.device)

        code, _ = self.model.encode(x)
        code, _, _ = self.model._quantize(code)

        mask = torch.ones_like(code[..., :1])
        code = torch.cat([code, mask], dim=-1)

        return code

    @torch.no_grad()
    def encode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device)

        def fn():
            code, _ = self.model.encode(x)
            code, _, _ = self.model._quantize(code)
            return code

        return time_ms(
            fn,
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

    @torch.no_grad()
    def decode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device)
        code = self._encode_and_quantize_code(x)

        bs, _, h, w = x.shape
        steps = self.config.eval.sampling.sample_steps
        cfg = self.config.eval.sampling.cfg
        use_cfg = cfg != 1.0

        def fn():
            z = torch.randn((bs, 3, h, w), device=self.device, dtype=x.dtype)

            ts = torch.linspace(
                1.0,
                0.0,
                steps + 1,
                device=self.device,
                dtype=x.dtype,
            )

            for i in range(steps):
                t = torch.full(
                    (bs,),
                    ts[i],
                    device=self.device,
                    dtype=x.dtype,
                )
                dt = ts[i] - ts[i + 1]

                v_cond, _ = self.model.decode(z, code, t)

                if use_cfg:
                    null_code = code * 0.0
                    v_uncond, _ = self.model.decode(z, null_code, t)
                    v = v_uncond + cfg * (v_cond - v_uncond)
                else:
                    v = v_cond

                z = z - dt * v

            return z

        return time_ms(
            fn,
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

    def encode_gflops(self, x):
        x = x.to(self.device).float()

        img_idxs, txt_idxs = prepare_idxs(
            x,
            self.model.code_length,
            self.model.patch_size,
        )

        txt = torch.zeros(
            (x.shape[0], self.model.code_length, self.model.encoder_context_dim),
            device=x.device,
            dtype=torch.float32,
        )

        return gflops(
            self.model.encoder.float(),
            (x.float(), img_idxs, txt, txt_idxs, None),
        )

    @torch.no_grad()
    def decode_gflops(self, x):
        x = x.to(self.device)

        code = self._encode_and_quantize_code(x).float()

        bs, _, h, w = x.shape
        z = torch.randn((bs, 3, h, w), device=self.device, dtype=torch.float32)
        timesteps = torch.ones((bs,), device=self.device, dtype=torch.float32) * 0.5

        img_idxs, txt_idxs = prepare_idxs(
            z,
            self.model.code_length,
            self.model.patch_size,
        )

        single_call_gflops = gflops(
            self.model.decoder.float(),
            (z, img_idxs, code, txt_idxs, timesteps),
        )

        if single_call_gflops is None:
            return None

        steps = self.config.eval.sampling.sample_steps
        cfg = self.config.eval.sampling.cfg
        calls_per_step = 2 if cfg != 1.0 else 1

        return single_call_gflops * steps * calls_per_step
