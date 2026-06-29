import numpy as np
import torch
import os, sys
# ensure perco custom modules are importable by name used in pipeline configs
# _perco_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "perco_src"))
# if _perco_src not in sys.path:
#     sys.path.insert(0, _perco_src)
from cab.codec.abs import ImageCodecIface
from cab.models.perco_src.config import ConfigPerco as cfg_perco
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from cab.models.perco_src.pipeline_sd_perco import StableDiffusionPipelinePerco
import zlib
import torchac
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
import contextlib
import io

from cab.complexity import params_m, time_ms, gflops

@contextlib.contextmanager
def suppress_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def safe_params_m(modules):
    total = 0.0
    for m in modules:
        if m is not None:
            total += params_m(m)
    return total

def compute_cdf_uniform_prob(codebook_size, target_shape):
    """Obtain CDF from uniform distribution, cast to target_shape"""
    b, h, w = target_shape
    prob_per_entry = 1.0 / codebook_size

    # Compute the cumulative sum starting from 0
    cdf = torch.cumsum(torch.full((codebook_size,), prob_per_entry), dim=0)
    cdf = torch.cat([torch.zeros(1), cdf])
    cdf = cdf.view(1, 1, 1, -1).expand(b, h, w, -1)
    cdf = cdf.clone()
    cdf[..., -1] = 1.0
    return cdf

def compress_hyper_latent(z_hat_indices):
    """Compress hyper-latent to bytes using torchac."""
    _, cfg_cs = cfg_perco.rate_cfg[cfg_perco.target_rate]
    cdf = compute_cdf_uniform_prob(cfg_cs, z_hat_indices.shape)
    z_hat_indices = z_hat_indices.to(torch.int16).to('cpu')
    return torchac.encode_float_cdf(cdf, z_hat_indices, check_input_bounds=True)

def compress_text(input_text):
    """Compress the input text to bytes using zlib."""
    input_bytes = input_text.encode('utf-8')
    return zlib.compress(input_bytes, level=zlib.Z_BEST_COMPRESSION)

def calculate_bpp(compressed_data, num_pixels, bytes=True, num_bytes=None):
    """Calculate bpp given the compressed text and number of pixels."""
    scaling_factor = 8 if bytes else 1
    if num_bytes:
        return num_bytes * scaling_factor / num_pixels
    return len(compressed_data) * scaling_factor / num_pixels

class PerCoEncodeWrapper(torch.nn.Module):
    def __init__(self, pipe):
        super().__init__()
        self.vae = pipe.vae
        self.hyper_enc = pipe.hyper_enc

    def forward(self, x):
        latents = self.vae.encode((x * 2.0 - 1.0)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        hyper_latent = self.hyper_enc(latents)
        return hyper_latent.z_hat


class PerCoTextEncoderWrapper(torch.nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        return self.text_encoder(input_ids)[0]


class PerCoUNetWrapper(torch.nn.Module):
    def __init__(self, pipe, prompt_embeds, local_features, do_cfg=True):
        super().__init__()
        self.pipe = pipe
        self.unet = pipe.unet
        self.prompt_embeds = prompt_embeds
        self.local_features = local_features
        self.do_cfg = do_cfg

    def forward(self, latents, timestep):
        if self.do_cfg:
            latents = torch.cat([latents] * 2)
            local_features = torch.cat([self.local_features] * 2)
        else:
            local_features = self.local_features

        latents = self.pipe.scheduler.scale_model_input(latents, timestep)

        return self.unet(
            latents,
            timestep,
            encoder_hidden_states=self.prompt_embeds,
            local_features=local_features,
            return_dict=False,
        )[0]


class PerCoVAEDecodeWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        return self.vae.decode(
            latents / self.vae.config.scaling_factor,
            return_dict=False,
        )[0]

class PerCoImageCodec(ImageCodecIface):
    def __init__(self, ckpt_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ckpt_path = ckpt_path
        self.cfg_perco = cfg_perco
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained(cfg_perco.blip_model)
        self.blip2 = Blip2ForConditionalGeneration.from_pretrained(cfg_perco.blip_model).to(self.device)
        self.pipe = StableDiffusionPipelinePerco.from_pretrained(
            self.ckpt_path , safety_checker=None, requires_safety_checker=False
        ).to(self.device)

        self.blip2.eval()
        self.pipe.vae.eval()
        self.pipe.hyper_enc.eval()
        self.pipe.hyper_enc.quantizer.eval()
        self.pipe.unet.eval()
        self.pipe.text_encoder.eval()

        try:
            self.pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):
        # x: torch.Tensor, shape [B, C, H, W], range [0, 1]
        B, C, H, W = x.shape
        x_hat_list = []
        bpp_text_list = []
        bpp_hyper_latent_list = []
        bpp_total_list = []

        self.pipe = self.pipe.to(self.device)
        self.pipe.vae = self.pipe.vae.to(self.device)
        self.pipe.hyper_enc = self.pipe.hyper_enc.to(self.device)
        self.pipe.hyper_enc.quantizer = self.pipe.hyper_enc.quantizer.to(self.device)
        self.blip2 = self.blip2.to(self.device)
        # self.blip2.eval()

        for i in range(B):
            img = x[i].unsqueeze(0).to(self.device)  # [1, C, H, W]
         
            # 1. Get image caption (BLIP 2)
            inputs = self.processor(images=img[0] * 255, return_tensors="pt").to(self.device)

            generated_ids = self.blip2.generate(**inputs, max_length=self.cfg_perco.max_number_tokens)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            # 2. Compress caption (zlib)
            byte_stream_text = compress_text(generated_text)
            bpp_text = calculate_bpp(byte_stream_text, H * W)

            # 3. Run VAE encoder
            latents = self.pipe.vae.to(self.device).encode((img * 2. - 1.)).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor

            # 4. Run hyper-encoder
            self.pipe.hyper_enc.quantizer.eval()
            hyper_latent = self.pipe.hyper_enc(latents)
            z_hat, z_hat_indices = hyper_latent.z_hat, hyper_latent.indices

            # 5. Compress hyper-latent (AC)
            byte_stream_hyper_latent = compress_hyper_latent(z_hat_indices)
            bpp_hyper_latent = calculate_bpp(byte_stream_hyper_latent, H * W)
            bpp_total = bpp_text + bpp_hyper_latent

            # 6. Generate reconstruction
            generator = None
            x_hat_img = self.pipe(
                generated_text,
                z_hat,
                height=H,
                width=W,
                num_inference_steps=self.cfg_perco.num_inference_steps,
                guidance_scale=self.cfg_perco.guidance_scale,
                generator=generator
            ).images[0]
            x_hat_img = ToTensor()(x_hat_img)
            x_hat_list.append(x_hat_img.unsqueeze(0).to(self.device))
            bpp_text_list.append(bpp_text)
            bpp_hyper_latent_list.append(bpp_hyper_latent)
            bpp_total_list.append(bpp_total)

        # concat per-image reconstructions into a full-batch tensor
        xhat = torch.cat(x_hat_list, dim=0)  # shape: (batch, C, H, W)
        bpp = torch.tensor(bpp_total_list, dtype=torch.float32, device=self.device)  # shape: (batch,)
        return xhat, bpp
    
    def fake_input(self, image_size=256, batch_size=1, device=None):
        if device is None:
            device = self.device
        return torch.rand(batch_size, 3, image_size, image_size, device=device)

    def encode_params_m(self):
        # 推荐口径：不计 BLIP2 captioner
        modules = [
            self.pipe.vae.encoder,
            getattr(self.pipe.vae, "quant_conv", None),
            self.pipe.hyper_enc,
        ]
        return safe_params_m(modules)

    def decode_params_m(self):
        modules = [
            self.pipe.text_encoder,
            self.pipe.unet,
            getattr(self.pipe.vae, "post_quant_conv", None),
            self.pipe.vae.decoder,
        ]
        return safe_params_m(modules)

    @torch.no_grad()
    def _encode_for_complexity(self, x):
        """
        只用于复杂度测试：
        - 不调用 BLIP2
        - 用固定 prompt 作为解码端文本条件
        - 只生成 z_hat
        """
        x = x[:1].to(self.device, dtype=torch.float).clamp(0, 1)
        _, _, H, W = x.shape

        generated_text = ""

        latents = self.pipe.vae.encode((x * 2.0 - 1.0)).latent_dist.sample()
        latents = latents * self.pipe.vae.config.scaling_factor

        hyper_latent = self.pipe.hyper_enc(latents)
        z_hat = hyper_latent.z_hat

        return generated_text, z_hat, H, W

    @torch.no_grad()
    def encode_time_ms(self, x, warmup=2, repeat=5):
        """
        推荐口径：只测 VAE encoder + hyper encoder。
        不测 BLIP2 caption generation。
        """
        x = x[:1].to(self.device, dtype=torch.float).clamp(0, 1)

        def fn():
            latents = self.pipe.vae.encode((x * 2.0 - 1.0)).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor
            hyper_latent = self.pipe.hyper_enc(latents)
            z_hat_indices = hyper_latent.indices
            _ = compress_hyper_latent(z_hat_indices)
            return hyper_latent.z_hat

        with suppress_output():
            return time_ms(
                fn,
                self.device,
                warmup=warmup,
                repeat=repeat,
            )

    @torch.no_grad()
    def decode_time_ms(self, x, warmup=1, repeat=3):
        """
        Decode = text encoder + diffusion sampling + VAE decode。
        这里需要先生成一次 caption 和 z_hat，但不计入 decode 时间。
        """
        with suppress_output():
            generated_text, z_hat, H, W = self._encode_for_complexity(x)

        def fn():
            with suppress_output():
                return self.pipe(
                    generated_text,
                    z_hat,
                    height=H,
                    width=W,
                    num_inference_steps=self.cfg_perco.num_inference_steps,
                    guidance_scale=self.cfg_perco.guidance_scale,
                    generator=None,
                ).images[0]

        return time_ms(
            fn,
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

    @torch.no_grad()
    def encode_gflops(self, x):
        x = x[:1].to(self.device, dtype=torch.float).clamp(0, 1)

        wrapper = PerCoEncodeWrapper(
            self.pipe,
        ).to(self.device).eval()

        with suppress_output():
            return gflops(wrapper, x)

    @torch.no_grad()
    def decode_gflops(self, x):
        with suppress_output():
            generated_text, z_hat, H, W = self._encode_for_complexity(x)

        total = 0.0

        text_g = None
        unet_step_g = None
        vae_dec_g = None

        # 1. CLIP text encoder FLOPs
        try:
            text_inputs = self.pipe.tokenizer(
                generated_text,
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(self.device)

            text_wrapper = PerCoTextEncoderWrapper(
                self.pipe.text_encoder,
            ).to(self.device).eval()

            with suppress_output():
                text_g = gflops(text_wrapper, input_ids)

            if text_g is not None:
                total += text_g
        except Exception as e:
            print(f"[PerCo] text_encoder FLOPs failed: {e}")

        # 2. UNet one step FLOPs × steps
        try:
            do_cfg = self.cfg_perco.guidance_scale > 1.0

            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                generated_text,
                self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
            )

            if do_cfg:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            self.pipe.scheduler.set_timesteps(
                self.cfg_perco.num_inference_steps,
                device=self.device,
            )

            timestep = self.pipe.scheduler.timesteps[0]

            latent_shape = (
                1,
                self.pipe.unet.config.in_channels,
                H // self.pipe.vae_scale_factor,
                W // self.pipe.vae_scale_factor,
            )

            latents = torch.randn(
                latent_shape,
                device=self.device,
                dtype=prompt_embeds.dtype,
            )

            unet_wrapper = PerCoUNetWrapper(
                self.pipe,
                prompt_embeds=prompt_embeds,
                local_features=z_hat,
                do_cfg=do_cfg,
            ).to(self.device).eval()

            with suppress_output():
                unet_step_g = gflops(
                    unet_wrapper,
                    (latents, timestep),
                )

            if unet_step_g is not None:
                total += unet_step_g * int(self.cfg_perco.num_inference_steps)

        except Exception as e:
            print(f"[PerCo] UNet FLOPs failed: {e}")

        # 3. VAE decoder FLOPs
        try:
            latent_shape = (
                1,
                self.pipe.unet.config.in_channels,
                H // self.pipe.vae_scale_factor,
                W // self.pipe.vae_scale_factor,
            )

            latents = torch.randn(
                latent_shape,
                device=self.device,
                dtype=torch.float32,
            )

            vae_dec_wrapper = PerCoVAEDecodeWrapper(
                self.pipe.vae,
            ).to(self.device).eval()

            with suppress_output():
                vae_dec_g = gflops(vae_dec_wrapper, latents)

            if vae_dec_g is not None:
                total += vae_dec_g

        except Exception as e:
            print(f"[PerCo] VAE decoder FLOPs failed: {e}")

        print(
            "[PerCo Decode FLOPs] "
            f"text={text_g}, "
            f"unet_step={unet_step_g}, "
            f"steps={self.cfg_perco.num_inference_steps}, "
            f"vae_decode={vae_dec_g}"
        )

        if total == 0.0:
            return None

        return total