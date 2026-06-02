import numpy as np
import torch
import os, sys
# ensure perco custom modules are importable by name used in pipeline configs
_perco_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "perco_src"))
if _perco_src not in sys.path:
    sys.path.insert(0, _perco_src)
from cab.codec.abs import ImageCodecIface
from cab.models.perco_src.config import ConfigPerco as cfg_perco
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from cab.models.perco_src.pipeline_sd_perco import StableDiffusionPipelinePerco
import zlib
import torchac
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage

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