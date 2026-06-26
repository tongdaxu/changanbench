# cab/codec/jpeg.py
import torch
import numpy as np
from PIL import Image
import io

from cab.codec.abs import ImageCodecIface
from cab.complexity import time_ms

class JPEGImageCodec(ImageCodecIface):
    """JPEG Image Codec wrapper for standard JPEG compression evaluation."""

    def __init__(self, quality=90, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality = quality

    def fake_input(self, image_size=256, batch_size=1, device=None):
        if device is None:
            device = torch.device("cpu")
        return torch.rand(batch_size, 3, image_size, image_size, device=device)

    def encode_params_m(self):
        return 0.0

    def decode_params_m(self):
        return 0.0

    def encode_gflops(self, x):
        return None

    def decode_gflops(self, x):
        return None

    def _tensor_to_pil_list(self, x):
        x_cpu = (
            (x * 255)
            .clamp(0, 255)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        return [Image.fromarray(img_np, mode="RGB") for img_np in x_cpu]

    def _encode_pil_images(self, pil_images):
        buffers = []

        for pil_img in pil_images:
            buffer = io.BytesIO()
            pil_img.save(
                buffer,
                format="JPEG",
                quality=self.quality,
            )
            buffers.append(buffer)

        return buffers

    def _decode_buffers(self, buffers):
        recs = []

        for buffer in buffers:
            buffer.seek(0)
            pil_rec = Image.open(buffer).convert("RGB")
            rec_np = np.array(pil_rec, dtype=np.float32) / 255.0
            recs.append(torch.from_numpy(rec_np).permute(2, 0, 1))

        return recs

    @torch.no_grad()
    def encode_time_ms(self, x, warmup=5, repeat=20):
        x = x.detach().to(dtype=torch.float)
        pil_images = self._tensor_to_pil_list(x)

        t = time_ms(
            lambda: self._encode_pil_images(pil_images),
            torch.device("cpu"),
            warmup=warmup,
            repeat=repeat,
        )

        return t / x.shape[0]

    @torch.no_grad()
    def decode_time_ms(self, x, warmup=5, repeat=20):
        x = x.detach().to(dtype=torch.float)
        pil_images = self._tensor_to_pil_list(x)
        buffers = self._encode_pil_images(pil_images)

        t = time_ms(
            lambda: self._decode_buffers(buffers),
            torch.device("cpu"),
            warmup=warmup,
            repeat=repeat,
        )

        return t / x.shape[0]

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):
        B, C, H, W = x.shape
        device = x.device

        pil_images = self._tensor_to_pil_list(x)
        buffers = self._encode_pil_images(pil_images)

        recons = []
        bpps = []

        for buffer in buffers:
            bytes_encoded = buffer.getbuffer().nbytes

            buffer.seek(0)
            pil_rec = Image.open(buffer).convert("RGB")
            rec_np = np.array(pil_rec, dtype=np.float32) / 255.0

            bpp = bytes_encoded * 8.0 / (H * W)

            recons.append(torch.from_numpy(rec_np).permute(2, 0, 1))
            bpps.append(bpp)

        rec_tensor = torch.stack(recons, dim=0).to(
            device=device,
            dtype=x.dtype,
        )

        bpp_tensor = torch.tensor(
            bpps,
            dtype=torch.float32,
            device=device,
        )

        return rec_tensor, bpp_tensor