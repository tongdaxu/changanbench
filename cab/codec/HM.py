# cab/codec/hm.py
import torch
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
import os
from cab.codec.abs import ImageCodecIface

class HMImageCodec(ImageCodecIface):
    """HM (H.265/HEVC) Image Codec wrapper using HM reference software."""
    
    def __init__(self, qp=32, hm_encoder_path=None, *args, **kwargs):
        """
        Args:
            qp: Quantization parameter (0-51, lower = better quality)
            hm_encoder_path: Path to HM encoder executable (e.g., TAppEncoderStatic)
        """
        super().__init__(*args, **kwargs)
        self.qp = qp
        self.hm_encoder_path = hm_encoder_path
    
    @torch.no_grad()
    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: (B, 3, H, W) tensor in range [0, 1]
        Returns:
            xhat: reconstructed image (B, 3, H, W)
            bpp: (B,) bits per pixel
        """
        B, C, H, W = x.shape
        device = x.device
        x_cpu = (x * 255).clamp(0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        
        recons = []
        bpps = []
        
        with tempfile.TemporaryDirectory(prefix="hm_") as tmpdir:
            for i, img_np in enumerate(x_cpu):
                # Save as YUV 4:2:0 (standard for HEVC)
                pil_img = Image.fromarray(img_np)
                yuv_img = pil_img.convert('YCbCr')  # Simplified YUV conversion
                yuv_path = Path(tmpdir) / f"input_{i}.yuv"
                bin_path = Path(tmpdir) / f"output_{i}.bin"
                rec_path = Path(tmpdir) / f"rec_{i}.yuv"
                
                # Save as raw YUV
                yuv_arr = np.array(yuv_img).astype(np.uint8)

                Y  = yuv_arr[:, :, 0]
                Cb = yuv_arr[:, :, 1][::2, ::2]
                Cr = yuv_arr[:, :, 2][::2, ::2]

                with open(yuv_path, "wb") as f:
                    Y.tofile(f)
                    Cb.tofile(f)
                    Cr.tofile(f)
                
                # Encode with HM
                cmd = [
                    self.hm_encoder_path,
                    "-c", "config/image_codecs/encoder_intra_main.cfg",
                    "-c",  "config/image_codecs/HM.cfg",
                    "-i", str(yuv_path),
                    "-wdt", str(W),
                    "-hgt", str(H),
                    "-f", "1",  # 1 frame
                    "-fr", "1",
                    "-q", str(self.qp),
                    "-o", str(rec_path),
                    "-b", str(bin_path),
                ]
                
                # subprocess.run(cmd, capture_output=True, check=True)
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"HM encoder failed with code {result.returncode}\n"
                        f"STDOUT:\n{result.stdout}\n"
                        f"STDERR:\n{result.stderr}"
                    )
                
                # Calculate BPP
                bytes_encoded = bin_path.stat().st_size
                bpp = (bytes_encoded * 8.0) / (H * W)
                
                # Read reconstructed YUV and convert back to RGB
                buf = np.fromfile(rec_path, dtype=np.uint8)

                n_y = H * W
                n_c = (H // 2) * (W // 2)

                Y = buf[:n_y].reshape(H, W)

                Cb = buf[n_y:n_y + n_c].reshape(H // 2, W // 2)
                Cr = buf[n_y + n_c:n_y + 2 * n_c].reshape(H // 2, W // 2)

                Cb = np.repeat(np.repeat(Cb, 2, axis=0), 2, axis=1)
                Cr = np.repeat(np.repeat(Cr, 2, axis=0), 2, axis=1)

                rec_yuv = np.stack([Y, Cb, Cr], axis=-1)
                rec_pil = Image.fromarray(rec_yuv, mode="YCbCr").convert("RGB")
                rec_np = np.array(rec_pil, dtype=np.float32) / 255.0
                
                recons.append(torch.from_numpy(rec_np).permute(2, 0, 1))
                bpps.append(bpp)
        
        rec_tensor = torch.stack(recons, dim=0).to(device=device, dtype=x.dtype)
        bpp_tensor = torch.tensor(bpps, dtype=torch.float32, device=device)
        
        return rec_tensor, bpp_tensor