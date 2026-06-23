# cab/codec/jpeg.py
import torch
import numpy as np
from PIL import Image
import io
import os
from cab.codec.abs import ImageCodecIface

class JPEGImageCodec(ImageCodecIface):
    """JPEG Image Codec wrapper for standard JPEG compression evaluation."""
    
    def __init__(self, quality=90, *args, **kwargs):
        """
        Args:
            quality: JPEG quality parameter (1-100)
        """
        super().__init__(*args, **kwargs)
        self.quality = quality

    def param_count(self, x, *args, **kwargs):
        return super().param_count(x, *args, **kwargs)
    
    @torch.no_grad()
    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: (B, 3, H, W) tensor in range [0, 1]
        Returns:
            xhat: reconstructed image (B, 3, H, W) in range [0, 1]
            bpp: (B,) bits per pixel
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Convert to CPU for PIL processing
        x_cpu = (x * 255).clamp(0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        
        recons = []
        bpps = []
        
        for img_np in x_cpu:
            # Encode to JPEG
            pil_img = Image.fromarray(img_np, mode='RGB')
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=self.quality)
            bytes_encoded = buffer.getbuffer().nbytes
            
            # Decode from JPEG
            buffer.seek(0)
            pil_rec = Image.open(buffer).convert('RGB')
            rec_np = np.array(pil_rec, dtype=np.float32) / 255.0
            
            # Calculate BPP
            bpp = (bytes_encoded * 8.0) / (H * W)
            
            recons.append(torch.from_numpy(rec_np).permute(2, 0, 1))
            bpps.append(bpp)
        
        rec_tensor = torch.stack(recons, dim=0).to(device=device, dtype=x.dtype)
        bpp_tensor = torch.tensor(bpps, dtype=torch.float32, device=device)
        
        return rec_tensor, bpp_tensor