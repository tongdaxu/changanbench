import torch
import math
from vector_quantize_pytorch import FSQ
from cab.codec.abs import ImageCodecIface


class FSQImageTokenizer(ImageCodecIface):
    """Finite Scalar Quantization tokenizer for image compression.
    
    Args:
        levels: List of quantization levels per dimension. Default: [8, 5, 5, 5]
        encode_bpp: If True, calculate actual bits per pixel based on codebook size.
    """

    def __init__(self, levels=None, downsample_factor=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.levels = levels
        self.downsample_factor = downsample_factor
        self.dim = len(levels)
        self.quantizer = FSQ(levels=levels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantizer.to(self.device)
        
        # Calculate codebook size and bits needed
        self.codebook_size = math.prod(levels)
        self.bits_per_token = math.ceil(math.log2(self.codebook_size))

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: Input image tensor (B, 3, H, W) in range [0, 1]
        
        Returns:
            xhat: Reconstructed image (B, 3, H, W) in range [0, 1]
            bpp: Bits per pixel (B,) tensor
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Reshape: (B, 3, H, W) -> (B*H*W, 3)
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Normalize to [-1, 1] range for FSQ
        x_normalized = x_flat * 2 - 1
        
        # Add channel dimension: (B*H*W, 3) -> (B*H*W, 1, 3)
        x_in = x_normalized.unsqueeze(1)
        
        # Quantize
        x_quantized, indices = self.quantizer(x_in)
        x_quantized = x_quantized.squeeze(1)  # (B*H*W, self.dim)
        
        # Use first 3 channels for reconstruction
        x_recon_flat = x_quantized[:, :C]
        
        # Denormalize back to [0, 1] range
        x_recon_flat = (x_recon_flat + 1) / 2
        x_recon_flat = torch.clamp(x_recon_flat, 0, 1)
        
        # Reshape back to image: (B*H*W, 3) -> (B, 3, H, W)
        xhat = x_recon_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Calculate BPP
        num_tokens = (x.size(2) // self.downsample_factor) * (x.size(3) // self.downsample_factor)
        bpp = (num_tokens * self.bits_per_token) / (x.size(2) * x.size(3))
        
        bpp = torch.full((B,), bpp, dtype=torch.float32, device=device)
        
        return xhat, bpp