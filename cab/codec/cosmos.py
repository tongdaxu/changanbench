import torch
import numpy as np
from cab.codec.abs import ImageCodecIface
from cab.models.cosmos_tokenizer.image_lib import ImageTokenizer

class CosmosImageTokenizer(ImageCodecIface):
    def __init__(self, quality, checkpoint_enc, checkpoint_dec, downsample_factor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality = quality
        self.checkpoint_enc = checkpoint_enc
        self.checkpoint_dec = checkpoint_dec
        self.downsample_factor = downsample_factor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = ImageTokenizer(checkpoint_enc=self.checkpoint_enc, checkpoint_dec=self.checkpoint_dec).to(self.device)
        self.tokenizer.eval()

    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device, dtype=torch.float)
        
        # Encode and decode using the tokenizer
        xhat = self.tokenizer(x).float()
        
        # Calculate bpp based on the number of tokens and image size
        num_codebook_entries = 64000 #(8,8,8,5,5,5)
        bits_per_token = np.log2(num_codebook_entries)
        num_tokens = (x.size(2) // self.downsample_factor) * (x.size(3) // self.downsample_factor)
        bpp = (num_tokens * bits_per_token) / (x.size(2) * x.size(3))
        
        return xhat, torch.tensor([bpp], dtype=torch.float32, device=x.device)

