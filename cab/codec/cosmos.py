import torch
import numpy as np
from cab.codec.abs import ImageCodecIface
from cab.models.cosmos_tokenizer.image_lib import ImageTokenizer
from cab.complexity import params_m, time_ms, gflops

class CosmosEncodeWrapper(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, x):
        out = self.tokenizer.encode(x)
        if isinstance(out, tuple):
            return out[0]
        return out


class CosmosDecodeWrapper(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, z):
        return self.tokenizer.decode(z)

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

    @torch.no_grad()
    def encode_tokens(self, x):
        x = x.to(self.device, dtype=self.tokenizer._dtype)
        out = self.tokenizer.encode(x)

        if isinstance(out, tuple):
            return out[0]

        return out

    @torch.no_grad()
    def decode_tokens(self, z):
        if isinstance(z, torch.Tensor):
            z = z.to(self.device)
        return self.tokenizer.decode(z)

    def encode_params_m(self):
        if self.tokenizer._enc_model is None:
            return 0.0
        return params_m(self.tokenizer._enc_model)

    def decode_params_m(self):
        if self.tokenizer._dec_model is None:
            return 0.0
        return params_m(self.tokenizer._dec_model)

    @torch.no_grad()
    def encode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, dtype=self.tokenizer._dtype)
        enc = CosmosEncodeWrapper(self.tokenizer).to(self.device).eval()

        return time_ms(
            lambda: enc(x),
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

    @torch.no_grad()
    def decode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, dtype=self.tokenizer._dtype)

        z = self.encode_tokens(x)
        dec = CosmosDecodeWrapper(self.tokenizer).to(self.device).eval()

        return time_ms(
            lambda: dec(z),
            self.device,
            warmup=warmup,
            repeat=repeat,
        )
    
    @torch.no_grad()
    def encode_gflops(self, x):
        x = x.to(self.device, dtype=self.tokenizer._dtype)
        enc = CosmosEncodeWrapper(self.tokenizer).to(self.device).eval()
        return gflops(enc, x)

    @torch.no_grad()
    def decode_gflops(self, x):
        x = x.to(self.device, dtype=self.tokenizer._dtype)

        z = self.encode_tokens(x)
        dec = CosmosDecodeWrapper(self.tokenizer).to(self.device).eval()

        return gflops(dec, z)