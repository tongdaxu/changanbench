import torch
import math
from omegaconf import OmegaConf
from cab.utils import get_obj_from_str
from cab.codec.abs import ImageCodecIface
from cab.complexity import params_m, time_ms, gflops

def instantiate_from_config(config):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

class FSQEncodeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.encode(x, return_reg_log=False)


class FSQDecodeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z):
        return self.model.decode(z)

class FSQImageTokenizer(ImageCodecIface):
    """Finite Scalar Quantization tokenizer for image compression.
    
    Args:
        levels: List of quantization levels per dimension. Default: [8, 5, 5, 5]
        encode_bpp: If True, calculate actual bits per pixel based on codebook size.
    """

    def __init__(self, ckpt_path=None, base=None, levels=None, downsample_factor=32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = OmegaConf.load(base)
        self.model = instantiate_from_config(config.model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.eval().to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path)["state_dict"])


        self.downsample_factor = downsample_factor
        # Calculate codebook size and bits needed
        self.codebook_size = math.prod(levels)
        self.bits_per_token = math.ceil(math.log2(self.codebook_size))

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):
        x = x.to(self.device, dtype=torch.float)
        zhat = self.model.encode(x, return_reg_log=False)
        xhat = self.model.decode(zhat)
        
        
        # Calculate BPP
        num_tokens = (x.size(2) // self.downsample_factor) * (x.size(3) // self.downsample_factor)
        bpp = (num_tokens * self.bits_per_token) / (x.size(2) * x.size(3))
        
        bpp = torch.tensor([bpp], dtype=torch.float32, device=x.device)
        
        return xhat, bpp
    
    def fake_input(self, image_size=256, batch_size=1, device=None):
        if device is None:
            device = self.device
        return torch.rand(batch_size, 3, image_size, image_size, device=device)

    def encode_params_m(self):
        if hasattr(self.model, "encoder"):
            return params_m(self.model.encoder)
        return params_m(self.model)

    def decode_params_m(self):
        if hasattr(self.model, "decoder"):
            return params_m(self.model.decoder)
        return params_m(self.model)

    @torch.no_grad()
    def encode_tokens(self, x):
        x = x.to(self.device, dtype=torch.float)
        return self.model.encode(x, return_reg_log=False)

    @torch.no_grad()
    def decode_tokens(self, z):
        z = z.to(self.device)
        return self.model.decode(z)

    @torch.no_grad()
    def encode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, dtype=torch.float)
        enc = FSQEncodeWrapper(self.model).to(self.device).eval()

        return time_ms(
            lambda: enc(x),
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

    @torch.no_grad()
    def decode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, dtype=torch.float)
        z = self.encode_tokens(x).detach()

        dec = FSQDecodeWrapper(self.model).to(self.device).eval()

        return time_ms(
            lambda: dec(z),
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

    @torch.no_grad()
    def encode_gflops(self, x):
        x = x.to(self.device, dtype=torch.float)
        enc = FSQEncodeWrapper(self.model).to(self.device).eval()
        return gflops(enc, x)

    @torch.no_grad()
    def decode_gflops(self, x):
        x = x.to(self.device, dtype=torch.float)
        z = self.encode_tokens(x).detach()

        dec = FSQDecodeWrapper(self.model).to(self.device).eval()
        return gflops(dec, z)