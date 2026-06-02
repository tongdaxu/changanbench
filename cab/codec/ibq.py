import torch
import os, sys
_ibq_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "ibq"))
if sys.path.insert(0, _ibq_src):
    pass
import numpy as np
from omegaconf import OmegaConf
from cab.codec.abs import ImageCodecIface
# from cab.models.ibq.src.Open_MAGVIT2.models.lfqgan import VQModel
from cab.models.ibq.src.IBQ.models.ibqgan import IBQ
import yaml

## for different model configuration
MODEL_TYPE = {
    # "Open-MAGVIT2": VQModel,
    "IBQ": IBQ
}

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan_new(config, model_type, ckpt_path=None, is_gumbel=False):
    model = MODEL_TYPE[model_type](**config.model.init_args) 
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

class IBQImageTokenizer(ImageCodecIface):
    def __init__(self, ckpt_path, ibq_config, codebook_size, downsample_factor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ckpt_path = ckpt_path
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.codebook_size = codebook_size
        self.downsample_factor = downsample_factor

        config_model = load_config(ibq_config, display=False)
        model = load_vqgan_new(config_model, model_type="IBQ", ckpt_path=ckpt_path).to(self.device) #please specify your own path here

        model = model.cuda()
        model.load_state_dict(torch.load(ckpt_path)["state_dict"])
        self.model = model.eval()


    @torch.no_grad()
    def forward(self, x, *args, **kwargs):

        x = x.to(self.device, dtype=torch.float)
        quant, qloss, (_, _, indices) = self.model.encode(x)
        xhat = self.model.decode(quant).clamp(-1, 1)
        bits_per_token = np.log2(self.codebook_size)
        num_tokens = (x.size(2) // self.downsample_factor) * (x.size(3) // self.downsample_factor)
        total_bits = num_tokens * bits_per_token
        bpp = total_bits / (x.size(2) * x.size(3))
        bpp = torch.tensor([bpp], dtype=torch.float32, device=x.device)

        return xhat, bpp