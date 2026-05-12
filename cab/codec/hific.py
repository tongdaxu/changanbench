import torch
import os
import numpy as np
from cab.codec.abs import ImageCodecIface
from cab.models.hific_src.model import Model 
from cab.models.hific_src.helpers import utils
from cab.models.hific_src.default_config import ModelModes
from collections import defaultdict

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class HiFiCImageCodec(ImageCodecIface):
    def __init__(self, quality, ckpt_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality = quality
        self.ckpt_path = ckpt_path

        self.device = utils.get_device()
        self.logger = utils.logger_setup(logpath=os.path.join('logs'), filepath=os.path.abspath(__file__))
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        model_args = ckpt['args']
        model_args = Struct(**model_args)  

        self.model = Model(
            args=model_args,
            logger=self.logger,
            storage_train=defaultdict(list),
            storage_test=defaultdict(list),
            model_mode=ModelModes.EVALUATION,
            model_type=model_args.model_type,  
        ).to(self.device)
        
        self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):

        x = x.to(self.device, dtype=torch.float)
        xhat, bpp = self.model(x)
        bpp = torch.tensor([bpp], dtype=torch.float32, device=x.device)
        return xhat, bpp