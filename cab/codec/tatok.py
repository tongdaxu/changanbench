import torch
import numpy as np
from cab.codec.abs import ImageCodecIface
import torch
from huggingface_hub import hf_hub_download
from cab.models.tatok.t2i_inference import T2IConfig, TextToImageInference

class TaTokImageTokenizer(ImageCodecIface):
    def __init__(self, quality, ckpt_name, scale, seq_length, codebook_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality = quality
        self.ckpt_path = hf_hub_download("csuhan/TA-Tok", ckpt_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = scale
        self.seq_length = seq_length
        self.codebook_size = codebook_size
        # init tokenizer
        config = T2IConfig(
            ar_path=self.ckpt_path,
            encoder_path = hf_hub_download("csuhan/TA-Tok", "ta_tok.pth"),
            decoder_path = hf_hub_download("peizesun/llamagen_t2i", "vq_ds16_t2i.pt"),
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            scale = self.scale,
            seq_len = self.seq_length,
        )

        self.model = TextToImageInference(config)

    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device)
        indices = self.model.visual_tokenizer.get_encoder_indices((x + 1.0) / 2.0)
        rec = self.model.visual_tokenizer.decode_from_encoder_indices(indices).to(torch.float32) / 255.0 # reconstruction

        rec = rec.permute(0,3,1,2)
        xhat = rec * 2.0 - 1.0
        xhat = xhat.to(self.device)

        bits_per_token = np.log2(self.codebook_size)
        bpp = self.seq_length * bits_per_token / (x.size(2) * x.size(3))
        bpp = torch.tensor([bpp], dtype=torch.float32, device=x.device)
        return xhat, bpp


