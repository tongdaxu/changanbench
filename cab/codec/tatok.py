import torch
import numpy as np
from cab.codec.abs import ImageCodecIface
import torch
from huggingface_hub import hf_hub_download
from pathlib import Path
from cab.models.tatok.t2i_inference import T2IConfig, TextToImageInference
from cab.complexity import params_m, time_ms, gflops


def _project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def _resolve_or_download(repo_id, filename, local_path=None, token=None, revision=None):
    if local_path:
        path = _project_path(local_path)
        if path.exists():
            return str(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return hf_hub_download(
            repo_id,
            filename,
            local_dir=str(path.parent),
            local_dir_use_symlinks=False,
            token=token,
            revision=revision,
        )
    return hf_hub_download(repo_id, filename, token=token, revision=revision)

class TaTokEncodeWrapper(torch.nn.Module):
    def __init__(self, visual_tokenizer):
        super().__init__()
        self.visual_tokenizer = visual_tokenizer

    def forward(self, x):
        # x: [-1, 1] -> [0, 1]
        x = (x + 1.0) / 2.0
        return self.visual_tokenizer.get_encoder_indices(x)

class TaTokEncoderDecodeBottleneckWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, indices):
        return self.encoder.decode_from_bottleneck(indices)


class TaTokDecoderWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, ar_indices):
        return self.decoder.decode_from_bottleneck(ar_indices)

class TaTokImageTokenizer(ImageCodecIface):
    def __init__(
        self,
        quality,
        ckpt_name,
        scale,
        seq_length,
        codebook_size,
        ckpt_path=None,
        encoder_path=None,
        decoder_path=None,
        hf_token=None,
        hf_revision=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.quality = quality
        self.ckpt_path = _resolve_or_download(
            "csuhan/TA-Tok",
            ckpt_name,
            ckpt_path,
            token=hf_token,
            revision=hf_revision,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = scale
        self.seq_length = seq_length
        self.codebook_size = codebook_size
        encoder_path = _resolve_or_download(
            "csuhan/TA-Tok",
            "ta_tok.pth",
            encoder_path,
            token=hf_token,
            revision=hf_revision,
        )
        decoder_path = _resolve_or_download(
            "peizesun/llamagen_t2i",
            "vq_ds16_t2i.pt",
            decoder_path,
            token=hf_token,
            revision=hf_revision,
        )
        # init tokenizer
        config = T2IConfig(
            ar_path=self.ckpt_path,
            encoder_path = encoder_path,
            decoder_path = decoder_path,
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
    
    def fake_input(self, image_size=256, batch_size=1, device=None):
        if device is None:
            device = self.device
        return torch.rand(batch_size, 3, image_size, image_size, device=device) * 2 - 1

    def encode_params_m(self):
        # image -> encoder indices
        modules = [
            self.model.visual_tokenizer.encoder,
        ]
        return sum(params_m(m) for m in modules)

    def decode_params_m(self):
        # indices -> encoder feats -> AR sample -> VQ decoder
        modules = [
            self.model.visual_tokenizer.encoder.bottleneck,
            self.model.visual_tokenizer.encoder.decoder,
            self.model.visual_tokenizer.encoder.decode_task_layer,
            self.model.visual_tokenizer.ar_model,
            self.model.visual_tokenizer.decoder,
        ]
        return sum(params_m(m) for m in modules)

    @torch.no_grad()
    def encode_tokens(self, x):
        x = x.to(self.device)
        x01 = (x + 1.0) / 2.0
        return self.model.visual_tokenizer.get_encoder_indices(x01)

    @torch.no_grad()
    def decode_tokens(self, indices):
        indices = indices.to(self.device)
        rec = self.model.visual_tokenizer.decode_from_encoder_indices(
            indices,
            {"cfg_scale": getattr(self.model.config, "cfg_scale", 1.0)},
        )

        # uint8 BHWC -> float BCHW [-1, 1]
        rec = rec.to(torch.float32) / 255.0
        rec = rec.permute(0, 3, 1, 2).contiguous()
        return rec * 2.0 - 1.0

    @torch.no_grad()
    def encode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device)

        enc = TaTokEncodeWrapper(
            self.model.visual_tokenizer,
        ).to(self.device).eval()

        return time_ms(
            lambda: enc(x),
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

    @torch.no_grad()
    def decode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device)
        indices = self.encode_tokens(x).detach()

        decode_args = {
            "cfg_scale": getattr(self.model.config, "cfg_scale", 1.0),
            "temperature": getattr(self.model.config, "temperature", 1.0),
            "top_k": getattr(self.model.config, "top_k", 0),
            "top_p": getattr(self.model.config, "top_p", 1.0),
        }

        return time_ms(
            lambda: self.model.visual_tokenizer.decode_from_encoder_indices(
                indices,
                decode_args,
            ),
            self.device,
            warmup=warmup,
            repeat=repeat,
        )
    
    @torch.no_grad()
    def encode_gflops(self, x):
        x = x.to(self.device)

        enc = TaTokEncodeWrapper(
            self.model.visual_tokenizer,
        ).to(self.device).eval()

        return gflops(enc, x)

    @torch.no_grad()
    def decode_gflops(self, x):
        x = x.to(self.device)
        indices = self.encode_tokens(x).detach()

        vt = self.model.visual_tokenizer

        # 1. encoder.decode_from_bottleneck(indices)
        enc_dec_wrapper = TaTokEncoderDecodeBottleneckWrapper(
            vt.encoder
        ).to(self.device).eval()

        enc_dec_flops = gflops(enc_dec_wrapper, indices)

        # 2. 生成 ar_indices
        # 注意：这里不统计 ar_sample FLOPs，只用它生成 decoder 输入
        decode_args = {
            "cfg_scale": getattr(self.model.config, "cfg_scale", 1.0),
            "temperature": getattr(self.model.config, "temperature", 1.0),
            "top_k": getattr(self.model.config, "top_k", 0),
            "top_p": getattr(self.model.config, "top_p", 1.0),
        }

        encoder_feats = vt.encoder.decode_from_bottleneck(indices)

        try:
            ar_indices = vt.ar_sample(encoder_feats, decode_args)
        except TypeError:
            ar_indices = vt.ar_sample(encoder_feats)

        if isinstance(ar_indices, tuple):
            ar_indices = ar_indices[0]

        ar_indices = ar_indices.detach()

        # 3. decoder.decode_from_bottleneck(ar_indices)
        dec_wrapper = TaTokDecoderWrapper(
            vt.decoder
        ).to(self.device).eval()

        dec_flops = gflops(dec_wrapper, ar_indices)

        if enc_dec_flops is None and dec_flops is None:
            return None

        total = 0.0
        if enc_dec_flops is not None:
            total += enc_dec_flops
        if dec_flops is not None:
            total += dec_flops

        print(
            f"[TA-Tok Decode FLOPs] "
            f"encoder_decode={enc_dec_flops}, "
            f"vq_decoder={dec_flops}, "
            f"AR_sampling=N/A"
        )

        return total
