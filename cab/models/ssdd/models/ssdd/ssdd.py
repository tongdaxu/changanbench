# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
from os import PathLike
from pathlib import Path
from typing import Mapping, Optional, Union

import torch
import torch.nn as nn
import yaml
from safetensors.torch import load_file as safe_load_file

from ...flow import FlowMatchingTrainer, FMEulerSampler
from ...mutils.torch_utils import freeze_model
from ...mutils.train_utils import init_weights
from ..blocks.diag_gauss import DiagonalGaussianDistribution
from ..model_utils import TrainStepResult
from ..pretrained_encoders import SDVAEPretrainedEncoder
from ..vq_encoder import VQEncoder
from .uvit import UViTDecoder


class SSDD(nn.Module):
    def __init__(
        self,
        encoder: Optional[Mapping | nn.Module] = None,
        encoder_checkpoint: Optional[str] = None,
        encoder_train: bool = False,
        decoder: Optional[Mapping] = None,
        fm_trainer: Optional[Mapping] = None,
        fm_sampler: Optional[Mapping] = None,
        checkpoint: Optional[str] = None,
    ):
        super().__init__()

        ### Submodules ###

        self.encoder_train = encoder_train
        self.encoder = self.make_encoder(encoder, encoder_checkpoint)
        # self.decoder = UViTDecoder.make(decoder)
        if isinstance(decoder, Mapping):
            decoder_size = decoder.get("size", None)
            decoder_kwargs = {k: v for k, v in decoder.items() if k != "size"}
            self.decoder = UViTDecoder.make(size=decoder_size, **decoder_kwargs)
        else:
            self.decoder = UViTDecoder.make(decoder)

        ### Flow-matching ###

        self.fm_trainer = FlowMatchingTrainer(**(fm_trainer or {}))
        self.fm_sampler = FMEulerSampler(**(fm_sampler or {}))

        ## Weights init ###
        self.init_weights(checkpoint=checkpoint)

    def make_encoder(self, encoder, encoder_checkpoint):
        if encoder == "sdvae":
            encoder = SDVAEPretrainedEncoder()

        if not isinstance(encoder, nn.Module):
            # Check if matches pattern f?c? with regex
            assert isinstance(encoder, str)
            vqenc_cfg_re = r"^f(\d+)c(\d+)$"
            vqenc_cfg_match = re.match(vqenc_cfg_re, encoder)
            if vqenc_cfg_match:
                patch_size = int(vqenc_cfg_match.group(1))
                z_dim = int(vqenc_cfg_match.group(2))
                encoder = VQEncoder.make(z_dim=z_dim, patch_size=patch_size, encoder_checkpoint=encoder_checkpoint)
            else:
                raise ValueError(f"Invalid encoder config: {encoder}")

        if not self.encoder_train:
            freeze_model(encoder)

        return encoder

    def init_weights(self, method="kaiming_normal", **kwargs):
        init_weights(self, method=method, **kwargs)

    def encode(self, x) -> DiagonalGaussianDistribution:
        return self.encoder(x)

    def decode(
        self,
        z: torch.Tensor,
        steps: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fn_kwargs = {"z": z}

        B, _, zH, zW = z.shape
        H, W = zH * self.encoder.patch_size, zW * self.encoder.patch_size

        ret = self.fm_sampler.sample(
            self.decoder,
            self.fm_trainer,
            shape=(B, self.decoder.out_dim, H, W),
            steps=steps,
            fn_kwargs=fn_kwargs,
            noise=noise,
        )

        return ret

    def forward(
        self,
        gt_x: torch.Tensor,
        steps: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        from_noise=False,
        as_teacher=False,
    ) -> Union[torch.Tensor, TrainStepResult]:
        # Encoder
        encoded = None
        if z is None:
            if self.encoder_train:
                encoded = self.encode(gt_x)
            else:
                with torch.no_grad():
                    encoded = self.encode(gt_x)
            z = encoded.sample() if self.training else encoded.mode()

        # Decoder
        if not self.training:
            if as_teacher:
                if noise is None:
                    noise = torch.randn_like(gt_x)
            x_gen = self.decode(z, steps=steps, noise=noise)
            if as_teacher:
                return x_gen, z, noise
            return x_gen
        else:
            t = self.fm_trainer.sample_t(gt_x.shape[0], device=gt_x.device)
            if from_noise:
                t = torch.ones_like(t)

            # Use decoder to get a diffusion reconstruction loss
            diff_loss, (x_t, noise, noise_t, v_pred) = self.fm_trainer.loss(self.decoder, x=gt_x, t=t, fn_kwargs={"z": z}, noise=noise)

            # Compute auxiliary losses
            x0_pred = self.fm_trainer.step(x_t, v_pred, noise_t)

            losses = {"diffusion": diff_loss}
            if encoded is not None and self.encoder_train:
                losses["kl"] = encoded.kl().mean()

            return TrainStepResult(
                x0_gt=gt_x,
                x0_pred=x0_pred,
                xt=x_t,
                t=t,
                z=z,
                noise=noise,
                losses=losses,
            )

    def get_last_layer_weight(self):
        return self.decoder.conv_out.weight

    ### Loading / Checkpointing ###

    def load(
        self,
        weights: Union[str, Path, Mapping],
        strict: bool = True,
        freeze=False,
        eval=None,
    ):
        if not isinstance(weights, Mapping):
            weights = safe_load_file(weights)
        self.load_state_dict(weights, strict=strict)

        if eval or (eval is None and freeze):
            self.eval()
        if freeze:
            self.requires_grad_(False)
        return self

    @classmethod
    def build(cls, config, checkpoint=None, freeze=True, eval=True):
        """Build the model from a config name."""
        if isinstance(config, (str, PathLike)):
            with open(config, "r") as yaml_file:
                model_args = yaml.safe_load(yaml_file)["ssdd"]
        elif isinstance(config, Mapping):
            model_args = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}. Expected model size, path, or a mapping.")

        model = cls(**model_args)

        if checkpoint:
            model.load(checkpoint)
        if eval:
            model.eval()
        if freeze:
            model.requires_grad_(False)
        return model
