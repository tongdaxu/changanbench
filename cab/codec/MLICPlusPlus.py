import torch
import os
import numpy as np
from cab.codec.abs import ImageCodecIface
from cab.models.MLICPlusPlus.config.args import test_options
from cab.models.MLICPlusPlus.config.config import model_config
from cab.models.MLICPlusPlus.models import *
from cab.models.MLICPlusPlus.utils.testing import compress_one_image, decompress_one_image
from cab.complexity import params_m, time_ms, gflops
from cab.models.MLICPlusPlus.utils.ckbd import ckbd_split, ckbd_anchor, ckbd_nonanchor

class MLICPPEncodeFLOPsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        self.model.update_resolutions(x.size(2) // 16, x.size(3) // 16)

        y = self.model.g_a(x)
        z = self.model.h_a(y)
        z_hat = torch.round(z)

        hyper_params = self.model.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)

        y_slices = y.chunk(self.model.slice_num, dim=1)
        y_hat_slices = []

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)

            if idx == 0:
                params_anchor = self.model.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)

                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)

                slice_anchor = torch.round(slice_anchor - means_anchor) + means_anchor

                lrp_anchor = self.model.lrp_anchor[idx](
                    torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1)
                )
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)

                local_ctx = self.model.local_context[idx](slice_anchor)
                params_nonanchor = self.model.entropy_parameters_nonanchor[idx](
                    torch.cat([local_ctx, hyper_params], dim=1)
                )
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)

                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)

                slice_nonanchor = torch.round(slice_nonanchor - means_nonanchor) + means_nonanchor

                y_hat_slice = slice_anchor + slice_nonanchor

                lrp_nonanchor = self.model.lrp_nonanchor[idx](
                    torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1)
                )
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)

                y_hat_slices.append(y_hat_slice)

            else:
                global_inter_ctx = self.model.global_inter_context[idx](
                    torch.cat(y_hat_slices, dim=1)
                )
                channel_ctx = self.model.channel_context[idx](
                    torch.cat(y_hat_slices, dim=1)
                )

                params_anchor = self.model.entropy_parameters_anchor[idx](
                    torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1)
                )
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)

                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)

                slice_anchor = torch.round(slice_anchor - means_anchor) + means_anchor

                lrp_anchor = self.model.lrp_anchor[idx](
                    torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1)
                )
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)

                global_intra_ctx = self.model.global_intra_context[idx](
                    y_hat_slices[-1],
                    slice_anchor,
                )
                local_ctx = self.model.local_context[idx](slice_anchor)

                params_nonanchor = self.model.entropy_parameters_nonanchor[idx](
                    torch.cat(
                        [
                            local_ctx,
                            global_intra_ctx,
                            global_inter_ctx,
                            channel_ctx,
                            hyper_params,
                        ],
                        dim=1,
                    )
                )
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)

                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)

                slice_nonanchor = torch.round(slice_nonanchor - means_nonanchor) + means_nonanchor

                y_hat_slice = slice_anchor + slice_nonanchor

                lrp_nonanchor = self.model.lrp_nonanchor[idx](
                    torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1)
                )
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)

                y_hat_slices.append(y_hat_slice)

        return torch.cat(y_hat_slices, dim=1)

class MLICPPDecodeFLOPsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z_hat):
        self.model.update_resolutions(z_hat.size(2) * 4, z_hat.size(3) * 4)

        hyper_params = self.model.h_s(z_hat)
        hyper_scales, hyper_means = hyper_params.chunk(2, 1)

        y_hat_slices = []

        for idx in range(self.model.slice_num):
            if idx == 0:
                params_anchor = self.model.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)

                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)

                slice_anchor = torch.zeros_like(means_anchor)

                lrp_anchor = self.model.lrp_anchor[idx](
                    torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1)
                )
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)

                local_ctx = self.model.local_context[idx](slice_anchor)
                params_nonanchor = self.model.entropy_parameters_nonanchor[idx](
                    torch.cat([local_ctx, hyper_params], dim=1)
                )
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)

                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                slice_nonanchor = torch.zeros_like(means_nonanchor)

                y_hat_slice = slice_anchor + slice_nonanchor

                lrp_nonanchor = self.model.lrp_nonanchor[idx](
                    torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1)
                )
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)

                y_hat_slices.append(y_hat_slice)

            else:
                global_inter_ctx = self.model.global_inter_context[idx](
                    torch.cat(y_hat_slices, dim=1)
                )
                channel_ctx = self.model.channel_context[idx](
                    torch.cat(y_hat_slices, dim=1)
                )

                params_anchor = self.model.entropy_parameters_anchor[idx](
                    torch.cat([global_inter_ctx, channel_ctx, hyper_params], dim=1)
                )
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)

                means_anchor = ckbd_anchor(means_anchor)
                slice_anchor = torch.zeros_like(means_anchor)

                lrp_anchor = self.model.lrp_anchor[idx](
                    torch.cat(([hyper_means] + y_hat_slices + [slice_anchor]), dim=1)
                )
                slice_anchor = slice_anchor + ckbd_anchor(lrp_anchor)

                global_intra_ctx = self.model.global_intra_context[idx](
                    y_hat_slices[-1],
                    slice_anchor,
                )
                local_ctx = self.model.local_context[idx](slice_anchor)

                params_nonanchor = self.model.entropy_parameters_nonanchor[idx](
                    torch.cat(
                        [
                            local_ctx,
                            global_intra_ctx,
                            global_inter_ctx,
                            channel_ctx,
                            hyper_params,
                        ],
                        dim=1,
                    )
                )
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)

                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                slice_nonanchor = torch.zeros_like(means_nonanchor)

                y_hat_slice = slice_anchor + slice_nonanchor

                lrp_nonanchor = self.model.lrp_nonanchor[idx](
                    torch.cat(([hyper_means] + y_hat_slices + [y_hat_slice]), dim=1)
                )
                y_hat_slice = y_hat_slice + ckbd_nonanchor(lrp_nonanchor)

                y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.model.g_s(y_hat)
        return x_hat

class MLICPlusPlusImageCodec(ImageCodecIface):
    def __init__(self, quality, ckpt_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality = quality
        self.ckpt_path = ckpt_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = model_config()
        
        self.model = MLICPlusPlus(config=config).cuda().eval()
        checkpoint = torch.load(self.ckpt_path, map_location='cpu')
        # support checkpoints saved with DataParallel/DistributedDataParallel ("module." prefix)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            raw_state = checkpoint['state_dict']
        else:
            raw_state = checkpoint
        fixed_state = {}
        for k, v in raw_state.items():
            new_k = k
            if k.startswith("module."):
                new_k = k[len("module."):]
            fixed_state[new_k] = v
        self.model.load_state_dict(fixed_state)
        epoch = checkpoint["epoch"]
        self.save_dir = os.path.join('./experiments', 'codestream', '%02d' % (epoch + 1))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):
        x = x.to(self.device, dtype=torch.float)
        recon_list = []
        bpp_vals = []
        for i in range(x.shape[0]):
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
            img = x[i:i+1]
            bpp, enc_time = compress_one_image(model=self.model, x=img, stream_path=self.save_dir, H=img.shape[2], W=img.shape[3], img_name=f"{i}_r{rank}")
            xhat, dec_time = decompress_one_image(model=self.model, stream_path=self.save_dir, img_name=f"{i}_r{rank}")
            xhat = xhat.clamp(0, 1)
            if xhat.dim() == 3:  
                xhat = xhat.unsqueeze(0)

            # ensure reconstruction on same device as input batch
            out_img = xhat.to(x.device)
            recon_list.append(out_img)
            bpp_vals.append(bpp)

        # concat per-image reconstructions into a full-batch tensor
        xhat = torch.cat(recon_list, dim=0)  # shape: (batch, C, H, W)
        bpp = torch.tensor(bpp_vals, dtype=torch.float32, device=x.device)  # shape: (batch,)
        return xhat, bpp
    
    def fake_input(self, image_size=256, batch_size=1, device=None):
        if device is None:
            device = self.device
        return torch.rand(batch_size, 3, image_size, image_size, device=device)

    def encode_params_m(self):
        modules = [
            self.model.g_a,
            self.model.h_a,
            self.model.entropy_bottleneck,
            self.model.h_s,
            self.model.local_context,
            self.model.channel_context,
            self.model.global_inter_context,
            self.model.global_intra_context,
            self.model.entropy_parameters_anchor,
            self.model.entropy_parameters_nonanchor,
            self.model.lrp_anchor,
            self.model.lrp_nonanchor,
        ]
        return sum(params_m(m) for m in modules if m is not None)

    def decode_params_m(self):
        modules = [
            self.model.entropy_bottleneck,
            self.model.h_s,
            self.model.local_context,
            self.model.channel_context,
            self.model.global_inter_context,
            self.model.global_intra_context,
            self.model.entropy_parameters_anchor,
            self.model.entropy_parameters_nonanchor,
            self.model.lrp_anchor,
            self.model.lrp_nonanchor,
            self.model.g_s,
        ]
        return sum(params_m(m) for m in modules if m is not None)

    @torch.no_grad()
    def encode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, dtype=torch.float)

        return time_ms(
            lambda: self.model.compress(x),
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

    @torch.no_grad()
    def decode_time_ms(self, x, warmup=5, repeat=20):
        x = x.to(self.device, dtype=torch.float)

        out = self.model.compress(x)
        strings = out["strings"]
        shape = out["shape"]

        return time_ms(
            lambda: self.model.decompress(strings, shape),
            self.device,
            warmup=warmup,
            repeat=repeat,
        )

    @torch.no_grad()
    def encode_gflops(self, x):
        x = x.to(self.device, dtype=torch.float)

        wrapper = MLICPPEncodeFLOPsWrapper(self.model).to(self.device).eval()
        return gflops(wrapper, x)

    @torch.no_grad()
    def decode_gflops(self, x):
        x = x.to(self.device, dtype=torch.float)

        y = self.model.g_a(x)
        z = self.model.h_a(y)
        z_hat = torch.round(z)

        wrapper = MLICPPDecodeFLOPsWrapper(self.model).to(self.device).eval()
        return gflops(wrapper, z_hat)