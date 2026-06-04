import os
import math
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.nn as nn

from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms

import diffusers
from diffusers.utils.import_utils import is_xformers_available

from my_utils.testing_utils import parse_args_testing
from color_fix import adain_color_fix_quant
from StableCodec import StableCodec
from PIL import Image
from compress_utils import *

import time


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor


def compress_one_image(net, bin_path, ori_h, ori_w, img_name, x):
    with torch.no_grad():
        output_dict = net.compress(x)
    shape = output_dict["shape"]
    if not os.path.exists(bin_path): os.makedirs(bin_path)
    output = os.path.join(bin_path, img_name)
    with Path(output).open("wb") as f:
        write_body(f, shape, output_dict["strings"])
    size = filesize(output)
    bpp = float(size) * 8 / (ori_h * ori_w)
    return bpp


def decompress_one_image(net, bin_path, ori_h, ori_w, img_name, prompt):
    output = os.path.join(bin_path, img_name)
    with Path(output).open("rb") as f:
        strings, shape = read_body(f)
    with torch.no_grad():
        out_img = net.decompress(strings, shape, prompt)
    out_img = out_img[:, :, 0 : ori_h, 0 : ori_w]
    out_img = (out_img * 0.5 + 0.5).float().cpu().detach()
    return out_img


def main(args):

    class EncoderWrapper(nn.Module):
        # 包装完整编码流程用于 FLOPs 分析（不包括熵编码）
        def __init__(self, aux_codec, vae_encoder, vae_scaling_factor):
            super().__init__()
            self.aux_codec = aux_codec
            self.vae_encoder = vae_encoder
            self.vae_scaling_factor = vae_scaling_factor
        
        def forward(self, x):
            # 辅助编码器
            latent2 = self.aux_codec((x + 1) / 2)
            # VAE 编码器
            lq_latent = self.vae_encoder(x) * self.vae_scaling_factor
            return lq_latent, latent2
        
    class DecoderWrapper(nn.Module):
        # 包装解码流程（UNet + scheduler.step + VAE.decode）用于 FLOPs 分析
        def __init__(self, unet, vae, sched, timesteps, pos_caption_enc, vae_scaling_factor):
            super().__init__()
            self.unet = unet
            self.vae = vae
            self.sched = sched
            self.timesteps = timesteps
            # pos_caption_enc should be precomputed tensor (encoder hidden states)
            self.register_buffer("pos_caption_enc", pos_caption_enc.clone())
            self.vae_scaling_factor = vae_scaling_factor

        def forward(self, lq_latent):
            # lq_latent: full-latent tensor; unet consumes it directly
            # expand pos_caption_enc to batch size if necessary
            b = lq_latent.shape[0]
            enc = self.pos_caption_enc
            if enc.shape[0] != b:
                enc = enc.repeat(b, 1, 1)
            model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=enc).sample
            x_denoised = self.sched.step(model_pred, self.timesteps, lq_latent[:, :4], return_dict=True).prev_sample
            output_image = self.vae.decode(x_denoised / self.vae_scaling_factor).sample
            return output_image

    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path

    if args.seed is not None:
        set_seed(args.seed)

    net = StableCodec(sd_path=sd_path, args=args)
    net.cuda().eval()
    net.codec.update(force=True)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # --- 新增：对第一张图片进行编码时间基准测试并尝试用 fvcore 统计 FLOPs ---
    with open(args.img_path, 'r') as f:
        images = [line.strip() for line in f if line.strip()]

    if len(images) > 0:
        first_img_path = images[0]
        print(f'\nBenchmarking encoding using first image: {first_img_path}')
        img_tensor = preprocess_image(first_img_path, transform).cuda().unsqueeze(0)
        ori_h, ori_w = img_tensor.shape[2:]
        pad_h = (math.ceil(ori_h / 256)) * 256 - ori_h
        pad_w = (math.ceil(ori_w / 256)) * 256 - ori_w
        img_padded = F.pad(img_tensor, pad=(0, pad_w, 0, pad_h), mode='reflect')

        # warm-up
        for _ in range(10):
            with torch.no_grad():
                _ = net.compress(img_padded)
        torch.cuda.synchronize()

        runs = 100
        times = []
        for i in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = net.compress(img_padded)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        mean_time = sum(times) / len(times)
        encode_std = np.std(times)
        encode_cv = encode_std / mean_time if mean_time != 0 else 0
        print(f'Encoding time over {runs} runs: mean = {mean_time:.6f} s, std = {encode_std:.6f} s, cv = {encode_cv:.6f}')

        try:
            from fvcore.nn import FlopCountAnalysis, flop_count_table
            
            print('\n[Complete Encoder Pipeline FLOPs Analysis]')
            encoder_wrapper = EncoderWrapper(
                net.aux_codec, 
                net.vae.encoder,
                net.vae.config.scaling_factor
            ).cuda()
            
            flops = FlopCountAnalysis(encoder_wrapper, img_padded)
            # print(flop_count_table(flops))
            total_flops = flops.total()
            print(f'Total Encoder Pipeline FLOPs(G): {total_flops/1e9:,}')
            
        except Exception as e:
            print(f'FLOPs analysis failed:  {e}')

        # --- 新增：对解码部分的参数、时间和 FLOPs 评测 ---
        try:
            # 先用一次 compress 得到可用于 decompress 的表示（strings/shape）
            with torch.no_grad():
                output_dict = net.compress(img_padded)
            strings = output_dict.get("strings", None)
            shape = output_dict.get("shape", None)
            # 解码参数统计（UNet + VAE decoder）
            dec_param_count = sum(p.numel() for p in net.unet.parameters()) + sum(p.numel() for p in net.vae.decoder.parameters())
            print(f'\n[Decoder Params] UNet + VAE.decoder parameter count(M) = {dec_param_count/1e6:,}')

            # 解码时间基准：warm-up + 多次测量
            if strings is not None and shape is not None:
                # warm-up
                for _ in range(10):
                    with torch.no_grad():
                        _ = net.decompress(strings, shape, [1])
                    torch.cuda.synchronize()

                runs = 100
                dec_times = []
                for i in range(runs):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        _ = net.decompress(strings, shape, [1])
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    dec_times.append(t1 - t0)
                dec_mean = sum(dec_times) / len(dec_times)
                dec_std = np.std(dec_times)
                dec_cv = dec_std / dec_mean if dec_mean != 0 else 0
                print(f'Decoding time over {runs} runs: mean = {dec_mean:.6f} s, std = {dec_std:.6f} s, cv = {dec_cv:.6f}')
            else:
                print('Skipping decompress timing because compress did not return strings/shape.')

            # 尝试用 fvcore 测量解码 FLOPs（构建 decoder wrapper 并用 vae.encode 的结果作为输入）
            try:
                from fvcore.nn import FlopCountAnalysis, flop_count_table
                with torch.no_grad():
                    # # 使用真实的 lq_latent 作为解码器输入（近似解码时使用的 latent）
                    # lq_latent = net.vae.encode(img_padded).latent_dist.mode() * net.vae.config.scaling_factor
                    
                    # 使用 codec.decompress 得到真实的 lq_latent_hat（通道数与 UNet 输入一致）
                    lq_latent_hat, _ = net.codec.decompress(strings, shape)
                    # 确保在同一设备
                    lq_latent = lq_latent_hat.cuda()

                    # pos_caption_enc 取 net.pos_caption_enc（已在 StableCodec 中创建）
                    pos_enc = net.pos_caption_enc.detach().cpu()
                    decoder_wrapper = DecoderWrapper(net.unet, net.vae, net.sched, net.timesteps, pos_enc, net.vae.config.scaling_factor).cuda()
                    flops_dec = FlopCountAnalysis(decoder_wrapper, (lq_latent,))
                    total_flops_dec = flops_dec.total()
                    print(f'Total Decoder Pipeline FLOPs(G): {total_flops_dec/1e9:,}')
            except Exception as e2:
                print(f'Decoder FLOPs analysis failed: {e2}')

        except Exception as e:
            print(f'Decoder benchmark failed: {e}')

    bpp = []
    pos_tag_prompt = [1]
    with open(args.img_path, 'r') as f:
        images = [line.strip() for line in f if line.strip()]
    print(f'\nFind {str(len(images))} images in {args.img_path}\n')

    for img_path in images:

        print('[Processing]', img_path)
        (path, name) = os.path.split(img_path)
        fname, ext = os.path.splitext(name)
        outf = os.path.join(args.rec_path, fname+'.png')

        img = preprocess_image(img_path, transform).cuda().unsqueeze(0)
        ori_h, ori_w = img.shape[2:]

        pad_h = (math.ceil(ori_h / 256)) * 256 - ori_h
        pad_w = (math.ceil(ori_w / 256)) * 256 - ori_w
        img_padded = F.pad(img, pad=(0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            try:
                rate = compress_one_image(net, args.bin_path, ori_h, ori_w, fname, img_padded)
                out_img = decompress_one_image(net, args.bin_path, ori_h, ori_w, fname, pos_tag_prompt)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(str(name))
                    print("CUDA out of memory. Continuing to next image.")
                    torch.cuda.empty_cache() 
                    continue
                else:
                    raise

        output_pil = transforms.ToPILImage()(out_img[0].clamp(0.0, 1.0))

        bpp.append(rate)
        print('[BPP]', rate)

        if args.color_fix:
            img = (img * 0.5 + 0.5).float().cpu().detach()
            im_lr_resize = transforms.ToPILImage()(img[0].clamp(0.0, 1.0))
            output_pil = adain_color_fix_quant(output_pil, im_lr_resize, 16)

        output_pil.save(outf)

    print('\n[Average BPP]', np.mean(bpp))
    

if __name__ == "__main__":
    args = parse_args_testing()
    if not os.path.exists(args.rec_path): os.makedirs(args.rec_path)
    if not os.path.exists(args.bin_path): os.makedirs(args.bin_path)
    main(args)
