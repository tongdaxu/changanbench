"""Continuous evaluation script for FlowMo.

Code is (partially) from https://github.com/TencentARC/SEED-Voken.Thanks! 
"""

import functools
import os
import time

import lpips
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from PIL import Image
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from pytorch_msssim import ms_ssim
from DISTS_pytorch import DISTS

from flowmo import train_utils
from flowmo.baselines import baselines
from flowmo.inception import InceptionV3

# fvcore for flop counting
from fvcore.nn import FlopCountAnalysis, flop_count_table

from flowmo.models import prepare_idxs  # 添加这个导入

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB": 
        x = x.convert("RGB")
    return x


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J.Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   :  The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, (
        "Training and test mean vectors have different lengths"
    )
    assert sigma1.shape == sigma2.shape, (
        "Training and test covariances have different dimensions"
    )

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def get_subsampled_dataset(dataset, subsample_rate=1):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    indices = [
        idx
        for idx in np.arange(len(dataset))
        if (idx % (world_size * subsample_rate)) == rank
    ]
    return torch.utils.data.Subset(dataset, indices)


def _gather_concat_np(pred_np):
    pred = torch.from_numpy(pred_np).cuda().contiguous()
    out = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
    dist.all_gather(out, pred)
    return torch.concatenate(out).cpu().numpy()


def _calculate_fid(pred_xs, pred_recs):
    pred_xs = np.concatenate(pred_xs, axis=0)
    pred_recs = np.concatenate(pred_recs, axis=0)

    pred_xs = _gather_concat_np(pred_xs)
    pred_recs = _gather_concat_np(pred_recs)
    print(f"Compute FID from {len(pred_xs), len(pred_recs)} samples for pred,recs")

    mu_x = np.mean(pred_xs, axis=0)
    sigma_x = np.cov(pred_xs, rowvar=False)

    mu_rec = np.mean(pred_recs, axis=0)
    sigma_rec = np.cov(pred_recs, rowvar=False)

    fid_value = calculate_frechet_distance(mu_x, sigma_x, mu_rec, sigma_rec)
    return fid_value


def _reduce_scalar(scalar):
    scalar = torch.from_numpy(np.array([scalar])).cuda()
    dist.all_reduce(scalar, op=dist.ReduceOp.AVG)
    return scalar.cpu().numpy().item()

def profile_complexity(model, example_img, dtype, repeat_times=10, warmup_times=3):
    """
    Profile encode/decode complexity on a single example image.    
    Only outputs:  timing (ms), FLOPs (G), and decoder parameters (M).
    
    Args:
        model: FlowMo model
        example_img: Single example image tensor (1, C, H, W) on CUDA
        dtype: Data type for inference (e.g., torch.bfloat16)
        repeat_times: Number of timing repetitions for averaging
        warmup_times: Number of warmup runs before timing
        
    Returns:
        dict: Dictionary containing profiling results
    """
    import torch
    import einops
    
    profiling_results = {}
    
    model.eval()
    torch.cuda.synchronize()
    
    # Store original dtype and move model to target dtype for timing
    original_dtype = next(model.parameters()).dtype
    model = model.to(dtype)
    example_in = example_img.to(dtype)

    b, c, h, w = example_in.shape
    
    # ============================================================================
    # 临时禁用 torch.compile 以避免 dtype 问题
    # ============================================================================
    import torch._dynamo
    torch._dynamo.reset()  # 重置编译缓存
    
    # 创建未编译的包装函数
    def encode_no_compile(img):
        b, c, h, w = img.shape
        img_idxs, txt_idxs = prepare_idxs(img, model.code_length, model.patch_size)
        txt = torch.zeros(
            (b, model.code_length, model.encoder_context_dim), 
            device=img.device, dtype=img.dtype
        )
        _, code, aux = model.encoder(img, img_idxs, txt, txt_idxs, timesteps=None)
        return code, aux
    
    def decode_no_compile(img, code, timesteps):
        b, c, h, w = img.shape
        img_idxs, txt_idxs = prepare_idxs(img, model.code_length, model.patch_size)
        pred, _, decode_aux = model.decoder(img, img_idxs, code, txt_idxs, timesteps=timesteps)
        return pred, decode_aux
    
    def quantize_no_compile(code):
        """直接调用 _quantize 的逻辑，避免编译"""
        b, t, f = code.shape
        indices = None
        if model.config.model.quantization_type == "noop":
            quantized = code
            quantizer_loss = torch.tensor(0.0, device=code.device)
        elif model.config.model.quantization_type == "kl": 
            mean, logvar = _get_diagonal_gaussian(
                einops.rearrange(code, "b t f -> b (f t)")
            )
            code = einops.rearrange(
                _sample_diagonal_gaussian(mean, logvar),
                "b (f t) -> b t f",
                f=f // 2,
                t=t,
            )
            quantizer_loss = _kl_diagonal_gaussian(mean, logvar)
        elif model.config.model.quantization_type == "lfq":
            assert f % model.config.model.codebook_size_for_entropy == 0, f
            code_reshaped = einops.rearrange(
                code,
                "b t (fg fh) -> b fg (t fh)",
                fg=model.config.model.codebook_size_for_entropy,
            )
            (quantized, entropy_aux_loss, indices), breakdown = model.quantizer(
                code_reshaped, return_loss_breakdown=True
            )
            quantized = einops.rearrange(quantized, "b fg (t fh) -> b t (fg fh)", t=t)
            quantizer_loss = (
                entropy_aux_loss * model.config.model.entropy_loss_weight
                + breakdown.commitment * model.config.model.commit_loss_weight
            )
            code = quantized
        else: 
            raise NotImplementedError
        return code, indices, quantizer_loss
    
    # ============================================================================
    # 1.Encoder Timing & FLOPs
    # ============================================================================
    print("\nAnalyzing Encoder...")
    
    # Encoder timing
    enc_out = None
    try:
        # Warmup
        for _ in range(warmup_times):
            with torch.no_grad():
                code, _ = encode_no_compile(example_in)
        torch.cuda.synchronize()
        
        # Timing
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(repeat_times):
            with torch.no_grad():
                enc_out, _ = encode_no_compile(example_in)
        torch.cuda.synchronize()
        avg_enc_time_ms = ((time.time() - t0) / repeat_times) * 1000
        profiling_results['encode_time_ms'] = avg_enc_time_ms
        
    except Exception as e: 
        print(f"  Encode timing failed: {e}")
        import traceback
        traceback.print_exc()
        profiling_results['encode_time_ms'] = None
    
    # Encoder FLOPs
    try:
        model_fp32 = model.to(torch.float32)
        example_fp32 = example_in.to(torch.float32).contiguous()
        
        img_idxs_fp32, txt_idxs_fp32 = prepare_idxs(
            example_fp32, model_fp32.code_length, model_fp32.patch_size
        )
        txt_enc_fp32 = torch.zeros(
            (b, model_fp32.code_length, model_fp32.encoder_context_dim), 
            device=example_fp32.device,
            dtype=torch.float32
        )
        
        flops_enc = FlopCountAnalysis(
            model_fp32.encoder, 
            (example_fp32, img_idxs_fp32, txt_enc_fp32, txt_idxs_fp32, None)
        )
        profiling_results['encode_gflops'] = flops_enc.total() / 1e9
        
        # Restore dtype
        model = model.to(dtype)
        
    except Exception as e: 
        print(f"  Encoder FLOPs analysis failed:  {e}")
        profiling_results['encode_gflops'] = None
    
    # ============================================================================
    # 2.Decoder Timing & FLOPs (考虑 rf_sample 和 CFG)
    # ============================================================================
    print("Analyzing Decoder...")
    
    if enc_out is None:
        print("  Skipping decoder analysis (no encoder output)")
        profiling_results['decode_time_ms'] = None
        profiling_results['decode_gflops'] = None
        profiling_results['decode_total_time_ms'] = None
        profiling_results['decode_total_gflops'] = None
    else:
        # Quantize code
        try:
            code, _, _ = quantize_no_compile(enc_out)
            mask = torch.ones_like(code[..., : 1])
            code_with_mask = torch.cat([code, mask], dim=-1)
        except Exception as e:
            print(f"  Code quantization failed: {e}")
            import traceback
            traceback.print_exc()
            code_with_mask = None
        
        if code_with_mask is not None:
            # 获取采样配置
            try:
                sample_steps = model.config.eval.sampling.sample_steps
                cfg = model.config.eval.sampling.cfg
            except: 
                sample_steps = 25  # 默认值
                cfg = 1.0
            
            # 判断是否使用 CFG
            use_cfg = (cfg != 1.0)
            decode_calls_per_step = 2 if use_cfg else 1
            
            # Decoder 单步 timing
            try:
                z = torch.randn((b, 3, h, w), device=example_in.device, dtype=dtype)
                timesteps = torch.ones((b,), device=example_in.device, dtype=dtype) * 0.5
                
                # Warmup
                for _ in range(warmup_times):
                    with torch.no_grad():
                        _ = decode_no_compile(z, code_with_mask, timesteps)
                        if use_cfg:
                            # 模拟 CFG 的第二次调用
                            null_code = code_with_mask * 0.0
                            _ = decode_no_compile(z, null_code, timesteps)
                torch.cuda.synchronize()
                
                # Timing (单次 decode 调用)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(repeat_times):
                    with torch.no_grad():
                        _ = decode_no_compile(z, code_with_mask, timesteps)
                torch.cuda.synchronize()
                single_decode_time_ms = ((time.time() - t0) / repeat_times) * 1000
                
                # 计算每步和总时间
                per_step_time_ms = single_decode_time_ms * decode_calls_per_step
                total_time_ms = per_step_time_ms * sample_steps
                
                profiling_results['decode_time_ms'] = single_decode_time_ms
                profiling_results['decode_per_step_time_ms'] = per_step_time_ms
                profiling_results['decode_total_time_ms'] = total_time_ms
                profiling_results['sample_steps'] = sample_steps
                profiling_results['use_cfg'] = use_cfg
                profiling_results['cfg_value'] = cfg
                
            except Exception as e:
                print(f"  Decode timing failed: {e}")
                import traceback
                traceback.print_exc()
                profiling_results['decode_time_ms'] = None
                profiling_results['decode_per_step_time_ms'] = None
                profiling_results['decode_total_time_ms'] = None
            
            # Decoder FLOPs (单次调用)
            try:
                model_fp32 = model.to(torch.float32)
                z_fp32 = torch.randn((b, 3, h, w), device=example_in.device, dtype=torch.float32)
                code_fp32 = code_with_mask.to(torch.float32).contiguous()
                timesteps_fp32 = torch.ones((b,), device=z_fp32.device, dtype=torch.float32) * 0.5
                
                img_idxs_fp32, txt_idxs_fp32 = prepare_idxs(
                    z_fp32, model_fp32.code_length, model_fp32.patch_size
                )
                
                flops_dec = FlopCountAnalysis(
                    model_fp32.decoder,
                    (z_fp32, img_idxs_fp32, code_fp32, txt_idxs_fp32, timesteps_fp32)
                )
                single_decode_gflops = flops_dec.total() / 1e9
                per_step_gflops = single_decode_gflops * decode_calls_per_step
                total_gflops = per_step_gflops * sample_steps
                
                profiling_results['decode_gflops'] = single_decode_gflops
                profiling_results['decode_per_step_gflops'] = per_step_gflops
                profiling_results['decode_total_gflops'] = total_gflops
                
                # Restore dtype
                model = model.to(dtype)
                
            except Exception as e: 
                print(f"  Decoder FLOPs analysis failed: {e}")
                profiling_results['decode_gflops'] = None
                profiling_results['decode_per_step_gflops'] = None
                profiling_results['decode_total_gflops'] = None
        else:
            profiling_results['decode_time_ms'] = None
            profiling_results['decode_per_step_time_ms'] = None
            profiling_results['decode_total_time_ms'] = None
            profiling_results['decode_gflops'] = None
            profiling_results['decode_per_step_gflops'] = None
            profiling_results['decode_total_gflops'] = None
    
    # ============================================================================
    # 3.Decoder Parameters (in Millions)
    # ============================================================================
    try:
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        profiling_results['decoder_params_m'] = decoder_params / 1e6
    except Exception as e:
        print(f"  Parameter counting failed: {e}")
        profiling_results['decoder_params_m'] = None
    
    # Restore original dtype
    model = model.to(original_dtype)
    
    # ============================================================================
    # Print Results
    # ============================================================================
    print("\n" + "="*70)
    print("COMPLEXITY PROFILING RESULTS")
    print("="*70)
    
    # Encoder
    if profiling_results.get('encode_time_ms') is not None:
        print(f"Encode Time:                     {profiling_results['encode_time_ms']:>10.2f} ms")
    else:
        print(f"Encode Time:                   {'N/A':>10}")
    
    # Decoder
    if profiling_results.get('decode_time_ms') is not None:
        print(f"Decode Time (single call):      {profiling_results['decode_time_ms']:>10.2f} ms")
    else:
        print(f"Decode Time (single call):     {'N/A':>10}")
    
    if profiling_results.get('decode_per_step_time_ms') is not None:
        cfg_note = f" (CFG={profiling_results.get('cfg_value', 'N/A')})" if profiling_results.get('use_cfg') else ""
        print(f"Decode Time (per step):         {profiling_results['decode_per_step_time_ms']:>10.2f} ms{cfg_note}")
    else:
        print(f"Decode Time (per step):        {'N/A':>10}")
    
    if profiling_results.get('decode_total_time_ms') is not None:
        steps = profiling_results.get('sample_steps', 'N/A')
        print(f"Decode Time (total):            {profiling_results['decode_total_time_ms']:>10.2f} ms ({steps} steps)")
    else:
        print(f"Decode Time (total):           {'N/A':>10}")
    
    print("-"*70)
    
    # FLOPs
    if profiling_results.get('encode_gflops') is not None:
        print(f"Encode FLOPs:                   {profiling_results['encode_gflops']:>10.2f} G")
    else:
        print(f"Encode FLOPs:                  {'N/A':>10}")
    
    if profiling_results.get('decode_gflops') is not None:
        print(f"Decode FLOPs (single call):     {profiling_results['decode_gflops']:>10.2f} G")
    else:
        print(f"Decode FLOPs (single call):    {'N/A':>10}")
    
    if profiling_results.get('decode_per_step_gflops') is not None:
        cfg_note = f" (CFG={profiling_results.get('cfg_value', 'N/A')})" if profiling_results.get('use_cfg') else ""
        print(f"Decode FLOPs (per step):        {profiling_results['decode_per_step_gflops']:>10.2f} G{cfg_note}")
    else:
        print(f"Decode FLOPs (per step):       {'N/A':>10}")
    
    if profiling_results.get('decode_total_gflops') is not None:
        steps = profiling_results.get('sample_steps', 'N/A')
        print(f"Decode FLOPs (total):           {profiling_results['decode_total_gflops']:>10.2f} G ({steps} steps)")
    else:
        print(f"Decode FLOPs (total):          {'N/A':>10}")
    
    print("-"*70)
    
    # Parameters
    if profiling_results.get('decoder_params_m') is not None:
        print(f"Decoder Params:                 {profiling_results['decoder_params_m']:>10.2f} M")
    else:
        print(f"Decoder Params:                {'N/A':>10}")
    
    print("="*70 + "\n")
    
    return profiling_results

def eval_imagenet(model, config):
    config = train_utils.restore_config(config)
    model = model.eval()

    bs = config.data.eval_batch_size
    config.data.batch_size = bs

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    imagenet_dataset = torchvision.datasets.ImageFolder('/NEW_EDS/JJ_Group/lisq/bsq-vit/PhotoCD_PCD0992', transform=val_transform)
    
    # Check if only profiling complexity (single image test)
    if config.eval.get('profile_only', False):
        print("=" * 80)
        print("COMPLEXITY PROFILING MODE - Testing first image only")
        print("=" * 80)
        
        # Get first example
        try:
            example_img, _ = imagenet_dataset[0]
            example_img = example_img.unsqueeze(0).cuda()
            
            dtype = torch.bfloat16 if train_utils.bfloat16_is_available() else torch.float32
            
            # Run profiling
            profiling_results = profile_complexity(
                model=model,
                example_img=example_img,
                dtype=dtype,
                repeat_times=config.eval.get('profile_repeat_times', 10),
                warmup_times=config.eval.get('profile_warmup_times', 3)
            )
            
            print("=" * 80)
            print("PROFILING COMPLETED - Exiting without computing metrics")
            print("=" * 80)
            
            # Return empty results to signal early exit
            return None, profiling_results
            
        except Exception as e:
            print(f"Profiling failed: {e}")
            return None, {"error": str(e)}
    
    # ---- Normal evaluation mode continues below ----
    
    # FID score related
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx]).cuda()
    inception_model.eval()

    pred_xs = []
    pred_recs = []

    loss_fn_alex = lpips.LPIPS(net="alex").cuda()  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net="vgg").to(
        "cuda"
    )  # closer to "traditional" perceptual loss, when used for optimization
    lpips_alex = 0.0
    lpips_vgg = 0.0
    ssim_value = 0.0
    psnr_value = 0.0
    pairwise_inception_value = 0.0
    dists_model = DISTS().cuda()
    ms_ssim_value = 0.0
    dists_value = 0.0

    num_images = 0
    num_iter = 0

    num_examples = len(imagenet_dataset)
    print("Total number of examples:  ", num_examples)
    global_batch_size = bs * dist.get_world_size() * config.eval.subsample_rate
    assert num_examples % global_batch_size == 0, (
        num_examples,
        bs,
        dist.get_world_size(),
        config.eval.subsample_rate,
    )

    imagenet_dataset = get_subsampled_dataset(imagenet_dataset, subsample_rate=config.eval.subsample_rate)
    num_examples = len(imagenet_dataset)
    print("Total number of examples for this task:", num_examples)

    assert num_examples % bs == 0, (num_examples, bs)

    eval_batches = num_examples // bs

    # rebuild dataloader after subsampling
    dataloader = DataLoader(
        imagenet_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    assert len(dataloader) == eval_batches, (len(dataloader), eval_batches)

    if config.eval.eval_baseline == "sdxl":
        _, reconstruct_fn = baselines.load_sdxl()
    elif "titok" in config.eval.eval_baseline:
        _, reconstruct_fn = baselines.load_titok(config.eval.eval_baseline)
    elif config.eval.eval_baseline in ["llamagen16", "llamagen32"]: 
        _, reconstruct_fn = baselines.load_llamagen(config.eval.eval_baseline)
    elif config.eval.eval_baseline in ["cosmos_8x8", "cosmos_16x16"]:
        _, reconstruct_fn = baselines.load_cosmos(config.eval.eval_baseline)
    elif config.eval.eval_baseline == "flux":
        _, reconstruct_fn = baselines.load_flux()
    elif config.eval.eval_baseline == "magvitv2_tokenizer":
        _, reconstruct_fn = baselines.load_magvitv2_tokenizer_in16x16()
    elif config.eval.eval_baseline == "": 
        # run normal evaluation for FlowMo
        dtype = torch.bfloat16 if train_utils.bfloat16_is_available() else torch.float32
        print(f"Using dtype: {dtype}")
        reconstruct_fn = functools.partial(model.reconstruct, dtype=dtype)
    else:
        raise NotImplementedError

    generated_batches = []

    with torch.no_grad():
        for idx, batch in tqdm(
            enumerate(
                dataloader,
            ),
            total=len(dataloader),
        ):
            print("processing batch idx:  ", idx)
            images, _ = batch  # ImageFolder returns (image, label)
            images = images.cuda()
            num_images += images.shape[0]

            reconstructed_images = reconstruct_fn(images)
            reconstructed_images = reconstructed_images.clamp(-1, 1)

            images = (images + 1) / 2
            reconstructed_images = (reconstructed_images + 1) / 2

            generated_batches.append(
                np.clip(
                    reconstructed_images.permute((0, 2, 3, 1)).cpu().numpy() * 255,
                    0,
                    255,
                ).astype(np.uint8)
            )

            if config.eval.reconstruction: 
                # calculate lpips
                lpips_alex += loss_fn_alex(
                    images * 2 - 1, reconstructed_images * 2 - 1
                ).sum()
                lpips_vgg += loss_fn_vgg(
                    images * 2 - 1, reconstructed_images * 2 - 1
                ).sum()

                pred_x = inception_model(images)[0]
                pred_x = pred_x.squeeze(3).squeeze(2).cpu().numpy()
                pred_rec = inception_model(reconstructed_images)[0]
                pred_rec = pred_rec.squeeze(3).squeeze(2).cpu().numpy()

                pred_xs.append(pred_x)
                pred_recs.append(pred_rec)

                # calculate PSNR and SSIM
                rgb_restored = (
                    (reconstructed_images * 255.0)
                    .permute(0, 2, 3, 1)
                    .to("cpu", dtype=torch.uint8)
                    .numpy()
                )
                rgb_gt = (
                    (images * 255.0)
                    .permute(0, 2, 3, 1)
                    .to("cpu", dtype=torch.uint8)
                    .numpy()
                )
                rgb_restored = rgb_restored.astype(np.float32) / 255.0
                rgb_gt = rgb_gt.astype(np.float32) / 255.0
                ssim_temp = 0
                psnr_temp = 0
                B, _, _, _ = rgb_restored.shape
                for i in range(B):
                    rgb_restored_s, rgb_gt_s = rgb_restored[i], rgb_gt[i]
                    ssim_temp += ssim_loss(
                        rgb_restored_s,
                        rgb_gt_s,
                        data_range=1.0,
                        channel_axis=-1,
                    )
                    psnr_temp += psnr_loss(rgb_gt_s, rgb_restored_s)
                    # MS-SSIM expects (N,C,H,W), values in [0,1]
                    ms_ssim_value += ms_ssim(
                        reconstructed_images[i].unsqueeze(0),
                        images[i].unsqueeze(0),
                        data_range=1.0,
                    ).item()
                    # DISTS expects (N,C,H,W), values in [0,1]
                    dists_value += dists_model(
                        reconstructed_images[i].unsqueeze(0),
                        images[i].unsqueeze(0),
                    ).item()
                ssim_value += ssim_temp / B
                psnr_value += psnr_temp / B

                pairwise_inception_value += ((pred_x - pred_rec) ** 2).sum().item() / B

            num_iter += 1

            # Need this to prevent replicas from getting too out of sync w each other
            dist.barrier()

    generated_batches = np.concatenate(generated_batches)
    generated_batches = _gather_concat_np(generated_batches)
    print(generated_batches.shape)

    fid_value = _calculate_fid(pred_xs, pred_recs)

    if config.eval.reconstruction:
        lpips_alex_value = _reduce_scalar((lpips_alex / num_images).item())
        lpips_vgg_value = _reduce_scalar((lpips_vgg / num_images).item())
        ssim_value = _reduce_scalar(ssim_value / num_iter)
        psnr_value = _reduce_scalar(psnr_value / num_iter)
        pairwise_inception_value = _reduce_scalar(pairwise_inception_value / num_iter)
        ms_ssim_value = _reduce_scalar(ms_ssim_value / (num_iter * bs))
        dists_value = _reduce_scalar(dists_value / (num_iter * bs))

        reconstruction_metrics = {
            "LPIPS_ALEX:  ": lpips_alex_value,
            "LPIPS_VGG: ": lpips_vgg_value,
            "PSNR: ": psnr_value,
            "SSIM: ": ssim_value,
            "MS-SSIM: ": ms_ssim_value,
            "DISTS: ": dists_value,
            "PAIRWISE_INCEPTION":  pairwise_inception_value,
        }

    metrics = {"FID: ": fid_value, **reconstruction_metrics}
    return generated_batches, metrics


def _safe_load(checkpoint_path):
    while True:
        try:
            return torch.load(checkpoint_path)
        except Exception as e:
            print(e)
            time.sleep(30)


def main(args, config):
    config = train_utils.restore_config(config)

    torch.multiprocessing.set_start_method("forkserver", force=True)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    os.environ["TORCHELASTIC_ERROR_FILE"] = os.path.join(
        config.eval.eval_dir, "error_v0.json"
    )

    train_utils.soft_init()

    evaluated_checkpoints = set()
    eval_dir = config.eval.eval_dir
    assert eval_dir, eval_dir
    writer = SummaryWriter(eval_dir)

    seed = config.global_seed * dist.get_world_size() + dist.get_rank()
    torch.manual_seed(seed)

    print(OmegaConf.to_yaml(config))

    rank = dist.get_rank()
    print(dist.get_world_size())
    print(dist.get_rank())
    print(torch.cuda.device_count())

    device = rank % torch.cuda.device_count()
    print(device, torch.cuda.device_count())
    torch.cuda.set_device(device)

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
        eval_dir, f"torchinductor_ksarge_{args.experiment_name}_{str(rank)}"
    )
    main_tag = "eval/"

    while True:
        if config.eval.force_ckpt_path: 
            checkpoint_path = config.eval.force_ckpt_path
        else:
            checkpoint_path = train_utils.get_last_checkpoint(config, eval_dir)

        if checkpoint_path is None or checkpoint_path in evaluated_checkpoints: 
            print("No checkpoints to evaluate!  Sleeping...")
            time.sleep(30)
            continue

        print("Evaluating checkpoint!", checkpoint_path)
        model = train_utils.build_model(config)
        state_dict = train_utils.load_state_dict(checkpoint_path)
        total_steps = state_dict["total_steps"]
        model.load_state_dict(state_dict[config.eval.state_dict_key])

        images, metrics = eval_imagenet(model, config)
        
        # Check if profiling only mode
        if config.eval.get('profile_only', False):
            if dist.get_rank() == 0:
                print("Profiling completed.Results:")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
            torch.distributed.barrier()
            # Exit after profiling
            del state_dict, model
            return
        
        if dist.get_rank() == 0:
            print(metrics)
        torch.distributed.barrier()

        if dist.get_rank() == 0 and not config.eval.eval_baseline:
            for metric in metrics: 
                writer.add_scalar(
                    main_tag + metric,
                    metrics[metric],
                    global_step=total_steps,
                )
        evaluated_checkpoints.add(checkpoint_path)

        del state_dict, model, images
        if not config.eval.continuous:
            return


if __name__ == "__main__":
    try:
        args, config = train_utils.get_args_and_config()
        main(args, config)
    finally:
        torch.distributed.destroy_process_group()