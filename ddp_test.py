import torch.distributed as dist
import os
import torch
from omegaconf import OmegaConf
from cab.utils import instantiate_from_config
import argparse
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils
from cab.utils import inject_codec_zero_means
import cab.distributed as dist_utils

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="test changan bench with DDP")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument('--log_dir', type=str, default="./logs")
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--local-rank', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def create_data_loader(dataset, batch_size, num_workers):
    """Create distributed data loader"""
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        shuffle=False,
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=False,
    )
    
    return data_loader


def main():   
    args = parse_args()
    dist_utils.init_distributed_mode(args)
    dist_utils.random_seed(args.seed, dist_utils.get_rank())
    
    # Load config
    config = OmegaConf.load(args.config)
    
    datasets_name = config['datasets']
    codecs_name = config['codecs']
    metrics_name = config['metrics']

    codecs = []
    
    for cname in codecs_name:
        # create a resolved dict copy, inject codec-specific zero_mean flags, then back to OmegaConf
        cfg_dict = OmegaConf.to_container(config, resolve=True)
        cfg_dict = inject_codec_zero_means(cfg_dict, cname)
        cfg = OmegaConf.create(cfg_dict)

        # instantiate codec for this iteration
        codec = instantiate_from_config(cfg[cname]).cuda()
        codec.eval()

        dataset_zero_mean = cfg[cname].params.dataset_zero_mean
        metrics_zero_mean = cfg[cname].params.metrics_zero_mean

        # instantiate datasets (now with injected dataset_zero_mean)
        datasets = []
        for dname in datasets_name:
            dataset = instantiate_from_config(cfg[dname])
            datasets.append((dname, dataset))

        # instantiate metrics (now with injected metrics_zero_mean)
        metrics = []
        for mname in metrics_name:
            if mname != 'fid':
                metric = instantiate_from_config(cfg[mname])
                metrics.append((mname, metric))

        codecs.append((cname, codec, dataset_zero_mean, metrics_zero_mean, datasets, metrics))
    
    # Evaluation loop
    for cname, codec, dataset_zero_mean, metrics_zero_mean, datasets, metrics in codecs:
        for dname, dataset in datasets:
            cache_file_name = os.path.join(args.cache_dir, cname, dname)
            os.makedirs(cache_file_name, exist_ok=True)
            world_size = dist_utils.get_world_size()
            rank = dist_utils.get_rank()
            # Initialize result accumulation lists for each rank
            metric_results = {mname: [[] for _ in range(world_size)] for mname, _ in metrics}
            bpp_results = [[] for _ in range(world_size)]
            
            # FID-specific storage: collect activations per rank
            fid_activations_x = [[] for _ in range(world_size)]  # Original images
            fid_activations_xr = [[] for _ in range(world_size)]  # Reconstructed images

            total_num = 0
            
            # Create dataloader - each rank gets different data subset via DistributedSampler
            data_loader = create_data_loader(
                dataset, args.batch_size, args.num_workers
            )
            
            if rank == 0:
                data_loader = tqdm(data_loader, desc=f"Processing {cname}/{dname}")
            
            # Process batches - all ranks participate
            with torch.no_grad():
                max_save = 3
                for batch in data_loader:
                    img = batch["img"].cuda()
                    rec, bpp = codec(img)
                    if cname =='hific_q0' or cname == 'hific_q2':
                        img = (img + 1.) / 2
                    for i in range(min(img.shape[0], max_save)):
                        if dataset_zero_mean:
                            x_vis = img[i] * 0.5 + 0.5
                            x_recon_vis = rec[i] * 0.5 + 0.5
                        else:
                            x_vis = img[i]
                            x_recon_vis = rec[i]
                        vutils.save_image(x_recon_vis, os.path.join(cache_file_name, f'recon_{i}.png'))

                    
                    # Handle bpp
                    if isinstance(bpp, torch.Tensor):
                        bpp_tensor = bpp.cuda()
                    else:
                        bpp_tensor = torch.full((img.shape[0],), float(bpp), dtype=torch.float32, device=img.device)
                    
                    # Gather BPP from all ranks
                    gathered_bpp = [torch.zeros_like(bpp_tensor) for _ in range(world_size)]
                    dist.all_gather(gathered_bpp, bpp_tensor)
                    
                    for j in range(world_size):
                        bpp_results[j].append(gathered_bpp[j].detach().cpu())
                    
                    # Compute metrics for all ranks
                    for mname, metric in metrics:
                        out = metric(img, rec, zero_mean=metrics_zero_mean)
                        
                        # Handle tuple outputs (e.g., (ssim, msssim))
                        if isinstance(out, (tuple, list)):
                            if len(out) == 2:
                                scores0, scores1 = out
                                
                                # Gather first score
                                gathered0 = [torch.zeros_like(scores0) for _ in range(world_size)]
                                dist.all_gather(gathered0, scores0)
                                
                                # Gather second score
                                gathered1 = [torch.zeros_like(scores1) for _ in range(world_size)]
                                dist.all_gather(gathered1, scores1)
                                
                                # Accumulate for both metrics
                                for j in range(world_size):
                                    metric_results[mname][j].append(gathered0[j].detach().cpu())
                                
                                sec_name = 'msssim' if mname == 'ssim' else f"{mname}_2"
                                if sec_name not in metric_results:
                                    metric_results[sec_name] = [[] for _ in range(world_size)]
                                for j in range(world_size):
                                    metric_results[sec_name][j].append(gathered1[j].detach().cpu())
                            else:
                                # Handle cases with more than 2 outputs
                                for i, scores in enumerate(out):
                                    gathered = [torch.zeros_like(scores) for _ in range(world_size)]
                                    dist.all_gather(gathered, scores)
                                    
                                    key = f"{mname}_{i}"
                                    if key not in metric_results:
                                        metric_results[key] = [[] for _ in range(world_size)]
                                    for j in range(world_size):
                                        metric_results[key][j].append(gathered[j].detach().cpu())
                        else:
                            # Single output metric
                            gathered = [torch.zeros_like(out) for _ in range(world_size)]
                            dist.all_gather(gathered, out)
                            
                            for j in range(world_size):
                                metric_results[mname][j].append(gathered[j].detach().cpu())
                    
                    # FID: Collect activations (only if fid is in metrics)
                    if 'fid' in metrics_name:
                        from cab.evaluations.fid.get_fid import get_fid_batch
                        pred_x, pred_xr = get_fid_batch(img, rec, metrics_zero_mean=metrics_zero_mean)
                        
                        # Gather FID activations from all ranks
                        gathered_pred_x = [torch.zeros_like(pred_x) for _ in range(world_size)]
                        gathered_pred_xr = [torch.zeros_like(pred_xr) for _ in range(world_size)]
                        dist.all_gather(gathered_pred_x, pred_x)
                        dist.all_gather(gathered_pred_xr, pred_xr)
                        
                        for j in range(world_size):
                            fid_activations_x[j].append(gathered_pred_x[j].detach().cpu())
                            fid_activations_xr[j].append(gathered_pred_xr[j].detach().cpu())
                    
                    
                    total_num += world_size * img.shape[0]
            
            dist.barrier()
            
            # Aggregate and compute statistics on rank 0
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"Results: {cname} on {dname}")
                print(f"{'='*60}")
                
                # Process BPP
                for j in range(world_size):
                    bpp_results[j] = torch.cat(bpp_results[j], dim=0).numpy()
                bpp_flat = np.concatenate(bpp_results, axis=0)[:total_num]
                print(f"BPP: {np.mean(bpp_flat):.4f}") 
                

                # Process metrics
                for mname in metric_results.keys():
                    for j in range(world_size):
                        metric_results[mname][j] = torch.cat(metric_results[mname][j], dim=0).numpy()
                    
                    # Reorganize to restore original order
                    metric_reorg = []
                    for j in range(total_num):
                        metric_reorg.append(metric_results[mname][j % world_size][j // world_size])
                    metric_array = np.vstack(metric_reorg)
                    
                    print(f"{mname:12s}: {np.mean(metric_array):.4f} (±{np.std(metric_array):.4f})")
                
                # Process FID (if present)
                if 'fid' in metrics_name:
                    from cab.evaluations.fid.get_fid import compute_fid_from_activations
                    
                    # Reorganize activations
                    for j in range(world_size):
                        fid_activations_x[j] = torch.cat(fid_activations_x[j], dim=0).numpy()
                        fid_activations_xr[j] = torch.cat(fid_activations_xr[j], dim=0).numpy()
                    
                    all_pred_x_reorg = []
                    all_pred_xr_reorg = []
                    for j in range(total_num):
                        all_pred_x_reorg.append(fid_activations_x[j % world_size][j // world_size])
                        all_pred_xr_reorg.append(fid_activations_xr[j % world_size][j // world_size])
                    
                    all_pred_x = np.vstack(all_pred_x_reorg)
                    all_pred_xr = np.vstack(all_pred_xr_reorg)
                    
                    # Compute FID from aggregated activations
                    fid_score = compute_fid_from_activations(all_pred_x, all_pred_xr)
                    print(f"{'fid':12s}: {fid_score:.4f}")
                
                print(f"{'='*60}\n")
            
            dist.barrier()
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()