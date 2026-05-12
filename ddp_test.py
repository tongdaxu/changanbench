import torch.distributed as dist
import os
import torch
from omegaconf import OmegaConf
from cab.utils import instantiate_from_config
import argparse
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="test changan bench with DDP")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument('--log_dir', type=str, default="./logs")
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--local-rank', type=int, default=None)
    parser.add_argument('--zero-mean', type=bool, default=False, help='Only needed for visualization')
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
        drop_last=True,
    )
    
    return data_loader


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    
    args = parse_args()
    
    # Initialize DDP
    dist.init_process_group(
        backend='nccl',
        init_method="env://",
    )
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    datasets_name = config['datasets']
    codecs_name = config['codecs']
    metrics_name = config['metrics']
    
    # Instantiate components
    datasets = []
    codecs = []
    metrics = []
    
    for dataset_name in datasets_name:
        dataset = instantiate_from_config(config[dataset_name])
        datasets.append((dataset_name, dataset))
    
    for codec_name in codecs_name:
        codec = instantiate_from_config(config[codec_name])
        codec = codec.cuda()
        codec.eval()
        codecs.append((codec_name, codec))
    
    for metric_name in metrics_name:
        metric = instantiate_from_config(config[metric_name])
        metrics.append((metric_name, metric))
    
    # Evaluation loop
    for cname, codec in codecs:
        for dname, dataset in datasets:
            cache_file_name = os.path.join(args.cache_dir, cname, dname)
            os.makedirs(cache_file_name, exist_ok=True)
            
            # Initialize result accumulation lists for each rank
            metric_results = {mname: [[] for _ in range(world_size)] for mname, _ in metrics}
            bpp_results = [[] for _ in range(world_size)]
            total_num = 0
            
            # Create dataloader - each rank gets different data subset via DistributedSampler
            data_loader = create_data_loader(
                dataset, args.batch_size, args.num_workers
            )
            
            if rank == 0:
                data_loader = tqdm(data_loader, desc=f"Processing {cname}/{dname}")
            
            # Process batches - all ranks participate
            with torch.no_grad():
                saved = 0
                max_save = 3
                for batch in data_loader:
                    img = batch["img"].cuda()
                    rec, bpp = codec(img)
                    if cname =='hific_q0' or cname == 'hific_q2':
                        img = (img + 1.) / 2
                    # if saved < max_save:
                    #     for i in range(min(img.shape[0], max_save - saved)):
                    #         if args.zero_mean:
                    #             x_vis = img [i] * 0.5 + 0.5
                    #             x_recon_vis = rec[i] * 0.5 + 0.5
                    #         else:
                    #             x_vis = img[i]
                    #             x_recon_vis = rec[i]
                    #         vutils.save_image(x_recon_vis, os.path.join(cache_file_name, f'recon_{saved}.png'))
                    #         saved += 1
                            # if saved >= max_save:
                            #     assert 0

                    
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
                        out = metric(img, rec)
                        
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
                
                print(f"{'='*60}\n")
            
            dist.barrier()
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()