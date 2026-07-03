import torch.distributed as dist
import os
import torch
from omegaconf import OmegaConf
from cab.utils import instantiate_from_config
import argparse
import numpy as np
from tqdm import tqdm
from cab.utils import inject_codec_zero_means
import cab.distributed as dist_utils
from cab.evaluations.video_ddp import (
    adapt_metric_for_video,
    gather_dataset_metric,
    print_dataset_metric,
    save_reconstruction_preview,
)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="test changan bench with DDP")
    parser.add_argument("--mode", choices=["image", "video"], default="image")
    parser.add_argument("--image_codec_config", type=str, default="config/image_codecs",
                        help="codec config directory OR a single codec yaml file")
    parser.add_argument("--image_dataset_config", type=str, default="config/image_datasets.yaml")
    parser.add_argument("--image_metric_config", type=str, default="config/image_metrics.yaml")
    parser.add_argument("--config", type=str, default="", help="legacy video benchmark config yaml")
    parser.add_argument("--video_dataset_config", type=str, default="config/video_datasets")
    parser.add_argument("--video_codec_config", type=str, default="config/video_codecs",
                        help="video codec config directory OR a single codec yaml file")
    parser.add_argument("--video_metric_config", type=str, default="config/video_metrics.yaml")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument('--log_dir', type=str, default="./logs")
    parser.add_argument('--img_path', type=str, default=None)
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

def _ensure_1d_cpu(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().cpu()
    if t.dim() == 0:
        return t.unsqueeze(0)
    return t

def load_config_files(dataset_config_path, metric_config_path, codec_config_path, resolve=True):
    """Load and merge configs.
    
    dataset_config_path and codec_config_path can be either a single yaml file
    or a directory containing one yaml file per component.
    """
    try:
        if not os.path.exists(dataset_config_path):
            raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
        if not os.path.exists(metric_config_path):
            raise FileNotFoundError(f"Metric config not found: {metric_config_path}")
        if not os.path.exists(codec_config_path):
            raise FileNotFoundError(f"Codec config not found: {codec_config_path}")
        
        metric_cfg = OmegaConf.load(metric_config_path)

        def load_component_configs(config_path, component_name):
            component_cfgs = {}
            component_names = []

            if os.path.isfile(config_path):
                raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=resolve)

                if not isinstance(raw, dict):
                    raise ValueError(f"Invalid {component_name} config file: {config_path}")

                explicit_names = raw.get(f"{component_name}s")
                for key, value in raw.items():
                    if isinstance(value, dict) and "type" in value:
                        component_cfgs[key] = value

                if explicit_names is not None:
                    component_names = list(explicit_names)
                else:
                    component_names = list(component_cfgs)

                if not component_names:
                    raise ValueError(
                        f"No valid {component_name} configs found in {config_path}. "
                        f"Expected structure: {{{component_name}_name: {{type: ..., params: ...}}}}"
                    )

            elif os.path.isdir(config_path):
                for root, _, files in os.walk(config_path):
                    for fname in sorted(files):
                        if fname.endswith((".yaml", ".yml")):
                            fpath = os.path.join(root, fname)
                            raw = OmegaConf.to_container(OmegaConf.load(fpath), resolve=resolve)

                            if not isinstance(raw, dict):
                                continue

                            for key, value in raw.items():
                                if isinstance(value, dict) and "type" in value:
                                    component_cfgs[key] = value
                                    component_names.append(key)
            else:
                raise FileNotFoundError(f"{component_name.capitalize()} config path not found: {config_path}")

            if not component_names:
                raise ValueError(f"No valid {component_name} configs found in {config_path}")

            return component_cfgs, component_names

        dataset_cfgs, dataset_names = load_component_configs(dataset_config_path, "dataset")
        codec_cfgs = {}
        codec_names = []

        if os.path.isfile(codec_config_path):
            raw = OmegaConf.to_container(OmegaConf.load(codec_config_path), resolve=resolve)
            
            if not isinstance(raw, dict):
                raise ValueError(f"Invalid codec config file: {codec_config_path}")
            
            for key, value in raw.items():
                if isinstance(value, dict) and "type" in value:
                    codec_cfgs[key] = value
                    codec_names.append(key)
            
            if not codec_names:
                raise ValueError(
                    f"No valid codec configs found in {codec_config_path}. "
                    f"Expected structure: {{codec_name: {{type: ..., params: ...}}}}"
                )

        elif os.path.isdir(codec_config_path):
            for root, _, files in os.walk(codec_config_path):
                for fname in sorted(files):
                    if fname.endswith((".yaml", ".yml")):
                        fpath = os.path.join(root, fname)
                        raw = OmegaConf.to_container(OmegaConf.load(fpath), resolve=resolve)
                        
                        if not isinstance(raw, dict):
                            continue
                        
                        for key, value in raw.items():
                            if isinstance(value, dict) and "type" in value:
                                codec_cfgs[key] = value
                                codec_names.append(key)
        else:
            raise FileNotFoundError(f"Codec config path not found: {codec_config_path}")

        if not codec_names:
            raise ValueError(f"No valid codec configs found in {codec_config_path}")

        def extract_names(cfg, key):
            if isinstance(cfg, dict) or hasattr(cfg, "get"):
                if key in cfg and cfg[key] is not None:
                    return list(cfg[key])
                return [k for k, v in OmegaConf.to_container(cfg, resolve=resolve).items() if isinstance(v, dict)]
            return []

        metric_names = extract_names(metric_cfg, "metrics")

        if not dataset_names:
            raise ValueError(f"No datasets found in {dataset_config_path}")
        if not metric_names:
            raise ValueError(f"No metrics found in {metric_config_path}")

        main_dict = {
            "datasets": dataset_names,
            "metrics": metric_names,
            "codecs": sorted(codec_names),
        }

        for dname in dataset_names:
            main_dict[dname] = dataset_cfgs[dname]

        mt_container = OmegaConf.to_container(metric_cfg, resolve=resolve)
        for name in metric_names:
            if name in mt_container:
                main_dict[name] = mt_container[name]
        for k, v in mt_container.items():
            if k not in main_dict:
                main_dict[k] = v

        for cname in sorted(codec_names):
            main_dict[cname] = codec_cfgs[cname]

        return OmegaConf.create(main_dict)
    
    except Exception as e:
        import traceback
        print(f"Error loading configs: {e}")
        traceback.print_exc()
        raise


def load_video_config(config_path, dataset_config_path, metric_config_path, codec_config_path):
    """Load video experiment config and merge split codec/metric definitions."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Video config not found: {config_path}")

    base = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    if not isinstance(base, dict):
        raise ValueError(f"Invalid video config file: {config_path}")

    merged = dict(base)
    component_cfg = load_config_files(
        dataset_config_path,
        metric_config_path,
        codec_config_path,
        resolve=False,
    )
    component_dict = OmegaConf.to_container(component_cfg, resolve=False)
    merged.update({k: v for k, v in component_dict.items() if k not in {"datasets", "codecs", "metrics"}})

    missing = []
    for key in list(merged.get("datasets", [])) + list(merged.get("codecs", [])) + list(merged.get("metrics", [])):
        if key not in merged:
            missing.append(key)
    if missing:
        raise KeyError(
            "Missing video component definitions: "
            + ", ".join(missing)
            + ". Check --video_dataset_config, --video_codec_config, and --video_metric_config."
        )

    return OmegaConf.create(merged)


def main():   
    args = parse_args()
    is_video_benchmark = args.mode == "video" or bool(args.config)

    dist_utils.init_distributed_mode(args)
    dist_utils.random_seed(args.seed, dist_utils.get_rank())
    
    # Load config
    if is_video_benchmark:
        if args.config:
            config = load_video_config(
                args.config,
                args.video_dataset_config,
                args.video_metric_config,
                args.video_codec_config,
            )
        else:
            config = load_config_files(
                args.video_dataset_config,
                args.video_metric_config,
                args.video_codec_config,
                resolve=False,
            )
    else:
        config = load_config_files(args.image_dataset_config, args.image_metric_config, args.image_codec_config)

    datasets_name = config['datasets']
    codecs_name = config['codecs']
    metrics_name = config['metrics']

    codecs = []
    
    for cname in codecs_name:
        # create a resolved dict copy, inject codec-specific zero_mean flags, then back to OmegaConf
        cfg_dict = OmegaConf.to_container(config, resolve=False)
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
            metric = instantiate_from_config(cfg[mname])

            if hasattr(metric, "bind_codec"):
                metric.bind_codec(codec)

            if is_video_benchmark and not getattr(metric, "is_complexity_metric", False):
                metric = adapt_metric_for_video(str(mname), metric)

            metrics.append((mname, metric))

        codecs.append((cname, codec, dataset_zero_mean, metrics_zero_mean, datasets, metrics))

    # Evaluation loop
    for cname, codec, dataset_zero_mean, metrics_zero_mean, datasets, metrics in codecs:
        for dname, dataset in datasets:
            world_size = dist_utils.get_world_size()
            rank = dist_utils.get_rank()
            complexity_outputs = {}

            if rank == 0:
                for mname, metric in metrics:
                    if getattr(metric, "is_complexity_metric", False):
                        complexity_outputs[mname] = metric.compute(
                            device=next(codec.parameters()).device
                        )
            args.img_path = getattr(dataset, "root", None)
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
                max_save = 3
                for batch in data_loader:
                    img = batch["img"].cuda()
                    rec, bpp = codec(img)
                    if cname =='hific_q0' or cname == 'hific_q2':
                        img = (img + 1.) / 2
                    for i in range(min(img.shape[0], max_save)):
                        if dataset_zero_mean:
                            x_recon_vis = rec[i] * 0.5 + 0.5
                        else:
                            x_recon_vis = rec[i]
                        save_reconstruction_preview(x_recon_vis, os.path.join(cache_file_name, f'recon_{i}.png'))

                    
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
                        if getattr(metric, "is_complexity_metric", False):
                            continue
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
                                    metric_results[mname][j].append(_ensure_1d_cpu(gathered0[j]))
                                
                                sec_name = 'msssim' if mname == 'ssim' else f"{mname}_2"
                                if sec_name not in metric_results:
                                    metric_results[sec_name] = [[] for _ in range(world_size)]
                                for j in range(world_size):
                                    metric_results[sec_name][j].append(_ensure_1d_cpu(gathered1[j]))
                            else:
                                # Handle cases with more than 2 outputs
                                for i, scores in enumerate(out):
                                    gathered = [torch.zeros_like(scores) for _ in range(world_size)]
                                    dist.all_gather(gathered, scores)
                                    
                                    key = f"{mname}_{i}"
                                    if key not in metric_results:
                                        metric_results[key] = [[] for _ in range(world_size)]
                                    for j in range(world_size):
                                        metric_results[key][j].append(_ensure_1d_cpu(gathered[j]))
                        elif out is None:
                            # FID: accumulate activations internally, no output to gather
                            pass
                        else:
                            # Single output metric
                            gathered = [torch.zeros_like(out) for _ in range(world_size)]
                            dist.all_gather(gathered, out)
                            
                            for j in range(world_size):
                                metric_results[mname][j].append(_ensure_1d_cpu(gathered[j]))
                    
                    
                    total_num += world_size * img.shape[0]
            
            dist.barrier()

            # Gather FID activations from all ranks
            fid_gathered = {}
            dataset_metric_gathered = {}
            for mname, metric in metrics:
                if mname == 'fid':
                    local_x = np.vstack(metric.all_activations_x) if len(metric.all_activations_x) > 0 else np.empty((0,2048), dtype=np.float32)
                    local_xr = np.vstack(metric.all_activations_xr) if len(metric.all_activations_xr) > 0 else np.empty((0,2048), dtype=np.float32)
                    gathered_x = [None for _ in range(world_size)]
                    gathered_xr = [None for _ in range(world_size)]
                    dist.all_gather_object(gathered_x, local_x)
                    dist.all_gather_object(gathered_xr, local_xr)
                    fid_gathered[mname] = (gathered_x, gathered_xr)
                gathered = gather_dataset_metric(metric, world_size)
                if gathered is not None:
                    dataset_metric_gathered[mname] = gathered
            
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
                for mname, metric in metrics:
                    if getattr(metric, "is_complexity_metric", False):
                        print(metric.format_result())

                # Process metrics
                for mname in metric_results.keys():
                    if mname == 'fid' or mname in dataset_metric_gathered:
                        continue  
                    if not any(metric_results[mname]):
                        continue
                    for j in range(world_size):
                        metric_results[mname][j] = torch.cat(metric_results[mname][j], dim=0).numpy()
                    
                    # Reorganize to restore original order
                    metric_reorg = []
                    for j in range(total_num):
                        metric_reorg.append(metric_results[mname][j % world_size][j // world_size])
                    metric_array = np.vstack(metric_reorg)
                    
                    print(f"{mname:12s}: {np.mean(metric_array):.4f} (±{np.std(metric_array):.4f})")
                
                for mname, metric in metrics:
                    if mname == 'fid':
                        gathered_x, gathered_xr = fid_gathered.get(mname, ([], []))
            
                        all_pred_x = np.vstack(gathered_x)
                        all_pred_xr = np.vstack(gathered_xr)
                        from cab.evaluations.fid.get_fid import compute_fid_from_activations
                        fid_score = compute_fid_from_activations(all_pred_x, all_pred_xr)
                        print(f"{'fid':12s}: {fid_score:.4f}")
                    elif mname in dataset_metric_gathered:
                        print_dataset_metric(mname, metric, dataset_metric_gathered[mname])
                
                print(f"{'='*60}\n")
            
            dist.barrier()
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
