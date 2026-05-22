import torch.distributed as dist

from omegaconf import OmegaConf
from cab.utils import instantiate_from_config
import argparse
import os
import torch
from tqdm import tqdm

from dataset.data import SimpleDataset

from evaluations.fid.fid_score import calculate_frechet_distance
from evaluations.lpips import get_lpips
from evaluations.psnr import get_psnr
from evaluations.ssim import get_ssim_and_msssim
from evaluations.fid.inception import InceptionV3

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="test changan bench")
    # logging params
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument('--log_dir', type=str, default="./logs", help='Directory to save logs')
    parser.add_argument('--dataset', type=str, default="kodak", help='Dataset name')
    parser.add_argument('--image_size', type=int, default=256, help='Size of input images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--distributed', action='store_true', help='Use distributed evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # 基础逻辑
    config = OmegaConf.load(args.config)

    datasets_name = config['datasets']
    codecs_name = config['codecs']
    metrics_name = config['metrics']

    datasets = []
    codecs = []
    metrics = []

    for dataset_name in datasets_name:
        dataset = instantiate_from_config(config[dataset_name])
        datasets.append((dataset_name, dataset))

    for codec_name in codecs_name:
        codec = instantiate_from_config(config[codec_name])
        codecs.append((codec_name, codec))

    for metric_name in metrics_name:
        metric = instantiate_from_config(config[metric_name])
        metrics.append((metric_name, metric))

    # ddp
    dist.init_process_group(
        backend='nccl',
        init_method="env://",
    )

    world_size = dist.get_world_size()

    BS = args.batch_size

    for cname, codec in codecs:

        for dname, dataset in datasets:
            
            cache_file_name = os.path.join(args.cache_dir, cname, dname)

            if os.path.exists(cache_file_name):
                imgs = torch.load(os.path.join(cache_file_name, "imgs.pt"))
                recs = torch.load(os.path.join(cache_file_name, "recs.pt"))
                bpps = torch.load(os.path.join(cache_file_name, "bpps.pt"))

            else:

                # distributed sampler
                # dataset split ...
                image_dataset = SimpleDataset(args.dataset, image_size=args.img_size)
                data_loader = ...

                imgs = []
                recs = []
                bpps = []
            
                for di, data in tqdm(data_loader):
                    img = data["img"]
                    rec, bpp = codec(img)

                    imgs.append(img)
                    recs.append(rec)
                    bpps.append(bpp)

                imgs = torch.cat(imgs)
                recs = torch.cat(recs)
                bpps = torch.cat(bpps)
                
                torch.save(imgs, os.path.join(cache_file_name, "imgs.pt"))
                torch.save(recs, os.path.join(cache_file_name, "recs.pt"))
                torch.save(bpps, os.path.join(cache_file_name, "bpps.pt"))

            for mname, metric in metrics:
                metric_value = metric(imgs, recs)

    # log format ... -> yaml 