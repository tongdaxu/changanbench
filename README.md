# ChanganBench

A benchmark toolkit for evaluating image codecs with common pixel-level, perceptual and distributional metrics (PSNR, SSIM, MS-SSIM, LPIPS, DISTS, FID). Designed for evaluation using PyTorch DDP.

## Description
ChanganBench runs codec inference on datasets, computes metrics, and aggregates results across distributed workers. Components are configurable via YAML and instantiated dynamically using [`cab.utils.instantiate_from_config`](./cab/utils.py).

## Table of Contents
- [Usage](#usage)  
- [Features](#features)  
- [Configuration](#configuration)  
- [Flow](#flow)  

## Usage
1. Install requirements and activate your environment .
2. Prepare a config (e.g. [`config/config.yaml`](./config/config.yaml)).
3. Run distributed evaluation :
```bash
# single-node multi-GPU
CUDAVISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node=8 \
  ddp_test.py \
  --config config/config.yaml \
  --cache_dir ./cache \
  --batch_size 32 \
  --num_workers 4
```


## Features
- Distributed evaluation using PyTorch DDP (`torch.distributed`).
- Modular config-driven instantiation: codecs, datasets, metrics via [`cab.utils.instantiate_from_config`](./cab/utils.py).
- Metrics :
  - PSNR: [`cab/evaluations/psnr.py`](./cab/evaluations/psnr.py)
  - SSIM / MS-SSIM: [`cab/evaluations/ssim.py`](./cab/evaluations/ssim.py)
  - LPIPS: [`cab/evaluations/lpips.py`](./cab/evaluations/lpips.py)
  - DISTS: [`cab/evaluations/dists.py`](./cab/evaluations/dists.py)
  - FID: [`cab/evaluations/fid/get_fid.py`](./cab/evaluations/fid/get_fid.py) + [`cab/evaluations/fid/fid_score.py`](./cab/evaluations/fid/fid_score.py)
- Codecs :
  - HiFiC: [`cab/codec/hific.py`](./cab/codec/hific.py)
  - SSDD: [`cab/codec/ssdd.py`](./cab/codec/ssdd.py)
  - ...
  -  More codecs can be added by implementing a new class with a `forward` method that takes an image and returns (reconstruction, bpp).
  
  
## Configuration
- [`config/config.yaml`](./config/config.yaml)


Config structure:
- `datasets`: list of dataset keys 
- `codecs`: list of codec keys
- `metrics`: list of metric keys

Each dataset/codec/metric key maps to a `type` string (module path + callable/class) and optional `params`. Example entry:
```yaml
kodak:
  type: cab.dataset.data.SimpleDataset
  params:
    root: /path/to/list.txt
    image_size: 256
    zero_mean: false
```

Important runtime args :
- `--config`: path to YAML config
- `--cache_dir`: directory to save visualizations / cache
- `--batch_size`, `--num_workers`, `--image_size`
- `--zero-mean`: affect metric preprocessing/visualization

## Flow
Below is a high-level flow of the evaluation pipeline.

```mermaid
flowchart LR
  A[Config YAML] --> B[instantiate components<br/>(datasets, codecs, metrics)]
  B --> C[Distributed Setup<br/>(torch.distributed)]
  C --> D[DistributedSampler + DataLoader]
  D --> E[Per-rank Inference:<br/>codec(img) -> (rec, bpp)]
  E --> F[Per-rank Metrics<br/>(psnr, ssim, lpips, dists, fid)]
  F --> G[All-gather per-batch results]
  G --> H[Rank 0 aggregation & statistics]
  H --> I[Save logs / images / cache]
```
