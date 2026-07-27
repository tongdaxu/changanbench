sudo docker build -t myapp:dev .

sudo docker run --rm -it \
  --gpus all \
  --ipc=host \
  -e TORCH_HOME=/root/.cache/torch \
  -e LD_LIBRARY_PATH=/root/changanbench/ImageCodecWeights/hm/lib \
  -v /data/benchmark/changanbench:/root/changanbench \
  -v /data9-2/BenchmarkData/cache:/root/.cache \
  -v /data9-2/BenchmarkData/cache/huggingface:/root/.cache/huggingface:ro \
  -v /data9-2/BenchmarkData/ImageCodecWeights:/root/changanbench/ImageCodecWeights \
  -v /data9-2/BenchmarkData/datasets/ImageDatasets/:/root/changanbench/ImageDatasets \
  myapp:dev \
  /bin/bash

source /root/miniconda3/etc/profile.d/conda.sh

conda activate infinity

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  ddp_test.py \
  --cache_dir ./rec_cache \
  --image_codec_config config/image_codecs/tatok_q0.yaml \
  --batch_size 4 \
  --num_workers 4

Company image evaluation (variable resolutions require `batch_size=1`):

```bash
CUDA_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  ddp_test_for_company.py \
  --cache_dir ./rec_cache \
  --image_dataset_config config/image_datasets/company_full.yaml \
  --image_codec_config config/image_codecs/jpeg_q37.yaml \
  --batch_size 1 \
  --num_workers 4
```

`config/image_datasets/company_patch.yaml` is retained only as a deprecation
notice and cannot be used as a dataset configuration. Use
`config/image_datasets/company_full.yaml`; BSQ and PerCo select their required
256/512 tiling inside their codecs.
