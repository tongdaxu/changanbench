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