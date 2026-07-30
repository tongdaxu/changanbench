# ChanganBench

ChanganBench 是一个配置驱动的图像与视频压缩评测工具。当前图片 benchmark 的目标是：通过一次命令选择任意 Dataset 配置、数据根目录和一组 Codec，串行完成评测并汇总结果。图像评测统一采用：

```python
rec, bpp = codec(img)
```

统一图片评测链路如下：

```text
run_benchmark.py
    └── 根据注册表切换 Conda 环境，并串行调用 ddp_test.py
```

- `run_benchmark.py`：面向用户的多模型图片 benchmark 调度器。
- `ddp_test.py`：单模型图片/视频评测执行器。
- `ddp_test_for_company.py`：旧的公司专项入口，暂时保留，但不属于新的统一调度链路。

公司 Dataset 通过 `--dataset-config` 和 `--dataset-root` 传给统一的 `ddp_test.py`。调度器会汇总每个模型的 JSON、日志、CSV 和 Markdown 结果。

## 功能概览

- 图片 Codec：JPEG、HM、VTM、ELIC、TCM、MLIC++、HiFiC、MS-ILLM、DiffEIC、StableCodec、PerCo、FSQ、BSQ、VAR、TA-Tok、Infinity、Cosmos、IBQ、FlowMo、SSDD。
- 图片指标：BPP、PSNR、SSIM、MS-SSIM、LPIPS、DISTS、FID、参数量、GFLOPs 和耗时。
- 图片数据集：Kodak、CLIC、ImageNet，以及本地公司原始分辨率图片。
- 视频评测：传统视频 Codec、DCVC 系列及相关视频指标；视频仍通过 `ddp_test.py` 和 `config/video_*` 配置运行。
- 支持 PyTorch Distributed Data Parallel；当前统一图片调度器默认每个模型使用单进程、单 GPU，并按模型串行执行。

## 运行前准备

正式评测环境必须提前具备：

1. NVIDIA 驱动和可用 GPU。
2. `illm`、`infinity`、`diffeic`、`stablecodec` 四个 Conda 环境。
3. 各 Codec YAML 引用的本地模型权重、传统 Codec 二进制和 Python 依赖。
4. Torch Hub、Hugging Face 等模型缓存。离线运行时不能依赖临时下载。
5. 本地图片数据目录。

`run_benchmark.py` 自身需要 Python 3 和 PyYAML。它不会安装环境或下载权重，只负责调用已有环境中的 `ddp_test.py`。

## Docker

以下示例将仓库、权重、缓存和 Dataset 以只读方式挂载，只给结果目录写权限。不要把公共仓库 `/data/benchmark/changanbench` 作为测试工作区，也不要将仓库以可写方式挂载到容器。

先设置宿主机路径：

```bash
REPO=/path/to/changanbench
OUTPUTS=/path/to/changanbench-outputs

mkdir -p "$OUTPUTS"/{tmp,matplotlib-cache,torch_extensions,cuda-cache}
```

机器 9 当前个人测试仓库示例：

```text
/data/benchmark/changanbench-company-test-wxt-20260730/changanbench
```

启动容器：

```bash
sudo docker run --rm -it \
  --gpus '"device=0"' \
  --ipc=host \
  -e TORCH_HOME=/root/.cache/torch \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e TMPDIR=/root/company-output/tmp \
  -e MPLCONFIGDIR=/root/company-output/matplotlib-cache \
  -e TORCH_EXTENSIONS_DIR=/root/company-output/torch_extensions \
  -e CUDA_CACHE_PATH=/root/company-output/cuda-cache \
  -e LD_LIBRARY_PATH=/root/changanbench/ImageCodecWeights/hm/lib \
  -v "$REPO:/root/changanbench:ro" \
  -v "$OUTPUTS:/root/company-output" \
  -v /data9-2/BenchmarkData/cache:/root/.cache:ro \
  -v /data9-2/BenchmarkData/ImageCodecWeights:/root/changanbench/ImageCodecWeights:ro \
  -v /data9-2/BenchmarkData/datasets/ImageDatasets:/root/changanbench/ImageDatasets:ro \
  myapp:dev \
  /bin/bash
```

进入容器后：

```bash
cd /root/changanbench
source /root/miniconda3/etc/profile.d/conda.sh
conda activate illm
```

`illm` 只用于启动总调度器；每个模型实际使用的环境由注册表决定。

### 构建镜像

机器 9 已有当前验证使用的 `myapp:dev`，通常不应重新构建或覆盖同名镜像。仅在镜像不存在，或 Dockerfile 有明确变更且已确认不会影响其他任务时，才执行：

```bash
sudo docker build -t myapp:dev .
```

## 快速开始

### 先执行 dry-run

`--dry-run` 会验证模型名、注册表和 YAML，生成每个模型的计划命令、JSON、日志和汇总状态，但不会启动 Conda 子环境、加载模型或占用 GPU。dry-run 生成的 `benchmark.csv` 和 `benchmark.md` 只包含调度状态，不包含真实指标。

```bash
python run_benchmark.py \
  --dataset-config config/image_datasets/company_full.yaml \
  --dataset-root /root/changanbench/ImageDatasets/PhotoCD_PCD0992/kodak \
  --output-dir /root/company-output/company_10 \
  --gpu 0 \
  --dry-run
```

当前调度器会在终端显示任务级状态和 dry-run 命令；每个模型的完整计划命令也会写入：

```text
/root/company-output/company_10/logs/<model>.log
```

### 运行默认 10 模型

省略 `--models` 时运行注册表中的默认 suite：

```bash
python run_benchmark.py \
  --dataset-config config/image_datasets/company_full.yaml \
  --dataset-root /root/changanbench/ImageDatasets/PhotoCD_PCD0992/kodak \
  --output-dir /root/company-output/company_10 \
  --gpu 0
```

默认模型为：

```text
bsq_q0 fsq_q0 jpeg_q22 var_q0 flowmo_q0
ssdd_q0 ibq_q0 perco_q0 tatok_q0 infinity_q0
```

其中前 8 个模型使用 `illm`，`tatok_q0` 和 `infinity_q0` 使用 `infinity`。TA-Tok 的环境归属来自机器 9 实测；`illm` 中的 Transformers 缺少 `Siglip2VisionConfig`，不能运行该模型。

正式运行期间：

- 终端显示当前序号、模型名、Conda 环境、START/SUCCESS/FAILED/SKIP、耗时和日志路径。
- 子进程的完整 stdout/stderr 写入 `<output-dir>/logs/<model>.log`，不会全部刷到终端。
- 主要汇总结果位于 `<output-dir>/benchmark.csv` 和 `<output-dir>/benchmark.md`。
- 单个模型失败时默认继续运行后续模型；全部任务结束后，只要存在失败模型，程序退出码为 1。
- 使用 `--fail-fast` 时，首个失败之后不再启动后续模型。

### 指定模型

模型名以空格分隔，名称区分大小写：

```bash
python run_benchmark.py \
  --models bsq_q0 jpeg_q22 tatok_q0 \
  --dataset-config config/image_datasets/company_full.yaml \
  --dataset-root /root/changanbench/ImageDatasets/PhotoCD_PCD0992/kodak \
  --output-dir /root/company-output/company_selected \
  --gpu 0
```

运行注册表内全部启用模型：

```bash
python run_benchmark.py \
  --models all \
  --dataset-config config/image_datasets/company_full.yaml \
  --dataset-root /root/changanbench/ImageDatasets/PhotoCD_PCD0992/kodak \
  --output-dir /root/company-output/company_all \
  --gpu 0
```

`--models all` 会包含尚未在当前 Docker 环境完成验证的模型，也会包含当前被运行时问题阻塞的 DiffEIC。未知模型、重复模型以及把 `all` 与其他名称混用都会在启动任务前报错。

## 公司原始分辨率图片

正式公司图片入口为：

```text
config/image_datasets/company_full.yaml
```

Dataset 读取完整图片，将高和宽中心裁剪为 512 的整数倍，并返回完整 Tensor。模型需要的 256×256 或 512×512 分块由对应 Codec 内部处理。公司评测使用 `--batch-size 1 --num-workers 0`；这也是调度器的默认值。

传入 `--dataset-root` 后，调度器会在输出目录创建：

```text
<output-dir>/config/dataset_override.yaml
```

该临时配置仅覆盖所选 Dataset 的 `params.root`，不会修改仓库中的原 YAML。

`config/image_datasets/company_patch.yaml` 仍保留在仓库中，但不是正式公司图片入口。

## 使用其他数据集

调度器不绑定 Kodak、公司图片或其他固定数据集。`--dataset-config` 可以指向任意符合仓库配置格式、且只定义一个 Dataset 的 YAML；`--dataset-root` 可以在不修改原 YAML 的情况下替换它的 `params.root`。

例如：

```bash
python run_benchmark.py \
  --models bsq_q0 jpeg_q22 \
  --dataset-config /path/to/dataset.yaml \
  --dataset-root /path/to/images \
  --output-dir /root/company-output/custom_dataset \
  --gpu 0
```

数据读取、裁剪和归一化由所选 Dataset 类决定，调度器本身不对图片做 resize、crop、pad 或分块。Codec 接收 Dataset 返回的完整 Tensor，并在内部完成自身需要的尺寸适配和分块。

对任意本地图片目录，可以复用 `config/image_datasets/company_full.yaml` 并通过 `--dataset-root` 指向实际目录。该 Dataset 会递归读取图片、跳过宽或高小于 512 的文件，并将其余图片中心裁剪到高宽均为 512 的整数倍，因此适合当前统一的公司整图 Codec 流程。

## 调度器参数

```text
--models NAME [NAME ...]  模型列表；省略时使用默认 suite，也可传 all
--dataset-config PATH     必填，单个 Dataset YAML
--dataset-root PATH       可选，覆盖 Dataset 的 params.root
--output-dir PATH         必填，结果输出目录
--metric-config PATH      默认 config/image_metrics.yaml
--registry PATH           默认 config/benchmarks/image_codec_registry.yaml
--batch-size N            默认 1
--num-workers N           默认 0
--gpu ID                  默认 0，并写入 CUDA_VISIBLE_DEVICES
--conda-executable PATH   默认 conda
--resume                  复用已有的成功 JSON
--dry-run                 生成并展示计划命令，不启动模型
--fail-fast               首个模型失败后不再启动后续模型
```

每个模型实际执行的形式为：

```bash
CUDA_VISIBLE_DEVICES=<gpu> conda run -n <env> torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  ddp_test.py \
  --mode image \
  --cache_dir <output>/cache/<model> \
  --image_codec_config <codec-yaml> \
  --image_dataset_config <dataset-yaml> \
  --image_metric_config config/image_metrics.yaml \
  --batch_size 1 \
  --num_workers 0 \
  --result-json <output>/json/<model>.json
```

默认情况下，单个模型失败不会阻止后续模型运行；全部任务完成后，若存在失败模型，调度器退出码为 1。使用 `--fail-fast` 时，首个失败之后的模型会记录为 `skipped_fail_fast`。

`--resume` 只复用状态为 `success`、模型名匹配且 Dataset 名称相同的 JSON。更换数据目录或修改同名 Dataset 配置时，应使用新的输出目录，避免复用旧结果。

## 当前验证状态

注册表包含 20 个图片 Codec。“已注册”只表示调度器知道对应的 Codec YAML 和 Conda 环境，不代表已在当前 Docker 环境完成端到端验证。

当前机器 9 验证状态：

- 已通过公司 Kodak 数据单模型实际测试：

  ```text
  bsq_q0 fsq_q0 jpeg_q22 var_q0 flowmo_q0
  ssdd_q0 ibq_q0 perco_q0 tatok_q0 infinity_q0
  ```

- 上述前 8 个模型使用 `illm`；`tatok_q0` 和 `infinity_q0` 使用 `infinity`。
- 默认 10 模型已通过统一调度器 dry-run，环境选择、命令生成和输出目录结构已确认。
- DiffEIC 当前被运行时问题阻塞：在当前 `myapp:dev` 的 `diffeic` 环境中，基础 Conv2d 测试也会出现段错误，因此不能归因于 Codec 分块重构或 GPU 调度。
- HM、VTM、ELIC、TCM、MLIC++、HiFiC、MS-ILLM、StableCodec、Cosmos Discrete 尚未完成当前统一流程的逐项实测，需要继续确认权重、缓存、二进制和依赖。

在真实的统一 10 模型批量运行完成并核对 `benchmark.csv` 前，不应将 dry-run 表格视为真实指标结果。

## 注册模型

注册表位于 `config/benchmarks/image_codec_registry.yaml`。当前注册 20 个图片 Codec 家族：

| 模型名 | Codec YAML | Conda 环境 |
|---|---|---|
| `jpeg_q22` | `config/image_codecs/jpeg_q22.yaml` | `illm` |
| `hm_q22` | `config/image_codecs/hm_q22.yaml` | `illm` |
| `vtm_q22` | `config/image_codecs/vtm_q22.yaml` | `illm` |
| `elic_q0` | `config/image_codecs/elic_q0.yaml` | `illm` |
| `tcm_q0` | `config/image_codecs/tcm_q0.yaml` | `illm` |
| `MLIC++_q0` | `config/image_codecs/MLIC++_q0.yaml` | `illm` |
| `hific_q0` | `config/image_codecs/hific_q0.yaml` | `illm` |
| `msillm_q0` | `config/image_codecs/msillm_q0.yaml` | `illm` |
| `diffeic_q0` | `config/image_codecs/diffeic_q0.yaml` | `diffeic` |
| `stablecodec_q0` | `config/image_codecs/stablecodec_q0.yaml` | `stablecodec` |
| `perco_q0` | `config/image_codecs/perco_q0.yaml` | `illm` |
| `fsq_q0` | `config/image_codecs/fsq_q0.yaml` | `illm` |
| `bsq_q0` | `config/image_codecs/bsq_q0.yaml` | `illm` |
| `var_q0` | `config/image_codecs/var_q0.yaml` | `illm` |
| `tatok_q0` | `config/image_codecs/tatok_q0.yaml` | `infinity` |
| `infinity_q0` | `config/image_codecs/infinity_q0.yaml` | `infinity` |
| `cosmos_discrete_q0` | `config/image_codecs/cosmos_discrete_q0.yaml` | `illm` |
| `ibq_q0` | `config/image_codecs/ibq_q0.yaml` | `illm` |
| `flowmo_q0` | `config/image_codecs/flowmo_q0.yaml` | `illm` |
| `ssdd_q0` | `config/image_codecs/ssdd_q0.yaml` | `illm` |

HM 和 VTM 还需要各自的本地二进制；MS-ILLM 需要已经填充的 `facebookresearch/NeuralCompression` Torch Hub 缓存。其他权重和缓存要求以对应 Codec YAML 及下载清单为准。

## 输出

每个模型都有独立的结果、日志和缓存目录：

```text
<output-dir>/
├── benchmark.csv
├── benchmark.md
├── config/
│   └── dataset_override.yaml    # 仅传入 --dataset-root 时生成
├── json/
│   └── <model>.json
├── logs/
│   └── <model>.log
└── cache/
    └── <model>/
```

`benchmark.csv` 和 `benchmark.md` 每个模型一行，列为：

```text
dataset, codec, status, bpp, psnr, ssim, ms_ssim,
lpips, dists, fid, parameters_m, gflops, time_ms,
environment, return_code, error_summary
```

缺失指标写为空值；失败模型仍会出现在汇总中，并记录返回码和日志末尾的错误摘要。每个 JSON 保存 `ddp_test.py` 在 rank 0 聚合得到的真实结果，以及调度器补充的环境、状态、返回码和执行命令。

对于 dry-run，上述文件记录的是计划状态，不是模型真实指标。正式运行完成后会在同一路径更新结果。

## 单模型入口

需要调试单个模型时，可直接运行 `ddp_test.py`：

```bash
conda run -n illm torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  ddp_test.py \
  --mode image \
  --cache_dir /root/company-output/bsq_single/cache \
  --image_codec_config config/image_codecs/bsq_q0.yaml \
  --image_dataset_config config/image_datasets/company_full.yaml \
  --image_metric_config config/image_metrics.yaml \
  --batch_size 1 \
  --num_workers 0 \
  --result-json /root/company-output/bsq_single/result.json
```

旧的 `ddp_test_for_company.py` 暂时保留，但新的多模型公司验收应优先使用 `run_benchmark.py`。

## 本地静态测试

以下检查不需要 GPU、Conda 子环境或模型权重：

```bash
python -m py_compile ddp_test.py run_benchmark.py
python -m pytest tests/test_run_benchmark.py -q
git diff --check
```

还可以使用 `--dry-run` 检查环境选择、命令、数据集覆盖和输出目录结构。

## 相关文件

- `run_benchmark.py`：统一图片 benchmark 调度入口。
- `config/benchmarks/image_codec_registry.yaml`：模型、Codec YAML 和 Conda 环境注册表。
- `ddp_test.py`：单模型图片/视频评测入口及 rank 0 JSON 输出。
- `ddp_test_for_company.py`：旧的公司专项入口，不属于统一多模型调度链路。
- `config/image_metrics.yaml`：图片指标配置。
- `config/image_datasets/company_full.yaml`：公司完整图片正式入口。
- `Reocrd.md`：早期机器 9 Docker 和单模型运行记录。
- `Progress.md`：Codec、指标和数据集状态记录。
