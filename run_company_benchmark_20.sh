#!/usr/bin/env bash
# 机器 9：公司 Kodak 默认 20 模型一键评测
#
# 默认行为：
#   1. 检查个人测试仓库、GPU、数据、权重与 Docker 镜像
#   2. 镜像不存在时构建 myapp:dev；已存在时直接复用
#   3. 启动一次性 Docker 容器
#   4. 在容器内激活 illm
#   5. 运行 run_benchmark.py；调度器按注册表自动切换 Conda 环境
#   6. 将结果保存在个人 outputs 目录
#
# 常用：
#   bash run_company_benchmark.sh
#   bash run_company_benchmark.sh --dry-run
#   bash run_company_benchmark.sh --output company_10 --resume
#   bash run_company_benchmark.sh --gpu 1
#
# 不会执行 git pull/reset/clean，不会修改公共仓库，也不会终止其他进程。

set -Eeuo pipefail

REPO="${REPO:-/data/benchmark/changanbench-company-test-wxt-20260730/changanbench}"
OUTPUTS="${OUTPUTS:-/data/benchmark/changanbench-company-test-wxt-20260730/outputs}"
IMAGE="${IMAGE:-myapp:dev}"
GPU="${GPU:-0}"
RUN_NAME="${RUN_NAME:-company_10_$(date +%Y%m%d_%H%M%S)}"

HOST_CACHE="${HOST_CACHE:-/data9-2/BenchmarkData/cache}"
HOST_WEIGHTS="${HOST_WEIGHTS:-/data9-2/BenchmarkData/ImageCodecWeights}"
HOST_DATASETS="${HOST_DATASETS:-/data9-2/BenchmarkData/datasets/ImageDatasets}"

DATASET_CONFIG="${DATASET_CONFIG:-config/image_datasets/company_full.yaml}"
DATASET_ROOT="${DATASET_ROOT:-/root/changanbench/ImageDatasets/PhotoCD_PCD0992/kodak}"
REGISTRY="${REGISTRY:-config/benchmarks/image_codec_registry.yaml}"

MODELS=(all)
DRY_RUN=0
RESUME=0
REBUILD=0
FAIL_FAST=0

usage() {
  cat <<'EOF'
用法：
  bash run_company_benchmark.sh [选项]

选项：
  --dry-run             只检查并生成计划，不运行模型
  --resume              复用同一输出目录中已有的成功 JSON
  --rebuild             即使 myapp:dev 已存在也重新构建镜像
  --fail-fast           首个模型失败后不再启动后续模型
  --all                 运行注册表全部启用模型（默认行为）
  --models NAME...      只运行指定模型；必须放在最后
  --output NAME         输出子目录名，默认带时间戳
  --gpu ID              GPU 编号，默认 0
  -h, --help            显示帮助

环境变量也可覆盖：
  REPO OUTPUTS IMAGE GPU RUN_NAME HOST_CACHE HOST_WEIGHTS HOST_DATASETS
  DATASET_CONFIG DATASET_ROOT REGISTRY
EOF
}

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --rebuild)
      REBUILD=1
      shift
      ;;
    --fail-fast)
      FAIL_FAST=1
      shift
      ;;
    --all)
      MODELS=(all)
      shift
      ;;
    --models)
      shift
      (($# > 0)) || { echo "错误：--models 后至少需要一个模型名。" >&2; exit 2; }
      MODELS=("$@")
      break
      ;;
    --output)
      (($# >= 2)) || { echo "错误：--output 缺少目录名。" >&2; exit 2; }
      RUN_NAME="$2"
      shift 2
      ;;
    --gpu)
      (($# >= 2)) || { echo "错误：--gpu 缺少 GPU 编号。" >&2; exit 2; }
      GPU="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "错误：未知参数 $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

fail() {
  echo "错误：$*" >&2
  exit 1
}

require_dir() {
  [[ -d "$1" ]] || fail "目录不存在：$1"
}

require_file() {
  [[ -f "$1" ]] || fail "文件不存在：$1"
}

[[ "$RUN_NAME" != */* && "$RUN_NAME" != "." && "$RUN_NAME" != ".." ]] \
  || fail "--output 只能是一个安全的子目录名，不能包含 /。"

command -v sudo >/dev/null 2>&1 || fail "找不到 sudo。"
command -v docker >/dev/null 2>&1 || fail "找不到 docker。"
command -v nvidia-smi >/dev/null 2>&1 || fail "找不到 nvidia-smi。"

require_dir "$REPO"
require_dir "$HOST_CACHE"
require_dir "$HOST_WEIGHTS"
require_dir "$HOST_DATASETS"
require_file "$REPO/run_benchmark.py"
require_file "$REPO/$REGISTRY"
require_file "$REPO/$DATASET_CONFIG"

# 只检查当前分支，不自动切换或修改 Git。
if command -v git >/dev/null 2>&1 && git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  BRANCH="$(git -C "$REPO" branch --show-current)"
  HEAD_INFO="$(git -C "$REPO" show -s --format='%h %s' HEAD)"
  echo "代码分支：${BRANCH:-<detached>}"
  echo "代码版本：$HEAD_INFO"

  if [[ "$BRANCH" != "refactor/unified-image-entry" ]]; then
    fail "当前分支不是 refactor/unified-image-entry；脚本不会自动切换分支。"
  fi

  if [[ -n "$(git -C "$REPO" status --short)" ]]; then
    echo "警告：个人测试仓库存在未提交修改；脚本会按当前文件运行，但不会改动它们。" >&2
  fi
fi

echo
echo "GPU ${GPU} 当前状态："
nvidia-smi -i "$GPU" \
  --query-gpu=index,name,memory.total,memory.used,memory.free \
  --format=csv,noheader || fail "无法访问 GPU ${GPU}。"

echo
echo "GPU ${GPU} 上现有计算进程（脚本不会终止它们）："
nvidia-smi -i "$GPU" \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader 2>/dev/null || true

if ((REBUILD)); then
  echo
  echo "按要求重新构建镜像：$IMAGE"
  sudo docker build -t "$IMAGE" "$REPO"
elif sudo docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo
  echo "复用现有镜像：$IMAGE"
else
  echo
  echo "镜像不存在，开始构建：$IMAGE"
  sudo docker build -t "$IMAGE" "$REPO"
fi

mkdir -p \
  "$OUTPUTS" \
  "$OUTPUTS/tmp" \
  "$OUTPUTS/matplotlib-cache" \
  "$OUTPUTS/torch_extensions" \
  "$OUTPUTS/cuda-cache"

HOST_RUN_DIR="$OUTPUTS/$RUN_NAME"
CONTAINER_RUN_DIR="/root/company-output/$RUN_NAME"

if [[ -e "$HOST_RUN_DIR" && $RESUME -eq 0 && $DRY_RUN -eq 0 ]]; then
  fail "输出目录已存在：$HOST_RUN_DIR。请换 --output 名称，或明确使用 --resume。"
fi

BENCH_ARGS=(
  python run_benchmark.py
  --models "${MODELS[@]}"
  --dataset-config "$DATASET_CONFIG"
  --dataset-root "$DATASET_ROOT"
  --output-dir "$CONTAINER_RUN_DIR"
  --gpu "$GPU"
  --conda-executable /root/miniconda3/bin/conda
)

((DRY_RUN)) && BENCH_ARGS+=(--dry-run)
((RESUME)) && BENCH_ARGS+=(--resume)
((FAIL_FAST)) && BENCH_ARGS+=(--fail-fast)

printf -v BENCH_COMMAND '%q ' "${BENCH_ARGS[@]}"

echo
echo "============================================================"
echo "准备启动公司图片评测"
echo "仓库：$REPO"
echo "镜像：$IMAGE"
echo "GPU：$GPU"
echo "模型：${MODELS[*]}"
echo "数据：$DATASET_ROOT"
echo "输出：$HOST_RUN_DIR"
((DRY_RUN)) && echo "模式：dry-run"
((RESUME)) && echo "模式：resume"
echo "============================================================"
echo

set +e
sudo docker run --rm \
  --gpus "device=${GPU}" \
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
  -v "$HOST_CACHE:/root/.cache:ro" \
  -v "$HOST_WEIGHTS:/root/changanbench/ImageCodecWeights:ro" \
  -v "$HOST_DATASETS:/root/changanbench/ImageDatasets:ro" \
  "$IMAGE" \
  /bin/bash -lc "
    set -Eeuo pipefail
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate illm
    cd /root/changanbench
    echo \"容器 Python：\$(which python)\"
    python -c \"import yaml; print('PyYAML OK')\"
    exec ${BENCH_COMMAND}
  "
EXIT_CODE=$?
set -e

echo
echo "============================================================"
if ((EXIT_CODE == 0)); then
  echo "评测命令完成。"
else
  echo "评测命令退出码：$EXIT_CODE"
fi
echo "宿主机结果目录：$HOST_RUN_DIR"
echo "CSV：$HOST_RUN_DIR/benchmark.csv"
echo "Markdown：$HOST_RUN_DIR/benchmark.md"
echo "日志：$HOST_RUN_DIR/logs/"
echo "============================================================"

exit "$EXIT_CODE"
