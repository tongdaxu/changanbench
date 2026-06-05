from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from changan_video.evaluations.config import fvd_options_from_config, str_to_bool
from changan_video.utils import inject_codec_zero_means, instantiate_from_config

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

torch = None
dist = None
dist_utils = None
OmegaConf = None


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="test changan bench with DDP")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def create_data_loader(dataset, batch_size, num_workers):
    if dist_utils.get_world_size() > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=False,
        )
    else:
        sampler = None

    return torch.utils.data.DataLoader(
        dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=False,
    )


def main() -> None:
    args = parse_args()
    if not args.config:
        raise SystemExit("--config is required.")

    _ensure_runtime_imports()
    dist_utils.init_distributed_mode(args)
    dist_utils.random_seed(args.seed, dist_utils.get_rank())

    config_path = Path(args.config)
    config = OmegaConf.load(config_path)
    datasets_name = list(config["datasets"])
    codecs_name = list(config["codecs"])
    metrics_name = list(config["metrics"])

    device = _runtime_device(args)

    codecs = []
    for codec_name in codecs_name:
        cfg_dict = OmegaConf.to_container(config, resolve=True)
        cfg_dict = inject_codec_zero_means(cfg_dict, str(codec_name))
        cfg = OmegaConf.create(cfg_dict)

        codec = instantiate_from_config(cfg[codec_name]).to(device)
        codec.eval()

        codec_params = _plain_params(cfg[codec_name])
        dataset_zero_mean = str_to_bool(codec_params.get("dataset_zero_mean", False))
        metrics_zero_mean = str_to_bool(codec_params.get("metrics_zero_mean", False))

        datasets = []
        for dataset_name in datasets_name:
            dataset = instantiate_from_config(cfg[dataset_name])
            datasets.append((str(dataset_name), dataset))

        metrics = []
        use_fid = False
        use_fvd = False
        for metric_name in metrics_name:
            metric_name = str(metric_name)
            special = _special_metric_kind(cfg, metric_name)
            if special == "fid":
                use_fid = True
                continue
            if special == "fvd":
                use_fvd = True
                continue
            metric = instantiate_from_config(cfg[metric_name])
            metrics.append((metric_name, metric))

        fvd_clip_length, fvd_clip_stride, fvd_model_path = fvd_options_from_config(
            cfg,
            metrics_name,
            base_dir=config_path.resolve().parent,
        )

        codecs.append(
            (
                str(codec_name),
                codec,
                dataset_zero_mean,
                metrics_zero_mean,
                datasets,
                metrics,
                use_fid,
                use_fvd,
                fvd_clip_length,
                fvd_clip_stride,
                fvd_model_path,
            )
        )

    for item in codecs:
        _evaluate_codec(args, device, item)

    if dist_utils.is_dist_avail_and_initialized():
        dist.destroy_process_group()


def _evaluate_codec(args, device, item) -> None:
    (
        codec_name,
        codec,
        dataset_zero_mean,
        metrics_zero_mean,
        datasets,
        metrics,
        use_fid,
        use_fvd,
        fvd_clip_length,
        fvd_clip_stride,
        fvd_model_path,
    ) = item

    world_size = dist_utils.get_world_size()
    rank = dist_utils.get_rank()

    fid_runner = _BatchFidRunner(device=device, metrics_zero_mean=metrics_zero_mean) if use_fid else None
    fvd_runner = (
        _BatchFvdRunner(
            device=device,
            metrics_zero_mean=metrics_zero_mean,
            clip_length=fvd_clip_length,
            clip_stride=fvd_clip_stride,
            model_path=fvd_model_path,
        )
        if use_fvd
        else None
    )

    for dataset_name, dataset in datasets:
        cache_file_name = Path(args.cache_dir) / codec_name / dataset_name
        if rank == 0:
            cache_file_name.mkdir(parents=True, exist_ok=True)

        metric_results = {name: [[] for _ in range(world_size)] for name, _ in metrics}
        bpp_results = [[] for _ in range(world_size)]
        fid_reference = [[] for _ in range(world_size)]
        fid_distorted = [[] for _ in range(world_size)]
        fvd_reference = [[] for _ in range(world_size)]
        fvd_distorted = [[] for _ in range(world_size)]

        data_loader = create_data_loader(dataset, args.batch_size, args.num_workers)
        if rank == 0 and tqdm is not None:
            data_loader = tqdm(data_loader, desc=f"Processing {codec_name}/{dataset_name}")

        with torch.no_grad():
            for batch in data_loader:
                img = batch["img"].to(device, non_blocking=True)
                rec, bpp = codec(img)

                bpp_tensor = _batch_vector(bpp, img.shape[0], device)
                for idx, gathered in enumerate(_gather_tensor(bpp_tensor)):
                    bpp_results[idx].append(gathered)

                for metric_name, metric in metrics:
                    out = metric(img, rec, zero_mean=metrics_zero_mean)
                    _gather_metric_output(metric_results, metric_name, out, img.shape[0], device)

                if fid_runner is not None:
                    pred_x, pred_xr = fid_runner.extract(img, rec)
                    for idx, gathered in enumerate(_gather_tensor(pred_x)):
                        fid_reference[idx].append(gathered)
                    for idx, gathered in enumerate(_gather_tensor(pred_xr)):
                        fid_distorted[idx].append(gathered)

                if fvd_runner is not None:
                    pred_x, pred_xr = fvd_runner.extract(img, rec)
                    for idx, gathered in enumerate(_gather_tensor(pred_x)):
                        fvd_reference[idx].append(gathered)
                    for idx, gathered in enumerate(_gather_tensor(pred_xr)):
                        fvd_distorted[idx].append(gathered)

        _barrier()
        if rank == 0:
            _print_results(
                codec_name,
                dataset_name,
                metric_results,
                bpp_results,
                fid_reference,
                fid_distorted,
                fvd_reference,
                fvd_distorted,
            )
        _barrier()


def _gather_metric_output(metric_results, metric_name, out, batch_size: int, device) -> None:
    if isinstance(out, (tuple, list)):
        if len(out) == 2:
            names = (metric_name, "msssim" if metric_name.startswith("ssim") else f"{metric_name}_2")
            for name, value in zip(names, out):
                if name not in metric_results:
                    metric_results[name] = [[] for _ in range(dist_utils.get_world_size())]
                _append_gathered(metric_results[name], _batch_vector(value, batch_size, device))
            return

        for idx, value in enumerate(out):
            name = f"{metric_name}_{idx}"
            if name not in metric_results:
                metric_results[name] = [[] for _ in range(dist_utils.get_world_size())]
            _append_gathered(metric_results[name], _batch_vector(value, batch_size, device))
        return

    _append_gathered(metric_results[metric_name], _batch_vector(out, batch_size, device))


def _append_gathered(target, tensor: torch.Tensor) -> None:
    for idx, gathered in enumerate(_gather_tensor(tensor)):
        target[idx].append(gathered)


def _batch_vector(value, batch_size: int, device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        return torch.full((batch_size,), float(value), dtype=torch.float32, device=device)
    value = value.to(device=device, dtype=torch.float32)
    if value.ndim == 0:
        return value.repeat(batch_size)
    if value.shape[0] != batch_size:
        return value.reshape(-1)
    return value.reshape(batch_size, -1).mean(dim=1)


def _gather_tensor(tensor: torch.Tensor) -> list[torch.Tensor]:
    tensor = tensor.contiguous()
    if dist_utils.get_world_size() == 1:
        return [tensor.detach().cpu()]
    gathered = [torch.zeros_like(tensor) for _ in range(dist_utils.get_world_size())]
    dist.all_gather(gathered, tensor)
    return [item.detach().cpu() for item in gathered]


def _print_results(
    codec_name,
    dataset_name,
    metric_results,
    bpp_results,
    fid_reference,
    fid_distorted,
    fvd_reference,
    fvd_distorted,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Results: {codec_name} on {dataset_name}")
    print(f"{'=' * 60}")

    bpp_flat = _concat_rank_chunks(bpp_results)
    if bpp_flat.size:
        print(f"BPP: {float(np.mean(bpp_flat)):.4f}")

    for metric_name in metric_results:
        values = _concat_rank_chunks(metric_results[metric_name])
        valid = values[~np.isnan(values)]
        if valid.size:
            print(f"{metric_name:12s}: {float(np.mean(valid)):.4f} (+/-{float(np.std(valid)):.4f})")
        else:
            print(f"{metric_name:12s}: nan (+/-nan)")

    if any(fid_reference):
        from changan_video.evaluations.fid.fid_score import compute_fid_score

        pred_x = _concat_rank_chunks(fid_reference, axis=0)
        pred_xr = _concat_rank_chunks(fid_distorted, axis=0)
        print(f"{'fid':12s}: {compute_fid_score(pred_xr, pred_x):.4f}")

    if any(fvd_reference):
        from changan_video.evaluations.fvd.fvd_score import compute_fvd_score

        pred_x = _concat_rank_chunks(fvd_reference, axis=0)
        pred_xr = _concat_rank_chunks(fvd_distorted, axis=0)
        print(f"{'fvd':12s}: {compute_fvd_score(pred_xr, pred_x):.4f}")

    print(f"{'=' * 60}\n")


def _concat_rank_chunks(rank_chunks, axis=None) -> np.ndarray:
    arrays = []
    for chunks in rank_chunks:
        if chunks:
            arrays.append(torch.cat(chunks, dim=0).numpy())
    if not arrays:
        return np.array([])
    if axis is None:
        return np.concatenate(arrays, axis=0).reshape(-1)
    return np.concatenate(arrays, axis=axis)


class _BatchFidRunner:
    def __init__(self, *, device, metrics_zero_mean: bool) -> None:
        from changan_video.evaluations.fid.inception import InceptionV3

        self.device = device
        self.metrics_zero_mean = metrics_zero_mean
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx], normalize_input=False).to(device)
        self.model.eval()

    def extract(self, x_input, x_recon) -> tuple[torch.Tensor, torch.Tensor]:
        return self._extract_one(x_input), self._extract_one(x_recon)

    def _extract_one(self, video: torch.Tensor) -> torch.Tensor:
        if not self.metrics_zero_mean:
            video = video * 2 - 1
        frames = video.permute(0, 2, 1, 3, 4).reshape(-1, video.shape[1], video.shape[3], video.shape[4])
        pred = self.model(frames)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, (1, 1))
        return pred.squeeze(3).squeeze(2)


class _BatchFvdRunner:
    def __init__(
        self,
        *,
        device,
        metrics_zero_mean: bool,
        clip_length: int,
        clip_stride: int,
        model_path: str | None,
    ) -> None:
        from changan_video.evaluations.fvd.fvd_score import get_i3d_model

        self.device = device
        self.metrics_zero_mean = metrics_zero_mean
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.model = get_i3d_model(model_path=model_path, device=str(device))

    def extract(self, x_input, x_recon) -> tuple[torch.Tensor, torch.Tensor]:
        return self._extract_one(x_input), self._extract_one(x_recon)

    def _extract_one(self, video: torch.Tensor) -> torch.Tensor:
        from changan_video.evaluations.fvd.fvd_score import extract_i3d_features

        if self.metrics_zero_mean:
            video = (video + 1.0) * 0.5
        video = video.detach().clamp(0.0, 1.0)
        arrays = (
            video.mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(0, 2, 3, 4, 1)
            .cpu()
            .numpy()
        )
        clips = []
        for item in arrays:
            clips.extend(_make_fvd_clips(item, self.clip_length, self.clip_stride))
        feats = extract_i3d_features(np.stack(clips, axis=0), model=self.model, device=str(self.device))
        return torch.from_numpy(feats).to(device=self.device, dtype=torch.float32)


def _make_fvd_clips(video: np.ndarray, clip_length: int, clip_stride: int) -> list[np.ndarray]:
    if clip_length <= 0:
        raise ValueError("fvd clip_length must be greater than 0")
    if clip_stride <= 0:
        raise ValueError("fvd clip_stride must be greater than 0")

    total = video.shape[0]
    starts = list(range(0, max(total - clip_length + 1, 1), clip_stride))
    if starts[-1] != max(total - clip_length, 0):
        starts.append(max(total - clip_length, 0))

    clips = []
    for start in starts:
        clip = video[start : start + clip_length]
        if clip.shape[0] < clip_length:
            pad = np.repeat(clip[-1:, ...], clip_length - clip.shape[0], axis=0)
            clip = np.concatenate([clip, pad], axis=0)
        clips.append(clip)
    return clips


def _plain_params(entry) -> dict:
    try:
        params = OmegaConf.to_container(entry.get("params", {}), resolve=True)
    except Exception:
        params = entry.get("params", {}) or {}
    return params or {}


def _special_metric_kind(config, metric_name: str) -> str | None:
    entry = config.get(metric_name, {}) or {}
    metric_type = str(entry.get("type", ""))
    text = f"{metric_name} {metric_type}".lower()
    if "fid" in text:
        return "fid"
    if "fvd" in text:
        return "fvd"
    return None


def _runtime_device(args):
    if torch.cuda.is_available():
        gpu = getattr(args, "gpu", 0)
        return torch.device("cuda", gpu)
    return torch.device("cpu")


def _barrier() -> None:
    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()


def _ensure_runtime_imports() -> None:
    global torch, dist, dist_utils, OmegaConf
    if torch is None:
        import torch as torch_module
        import torch.distributed as dist_module
        import changan_video.distributed as dist_utils_module
        from omegaconf import OmegaConf as omega_conf_module

        torch = torch_module
        dist = dist_module
        dist_utils = dist_utils_module
        OmegaConf = omega_conf_module


if __name__ == "__main__":
    main()
