from torchvision.datasets import VisionDataset
from PIL import Image
from glob import glob
from typing import List, Dict, Tuple
import torchvision.transforms.v2 as transforms
import torch


def _collect_image_paths(root: str) -> List[str]:
    if root.endswith(".txt"):
        with open(root, "r", encoding="utf-8") as f:
            lines = f.readlines()
        fpaths = [line.strip("\n") for line in lines if line.strip()]
    else:
        exts = ["*.JPEG", "*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"]
        fpaths = []
        for ext in exts:
            fpaths += sorted(glob(root + f"/**/{ext}", recursive=True))
    return fpaths


def _crop_to_multiple_of(img: Image.Image, multiple: int = 512, mode: str = "center") -> Tuple[Image.Image, Dict]:
    """
    将图片裁剪到宽高均为 multiple 的倍数。
    mode:
      - center: 中心裁剪
      - topleft: 左上裁剪
    """
    w, h = img.size
    new_w = (w // multiple) * multiple
    new_h = (h // multiple) * multiple

    if new_w <= 0 or new_h <= 0:
        raise ValueError(f"Image too small for multiple={multiple}, got size={(w, h)}")

    if mode == "center":
        left = (w - new_w) // 2
        top = (h - new_h) // 2
    elif mode == "topleft":
        left, top = 0, 0
    else:
        raise ValueError(f"Unsupported crop mode: {mode}")

    right = left + new_w
    bottom = top + new_h

    cropped = img.crop((left, top, right, bottom))
    crop_info = {
        "orig_w": w,
        "orig_h": h,
        "crop_left": left,
        "crop_top": top,
        "crop_w": new_w,
        "crop_h": new_h,
    }
    return cropped, crop_info


def _image_to_patches(img_tensor: torch.Tensor, patch_size: int = 512) -> Tuple[torch.Tensor, Dict]:
    """
    输入:
      img_tensor: [C, H, W]
    输出:
      patches: [N, C, patch_size, patch_size]
      patch_info: 包含网格信息，用于重组
    """
    c, h, w = img_tensor.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"Tensor size must be divisible by patch_size={patch_size}, got {(h, w)}")

    nh = h // patch_size
    nw = w // patch_size

    # unfold -> [C, nh, nw, ph, pw]
    patches = (
        img_tensor.unfold(1, patch_size, patch_size)
        .unfold(2, patch_size, patch_size)
        .permute(1, 2, 0, 3, 4)  # [nh, nw, C, ph, pw]
        .contiguous()
        .view(nh * nw, c, patch_size, patch_size)  # [N, C, ph, pw]
    )

    patch_info = {
        "patch_size": patch_size,
        "grid_h": nh,
        "grid_w": nw,
        "full_h": h,
        "full_w": w,
    }
    return patches, patch_info


def stitch_patches(patches: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """
    将 [N, C, P, P] 重组回 [C, H, W]，N = grid_h * grid_w
    """
    n, c, p, p2 = patches.shape
    assert p == p2
    assert n == grid_h * grid_w, f"N({n}) != grid_h*grid_w({grid_h * grid_w})"

    # [grid_h, grid_w, C, P, P] -> [C, H, W]
    out = (
        patches.view(grid_h, grid_w, c, p, p)
        .permute(2, 0, 3, 1, 4)   # [C, grid_h, P, grid_w, P]
        .contiguous()
        .view(c, grid_h * p, grid_w * p)
    )
    return out


class EnterpriseCameraImageDataset(VisionDataset):
    """
    统一支持两种模式:
      1) eval_mode='full'  : 返回整图 tensor（适合 JPEG 等）
      2) eval_mode='patch' : 返回 patch tensor [N,C,512,512]（适合 PerCo 等）
    """
    def __init__(
        self,
        root: str,
        zero_mean: bool = False,
        multiple: int = 512,
        patch_size: int = 512,
        eval_mode: str = "full",      # 'full' or 'patch'
        crop_mode: str = "center",    # 'center' or 'topleft'
    ):
        super().__init__(root)
        self.zero_mean = zero_mean
        self.multiple = multiple
        self.patch_size = patch_size
        self.eval_mode = eval_mode
        self.crop_mode = crop_mode

        if self.eval_mode not in ["full", "patch"]:
            raise ValueError(f"eval_mode must be 'full' or 'patch', got {self.eval_mode}")
        if self.multiple % self.patch_size != 0:
            # 常规场景 multiple=patch_size=512，也允许未来扩展
            raise ValueError("multiple should be divisible by patch_size for consistent tiling.")

        transform_list = [transforms.ToTensor()]
        if self.zero_mean:
            transform_list.append(
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )
        self.transform = transforms.Compose(transform_list)

        self.fpaths = _collect_image_paths(root)
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert("RGB")

        # 第一步：裁剪到 512 的倍数
        cropped_img, crop_info = _crop_to_multiple_of(
            img,
            multiple=self.multiple,
            mode=self.crop_mode
        )

        # 转 tensor
        img_tensor = self.transform(cropped_img)  # [C,H,W]

        if self.eval_mode == "full":
            # JPEG等整图编码器直接用
            return {
                "img": img_tensor,          # [C,H,W]
                "fpath": fpath,
                "mode": "full",
                "crop_info": crop_info,
            }

        # PerCo等 patch 模型
        patches, patch_info = _image_to_patches(img_tensor, patch_size=self.patch_size)
        return {
            "patches": patches,            # [N,C,512,512]
            "fpath": fpath,
            "mode": "patch",
            "crop_info": crop_info,
            "patch_info": patch_info,
        }