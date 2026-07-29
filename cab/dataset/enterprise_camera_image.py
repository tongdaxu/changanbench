from pathlib import Path
from PIL import Image
import torch
import numpy as np
import warnings
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms


class CompanyImageDataset(Dataset):

    CROP_MULTIPLE = 512

    IMG_EXTS = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    ]


    def __init__(
        self,
        root,
        zero_mean=None,
        mode="full",
        patch_size=512,
        crop_multiple=None
    ):

        self.root = Path(root)
        self.mode = mode
        self.patch_size = patch_size
        self.crop_multiple = self.CROP_MULTIPLE
        self.zero_mean = zero_mean
        if self.zero_mean:
            self.transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        files = sorted(
            [
                x for x in self.root.rglob("*")
                if x.suffix.lower()
                in self.IMG_EXTS
            ]
        )

        self.files = []
        skipped_files = []
        for path in files:
            with Image.open(path) as img:
                w, h = img.size
            if w < self.crop_multiple or h < self.crop_multiple:
                skipped_files.append(path)
            else:
                self.files.append(path)

        if skipped_files:
            max_examples = 5
            examples = ", ".join(str(path) for path in skipped_files[:max_examples])
            remaining = len(skipped_files) - max_examples
            suffix = f", and {remaining} more" if remaining > 0 else ""
            warnings.warn(
                f"Skipped {len(skipped_files)} image(s) with width or height smaller "
                f"than crop_multiple={self.crop_multiple}. Examples: {examples}{suffix}",
                stacklevel=2,
            )

        if not self.files:
            raise ValueError(
                f"No usable images found under {self.root}: images must have width and "
                f"height greater than or equal to crop_multiple={self.crop_multiple}."
            )



    def __len__(self):

        return len(self.files)



    def __getitem__(self,index):
        path=self.files[index]
        img=Image.open(path)\
                 .convert("RGB")
        img=self.center_crop_multiple(
            img,
            self.crop_multiple
        )
        tensor=self.to_tensor(img)
        tensor=self.transform(tensor) if self.zero_mean else tensor
        return {
            "img":tensor,
            "path":str(path)
        }



    ##################################
    # center crop to multiple
    ##################################

    def center_crop_multiple(
            self,
            img,
            size):


        w,h=img.size
        new_w=(w//size)*size
        new_h=(h//size)*size
        left=(w-new_w)//2
        top=(h-new_h)//2

        return img.crop(
            (
                left,
                top,
                left+new_w,
                top+new_h
            )
        )

    ##################################

    @staticmethod
    def to_tensor(img):
        arr=np.asarray(
            img,
            dtype=np.float32
        )/255.
        arr=torch.from_numpy(
            arr
        )
        return arr.permute(
            2,0,1
        )
