from pathlib import Path
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset


class CompanyImageDataset(Dataset):
    """Company RGB images center-cropped to a multiple of 512.

    Dataset output is always the complete cropped image; model-specific
    patching belongs to the codec.
    """

    IMG_EXTS = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    ]
    CROP_MULTIPLE = 512


    def __init__(
        self,
        root,
        zero_mean=None,
    ):
        self.root = Path(root)
        self.zero_mean = zero_mean

        self.files = sorted(
            [
                x for x in self.root.rglob("*")
                if x.suffix.lower()
                in self.IMG_EXTS
            ]
        )

        if not self.files:
            raise ValueError(f"No supported images found under {self.root}")



    def __len__(self):

        return len(self.files)



    def __getitem__(self,index):
        path=self.files[index]
        with Image.open(path) as source:
            img = source.convert("RGB")
            img = self.center_crop_multiple(img, self.CROP_MULTIPLE)
            tensor = self.to_tensor(img)

        if self.zero_mean:
            tensor = tensor.mul(2.0).sub(1.0)

        return {
            "img": tensor,
            "path": str(path),
        }



    ##################################
    # center crop to multiple
    ##################################

    def center_crop_multiple(
            self,
            img,
            size):


        w,h=img.size
        if w < size or h < size:
            raise ValueError(
                f"Image size {w}x{h} is smaller than the required "
                f"{size}x{size} crop."
            )
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
