from pathlib import Path
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms


class CompanyImageDataset(Dataset):

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
        patch_size=512
    ):

        self.root = Path(root)
        self.mode = mode
        self.patch_size = patch_size
        self.zero_mean = zero_mean
        if self.zero_mean:
            self.transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.files = sorted(
            [
                x for x in self.root.rglob("*")
                if x.suffix.lower()
                in self.IMG_EXTS
            ]
        )

        assert len(self.files)>0



    def __len__(self):

        return len(self.files)



    def __getitem__(self,index):
        path=self.files[index]
        img=Image.open(path)\
                 .convert("RGB")
        img=self.center_crop_multiple(
            img,
            self.patch_size
        )

        

        if self.mode=="full":
            tensor=self.to_tensor(img)
            tensor=self.transform(tensor) if self.zero_mean else tensor
            return {
                "img":tensor,
                "path":str(path),
                "mode":"full"

            }

        elif self.mode=="patch":
            patches,meta=self.make_patches(
                img
            )
            patches=self.transform(patches) if self.zero_mean else patches
            return {
                "patches":patches,
                "path":str(path),
                "mode":"patch",
                "meta":meta
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
    # split 512 patches
    ##################################

    def make_patches(
            self,
            img):
        w,h=img.size
        nx=w//self.patch_size
        ny=h//self.patch_size

        patches=[]

        for y in range(ny):
            for x in range(nx):
                patch=img.crop(
                    (
                        x*self.patch_size,
                        y*self.patch_size,
                        (x+1)*self.patch_size,
                        (y+1)*self.patch_size
                    )
                )

                patches.append(
                    self.to_tensor(
                        patch
                    )
                )


        patches=torch.stack(
            patches,
            dim=0
        )


        meta={
            "grid_w":nx,
            "grid_h":ny,
            "height":h,
            "width":w

        }


        return patches,meta




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