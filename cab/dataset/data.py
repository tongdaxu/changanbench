from torchvision.datasets import VisionDataset
from PIL import Image
from glob import glob
import torchvision.transforms.v2 as transforms

class SimpleDataset(VisionDataset):
    def __init__(self, root: str, image_size: int, zero_mean: bool):
        super().__init__(root)
        self.zero_mean = zero_mean
        transform_list = [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(), 
            ]
        if self.zero_mean:
            transform_list.append(
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )
        self.transform = transforms.Compose(transform_list)


        if root.endswith(".txt"):
            with open(root) as f:
                lines = f.readlines()
            self.fpaths = [line.strip("\n") for line in lines]
        else:
            self.fpaths = sorted(glob(root + "/**/*.JPEG", recursive=True))
            self.fpaths += sorted(glob(root + "/**/*.jpg", recursive=True))
            self.fpaths += sorted(glob(root + "/**/*.png", recursive=True))

        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert("RGB")
        img = self.transform(img)
        return {
            "img": img,
            "fpath": fpath,
        }