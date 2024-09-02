import os

import albumentations
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from natsort import natsorted
from typing import List, Literal


class L8Biome(Dataset):
    METAINFO = dict(
        classes=("fill", "cloud shadow", "clear", "thin cloud", "cloud"),
        palette=(
            (31, 119, 180),
            (255, 127, 14),
            (44, 160, 44),
            (214, 39, 40),
            (148, 103, 189),
        ),
        img_size=(512, 512),  # H, W
        ann_size=(512, 512),  # H, W
    )

    def __init__(
        self,
        root: str = None,
        bands: List[str] = ["2", "3", "4"],
        phase: Literal["train", "test", "val"] = "train",
        all_transform: albumentations.Compose = None,
        img_transform: albumentations.Compose = None,
        ann_transform: albumentations.Compose = None,
    ):
        self.root = root
        self.phase = phase
        self.bands = bands
        self.all_transform = all_transform
        self.img_transform = img_transform
        self.ann_transform = ann_transform
        self.data = self.load_data()

    def load_data(self):
        dirs = os.listdir(os.path.join(self.root, "img_crop"))
        dirs = natsorted(dirs)
        length = len(dirs)

        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1
        
        train_end = int(length * train_ratio)
        val_end = train_end + int(length * val_ratio)

        if self.phase == "train":
            dirs = dirs[: train_end]
        elif self.phase == "val":
            dirs = dirs[train_end :val_end]
        elif self.phase == "test":
            dirs = dirs[val_end :]
        else:
            raise ValueError(
                "phase must be train, val or test,but got {}".format(self.phase)
            )

        assert len(dirs) > 0, "No data found in {}".format(self.root)
        images_path = os.path.join(
            self.root, "img_crop", dirs[0], f"{dirs[0]}*_B{self.bands[0]}*.tif"
        )

        file_indexs = []
        for image_path in glob(images_path):
            filename = image_path.split(os.path.sep)[-2]
            index = (
                image_path.split(os.path.sep)[-1]
                .split(filename)[-1]
                .split(".")[0]
                .split(f"B{self.bands[0]}_")[-1]
            )
            file_indexs.append(index)
        data_list = []
        for dir in dirs:
            img_path = os.path.join(self.root, "img_crop", dir)
            filename = dir
            for index in file_indexs:
                data_list.append([img_path, filename, index])
        return data_list

    def __len__(self):
        return len(self.data)

    def __get_img_ann(self, img_path: str, filename: str, index: str):
        image = None
        for bands in self.bands:
            image_path = os.path.join(
                img_path, f"{filename}_B{bands[0]}_{index}.tif"
            )
            im = np.array(Image.open(image_path))[:,:,np.newaxis]
            if image is None:
                image = im
            else:
                image = np.concatenate((image, im), axis=-1)

        ann_path = os.path.join(
            img_path, f"{filename}_fixedmask_{index}.tif"
        ).replace("img_crop", "seg_crop")

        ann = np.array(Image.open(ann_path))
        return image, ann

    def __getitem__(self, idx):
        img_path, filename, index = self.data[idx]
        img, ann = self.__get_img_ann(img_path, filename, index)

        if self.all_transform:
            albumention = self.all_transform(image=img, mask=ann)
            img = albumention["image"]
            ann = albumention["mask"]

        if self.img_transform:
            img = self.img_transform(image=img)["image"]

        if self.ann_transform:
            ann = self.ann_transform(image=img)["image"]

        return {
            "img": img,
            "ann": np.int64(ann),
            "img_path": filename,
        }


if __name__ == "__main__":
    root = "/data/zouxuechao/cloudseg/l8_biome/BC"
    dataset = L8Biome(root=root)
    data = dataset[0]
    print(data['img'].shape,data['ann'].shape)
