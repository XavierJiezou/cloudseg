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
        self.filenames = self.load_data()

    def __read_txt(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            filenames = f.read().split("\n")[:-1]
        return filenames

    def load_data(self):
        assert self.phase in [
            "train",
            "val",
            "test",
        ], f"phase must is 'train','val','test',but got {self.phase}"
        file_path = os.path.join(self.root, f"{self.phase}.txt")
        filenames = self.__read_txt(file_path)
        return filenames

    def __len__(self):
        return len(self.filenames)

    def __process_ann(self, ann: np.ndarray):
        """
        对ann中的像素做一个映射
        """
        ann[ann == 64] = 1
        ann[ann == 128] = 2
        ann[ann == 192] = 3
        ann[ann == 255] = 4
        return ann

    def __normalize_image(self, image: np.ndarray):
        max_val = np.max(image)
        min_val = np.min(image)
        image = np.transpose(image, (2, 0, 1))
        if max_val == 0 and min_val == 0:
            return image.astype(np.float32)
        image = (image - min_val) / (max_val - min_val)
        image = image.astype(np.float32)
        return image

    def __get_ann_path(self, filename):
        filename = filename.split(".")[0]
        filename = filename + f"_fixedmask.tif"
        return filename

    def __get_img_ann(self, filename: str):
        image = None
        base_name = os.path.basename(filename).split(".")[0]
        for bands in self.bands:
            image_path = os.path.join(
                self.root, "img_dir", base_name + f"_B{bands}.tif"
            )
            im = np.array(Image.open(image_path))[:, :, np.newaxis]
            if image is None:
                image = im
            else:
                image = np.concatenate((image, im), axis=-1)

        ann_path = os.path.join(self.root, "ann_dir", self.__get_ann_path(filename))
        ann = np.array(Image.open(ann_path))
        ann = self.__process_ann(ann)
        return image, ann

    def __get_scene(self, filename):
        return filename.split("_")[0]

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img, ann = self.__get_img_ann(filename)
        scene = self.__get_scene(filename)

        if self.all_transform:
            albumention = self.all_transform(image=img, mask=ann)
            img = albumention["image"]
            ann = albumention["mask"]

        if self.img_transform:
            img = self.img_transform(image=img)["image"]

        if self.ann_transform:
            ann = self.ann_transform(image=img)["image"]

        img = self.__normalize_image(img)
        return {"img": img, "ann": np.int64(ann), "img_path": filename, "scene": scene}


if __name__ == "__main__":
    root = "/data/zouxuechao/cloudseg/l8_biome"
    dataset = L8Biome(root=root)
    data = dataset[0]
    print(data["img"].shape, data["ann"].shape, data["img_path"], data["scene"])
    for phase in ["train", "val", "test"]:
        dataset = L8Biome(root=root, phase=phase)
        print(f"{phase}:{len(dataset)}")
