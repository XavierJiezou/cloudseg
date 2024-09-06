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

    def __get_dirs(self):
        assert self.phase in ["train","val","test"], f"phase must is 'train','val','test',but got {self.phase}"
        filename = os.path.join(self.root,f"{self.phase}.txt")
        with open(filename,"r",encoding="utf-8") as f:
            dirs = f.read().split("\n")[:-1]
        return dirs

    def load_data(self):
        dirs = self.__get_dirs()
        assert len(dirs) > 0, "No data found in {}".format(self.root)
        images_path = os.path.join(
            self.root, "img_dir", dirs[0], f"{dirs[0].split('_')[1]}*_B{self.bands[0]}*.tif"
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
            img_path = os.path.join(self.root, "img_dir", dir)
            filename = dir
            for index in file_indexs:
                data_list.append([img_path, filename, index])
        return data_list

    def __len__(self):
        return len(self.data)
    
    def __process_ann(self,ann:np.ndarray):
        """
        对ann中的像素做一个映射
        """
        ann[ann == 64] = 1
        ann[ann == 128] = 2
        ann[ann == 192] = 3
        ann[ann == 255] = 4
        return ann
    
    def __normalize_image(self,image:np.ndarray):
        max_val = np.max(image)
        min_val = np.min(image)
        image = np.transpose(image,(2,0,1))
        if max_val == 0 and min_val == 0:
            return image.astype(np.float32)
        image = (image - min_val) / (max_val - min_val)
        image = image.astype(np.float32)
        return image
    
    def __get_mask_path(self,img_path):
        path_list = img_path.split(os.path.sep)
        path_list[-1] = path_list[-1].split("_")[1]
        return os.path.sep + os.path.join(*path_list)

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
        ).replace("img_dir", "ann_dir")
        ann = np.array(Image.open(ann_path))
        ann = self.__process_ann(ann)
        return image, ann

    def __getitem__(self, idx):
        img_path, filename, index = self.data[idx]
        scene,filename = filename.split("_")
        img, ann = self.__get_img_ann(img_path, filename, index)

        if self.all_transform:
            albumention = self.all_transform(image=img, mask=ann)
            img = albumention["image"]
            ann = albumention["mask"]

        if self.img_transform:
            img = self.img_transform(image=img)["image"]

        if self.ann_transform:
            ann = self.ann_transform(image=img)["image"]

        img = self.__normalize_image(img)
        return {
            "img": img,
            "ann": np.int64(ann),
            "img_path": filename,
            "scene":scene
        }


if __name__ == "__main__":
    root = "/data/zouxuechao/cloudseg/l8_biome"
    dataset = L8Biome(root=root)
    data = dataset[0]
    print(data['img'].shape,data['ann'].shape,data["img_path"],data["scene"])
    for phase in ["train","val","test"]:
        dataset = L8Biome(root=root,phase=phase)
        print(f"{phase}:{len(dataset)}")
