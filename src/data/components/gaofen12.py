import os
from glob import glob
from typing import Literal

import albumentations
import numpy as np
import tifffile as tf
from PIL import Image
from torch.utils.data import Dataset


class Gaofen12(Dataset):
    METAINFO = dict(
        classes=("clear sky", "cloud"),
        palette=((128, 192, 128), (255, 255, 255)),
        img_size=(384, 384),  # H, W
        ann_size=(384, 384),  # H, W
    )

    def __init__(
            self,
            root="/data/zouxuechao/cloudseg/gaofen12",
            phase: Literal["train", "val", "test"] = "train",
            serial: Literal["gaofen1", "gaofen2", "all"] = "all",
            all_transform: albumentations.Compose = None,
            img_transform: albumentations.Compose = None,
            ann_transform: albumentations.Compose = None,
    ) -> None:
        super().__init__()
        self.image_paths, self.mask_paths = self.__load_data(root, phase,serial)
        self.all_transform = all_transform
        self.img_transform = img_transform
        self.ann_transform = ann_transform

    def __load_data(self, root:str, phase:str, serial:str):
        filename = None
        if phase == "train":
            filename = "TrainBlock"
        elif phase == "val" or phase == "test":
            filename = "TestBlock"
        else:
            raise ValueError(
                "phase must be one of 'train','val','test', but got {}".format(phase)
            )
        if serial == "all":
            serial = "*"
        elif serial == "gaofen1":
            serial = "GF1MS-WHU"
        elif serial == "gaofen2":
            serial = "GF2MS-WHU"
        else:
            raise ValueError("serial must be one of 'gaofen1','gaofen2','all', but got {}".format(serial))
        mask_paths = glob(os.path.join(root, f"{serial}", f"*{filename}*", "*.tif"))
        image_paths = [
            filename.replace("_Mask", "").replace("tif", "tiff")
            for filename in mask_paths
        ]
        return image_paths, mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = tf.imread(image_path).transpose(1, 2, 0)
        mask = np.array(Image.open(mask_path))

        if self.all_transform:
            transformed = self.all_transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        if self.img_transform:
            image = self.img_transform(image=image)["image"]
        if self.ann_transform:
            mask = self.ann_transform(image=mask)["image"]

        return {"img": image, "ann": np.int64(mask), "img_path": image_path}


if __name__ == "__main__":
    import albumentations as albu
    from albumentations.pytorch.transforms import ToTensorV2

    # all_transform = transforms.Compose([
    #     transforms.RandomCrop((256, 256)),
    # ])
    all_transform = albu.Compose([albu.RandomCrop(250, 250)])

    img_transform = albu.Compose([albu.ToFloat(), ToTensorV2()])
    # img_transform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])

    # ann_transform = transforms.Compose([
    #     transforms.PILToTensor(),
    # ])
    for serial in ["all", "gaofen1", "gaofen2", "all"]:

        train_dataset = Gaofen12(
            phase="train",
            serial=serial,
            all_transform=all_transform,
            img_transform=img_transform,
            ann_transform=None,
        )

        for image_path, mask_path in zip(
                train_dataset.image_paths, train_dataset.mask_paths
        ):
            assert os.path.exists(image_path) and os.path.exists(
                mask_path
            ), f"{image_path} or {mask_path} not exists"
            assert int(image_path.split(os.path.sep)[-1].split(".")[0]) == int(
                mask_path.split(os.path.sep)[-1].split("_Ma")[0]
            ), f"{image_path} nor equal {mask_path}"
            assert os.path.sep.join(image_path.split(os.path.sep)[:-1]) == os.path.sep.join(
                mask_path.split(os.path.sep)[:-1]
            ), f"{image_path} nor equal {mask_path}"

        test_dataset = Gaofen12(
            phase="test",
            serial=serial,
            all_transform=all_transform,
            img_transform=img_transform,
            ann_transform=None,
        )
        for image_path, mask_path in zip(test_dataset.image_paths, test_dataset.mask_paths):
            assert os.path.exists(image_path) and os.path.exists(
                mask_path
            ), f"{image_path} or {mask_path} not exists"
            assert int(image_path.split(os.path.sep)[-1].split(".")[0]) == int(
                mask_path.split(os.path.sep)[-1].split("_Ma")[0]
            ), f"{image_path} nor equal {mask_path}"
            assert os.path.sep.join(image_path.split(os.path.sep)[:-1]) == os.path.sep.join(
                mask_path.split(os.path.sep)[:-1]
            ), f"{image_path} nor equal {mask_path}"

        # assert len(train_dataset) == train_dataset.METAINFO["train_size"]
        # assert len(test_dataset) == test_dataset.METAINFO["test_size"]

        train_sample = train_dataset[0]
        test_sample = test_dataset[0]

        print(train_sample["img"].shape, train_sample["ann"].shape)
        print(f"size: {len(train_dataset)}, size: {len(test_dataset)}")
