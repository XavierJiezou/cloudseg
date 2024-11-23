from pathlib import Path

import albumentations
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Cloud38(Dataset):
    METAINFO = dict(
        classes=("clear sky", "cloud"),
        palette=((128, 192, 128), (255, 255, 255)),
        img_size=(384, 384),  # H, W
        ann_size=(384, 384),  # H, W
        train_size=5800,
        test_size=2600,
    )

    def __init__(
        self,
        root="/data/zouxuechao/cloudseg/38-cloud/38-Cloud_training",
        all_transform: albumentations.Compose = None,
        img_transform: albumentations.Compose = None,
        ann_transform: albumentations.Compose = None,
    ):
        self.root = Path(root)
        r_dir, g_dir, b_dir, nir_dir, gt_dir = (
            self.root / "train_red",
            self.root / "train_green",
            self.root / "train_blue",
            self.root / "train_nir",
            self.root / "train_gt",
        )
        self.files = [
            self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir)
            for f in r_dir.iterdir()
            if not f.is_dir()
        ]
        self.all_transform = all_transform
        self.img_transform = img_transform
        self.ann_transform = ann_transform

    def combine_files(self, r_file: Path, g_dir, b_dir, nir_dir, gt_dir):

        files = {
            "red": r_file,
            "green": g_dir / r_file.name.replace("red", "green"),
            "blue": b_dir / r_file.name.replace("red", "blue"),
            "nir": nir_dir / r_file.name.replace("red", "nir"),
            "gt": gt_dir / r_file.name.replace("red", "gt"),
        }

        return files

    def open_mask(self, idx):

        raw_mask = np.array(Image.open(self.files[idx]["gt"]))
        raw_mask = np.where(raw_mask == 255, 1, 0)

        return raw_mask

    def open_as_array(self, idx, include_nir=True):
        raw_rgb = np.stack(
            [
                np.array(Image.open(self.files[idx]["red"])),
                np.array(Image.open(self.files[idx]["green"])),
                np.array(Image.open(self.files[idx]["blue"])),
            ],
            axis=2,
        )

        if include_nir:
            nir = np.expand_dims(np.array(Image.open(self.files[idx]["nir"])), axis=2)
            raw_rgb = np.concatenate([raw_rgb, nir], axis=2)

        return raw_rgb

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        img = self.open_as_array(idx)
        ann = self.open_mask(idx)


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
            "img_path": str(self.files[idx]['red']),
            "ann_path": str(self.files[idx]["gt"]),
        }


if __name__ == "__main__":
    import albumentations as albu
    from albumentations.pytorch.transforms import ToTensorV2
    import torch
    # all_transform = transforms.Compose([
    #     transforms.RandomCrop((256, 256)),
    # ])
    all_transform = albu.Compose([
        albu.RandomCrop(384,384)
    ])
    
    img_transform = albu.Compose([
        albu.ToFloat(),
        ToTensorV2()
    ])
    # img_transform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])

    # ann_transform = transforms.Compose([
    #     transforms.PILToTensor(),
    # ])
    train_dataset = Cloud38(
        all_transform=all_transform,
        img_transform=img_transform,
        ann_transform=None,
    )
    test_dataset = Cloud38(
        all_transform=all_transform,
        img_transform=img_transform,
        ann_transform=None,
    )

    # assert len(train_dataset) == train_dataset.METAINFO["train_size"]
    # assert len(test_dataset) == test_dataset.METAINFO["test_size"]

    train_sample = train_dataset[0]
    test_sample = test_dataset[0]

    assert (
        train_sample["img"].shape[1:]
        == test_sample["img"].shape[1:]
        == train_dataset.METAINFO["img_size"]
    )
    assert (
        train_sample["ann"].shape
        == test_sample["ann"].shape
        == train_dataset.METAINFO["ann_size"]
    )

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for train_sample in train_dataset:
        axs[0].imshow(train_sample["img"].permute(1, 2, 0))
        axs[0].set_title("Image")
        axs[1].imshow(
            torch.tensor(train_dataset.METAINFO["palette"])[train_sample["ann"]]
        )
        axs[1].set_title("Annotation")
        plt.suptitle('Land Cover Type', y=0.8)
        plt.tight_layout()
        plt.savefig("38cloud_sample.png", bbox_inches="tight")
        # break
