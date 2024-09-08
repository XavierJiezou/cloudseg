import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as albu
import numpy as np
from torch.utils.data import Dataset
import numpy as np
from typing import Literal, List, Tuple
import matplotlib.pyplot as plt
from src.utils.stretch import gaussian_stretch


class CloudSEN12High(Dataset):
    METAINFO = dict(
        classes=("clear", "thick cloud", "thin cloud", "cloud shadow"),
        palette=((31, 119, 18), (255, 127, 14), (44, 160, 44), (214, 39, 40)),
        img_size=(512, 512),
        ann_size=(512, 512),
        train_size=8490,
        val_size=535,
        test_size=975,
    )

    def __init__(
        self,
        root: str = "data/cloudsen12_high",
        phase: Literal["train", "val", "test"] = "train",
        level: Literal["l1c", "l2a"] = "l1c",
        bands: List[str] = ["B4", "B3", "B2"],
        all_transform: albu.Compose = None,
        img_transform: albu.Compose = None,
        ann_transform: albu.Compose = None,
    ):
        self.root = root
        self.phase = phase
        self.level = level
        self.bands = bands
        self.all_transform = all_transform
        self.img_transform = img_transform
        self.ann_transform = ann_transform
        self.image_data, self.label_data = self.load_data()

    def load_data(self):
        image_data = self.__load_image()
        label_data = self.__load_label()
        return image_data, label_data
    
    def __load_label(self) -> np.ndarray:
        label_path = os.path.join(self.root, self.phase, "LABEL_manual_hq.dat")
        label_data = np.memmap(filename=label_path, dtype=np.int8, mode="r").reshape(-1, *self.METAINFO["ann_size"])
        label_data = label_data.astype(np.int64)
        return label_data
    
    def __load_image(self) -> np.ndarray:
        image_data = {
            band: self.__load_image_by_band(band) for band in self.bands
        }
        return image_data

    def __load_image_by_band(self, band) -> Tuple[np.ndarray, np.ndarray]:
        if "S1" in band:
            image_path = os.path.join(self.root, self.phase, band) + ".dat"
            dtype = np.float32
        else:
            image_path = os.path.join(self.root, self.phase, "_".join([self.level.upper(), band])) + ".dat"
            dtype = np.int16
        image = np.memmap(filename=image_path, dtype=dtype, mode="r").reshape(-1, *self.METAINFO["img_size"])
        return image

    def __len__(self) -> int:
        return len(self.label_data)

    def __getitem__(self, idx):
        img = np.stack([self.image_data[band][idx] for band in self.bands])
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6) # 1e-6 to avoid division by zero
        ann = self.label_data[idx]
        
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
            "ann": ann,
        }


def test_cloudsen12_high():
    for phase in ["train", "val", "test"]:
        dataset = CloudSEN12High(phase=phase)
        assert len(dataset)==CloudSEN12High.METAINFO[f"{phase}_size"]
        assert dataset[0]["img"].shape == (len(["B4", "B3", "B2"]), *CloudSEN12High.METAINFO["ann_size"])
        assert dataset[0]["ann"].shape == CloudSEN12High.METAINFO["ann_size"]


def show_cloudsen12_high():
    all_transform = None
    
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1)
    plt.title("L1C")
    plt.axis("off")
    dataset = CloudSEN12High(phase="train", level="l1c", bands=["B4", "B3", "B2"], all_transform=all_transform)
    data = dataset[-1]["img"]
    data = (np.transpose(data, (1, 2, 0)) * 255).astype(np.uint8)
    data = gaussian_stretch(data)
    plt.imshow(data)
    
    plt.subplot(1, 4, 2)
    plt.title("L2A")
    plt.axis("off")
    dataset = CloudSEN12High(phase="train", level="l2a", bands=["B4", "B3", "B2"], all_transform=all_transform)
    data = dataset[-1]["img"]
    data = (np.transpose(data, (1, 2, 0)) * 255).astype(np.uint8)
    data = gaussian_stretch(data)
    plt.imshow(data)
    
    plt.subplot(1, 4, 3)
    plt.title("SAR")
    plt.axis("off")
    dataset = CloudSEN12High(phase="train", level="l1c", bands=["S1_VV", "S1_VH"], all_transform=all_transform)
    data = dataset[-1]["img"]
    new_channel = (data[0] + data[1]) / 2
    data = np.stack([data[0], data[1], new_channel])
    data = (np.transpose(data, (1, 2, 0)) * 255).astype(np.uint8)
    data = gaussian_stretch(data)
    plt.imshow(data)
    
    plt.subplot(1, 4, 4)
    plt.title("ANN")
    plt.axis("off")
    ann = dataset[-1]["ann"]
    plt.imshow(ann)
    
    plt.savefig("cloudsen12_high.png", bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    test_cloudsen12_high()
    show_cloudsen12_high()
