import os

import albumentations as albu
import numpy as np
from torch.utils.data import Dataset
import numpy as np
from glob import glob
from typing import Literal, List, Tuple


class CloudSEN12High(Dataset):
    METAINFO = dict(
        classes=("clear", "thick cloud", "thin cloud", "cloud shadow"),
        palette=((31, 119, 18), (255, 127, 14), (44, 160, 44), (214, 39, 40)),
        ann_size=(512, 512),
        train_size=8490,
        val_size=535,
        test_size=975,
    )

    def __init__(
        self,
        root: str = "/data/zouxuechao/cloudseg/cloudsen12_high",
        phase: Literal["train", "val", "test"] = "train",
        level: Literal["l1c", "l2a"] = "l1c",
        bands: List[str] = None,  # ["b02", "b03", "b04", "b08", "vv", "vh", "angle"]
        all_transform: albu.Compose = None,
        img_transform: albu.Compose = None,
        ann_transform: albu.Compose = None,
    ):
        self.root = root
        self.phase = phase
        self.level = level.upper()
        self.bands = bands
        self.all_transform = all_transform
        self.img_transform = img_transform
        self.ann_transform = ann_transform
        self.channel_data, self.channel_dtype, self.label_data = self.load_data()
        self.eps = 1e-6 #最大最小归一化时防止分母为0

    def load_data(self):
        channel_list = []
        channel_dtype_list = []
        for bands in self.bands:
            level_name = None if bands.startswith("S1_") else self.level
            data, data_type = self.__load_channel_data(level_name, bands)
            channel_list.append(data)
            channel_dtype_list.append(data_type)
        label_path = os.path.join(self.root, self.phase, "LABEL_manual_hq.dat")
        label_data = np.memmap(filename=label_path, dtype=np.int8, mode="r").reshape(-1, *self.METAINFO["ann_size"])
        return channel_list, channel_dtype_list, label_data

    def __load_channel_data(
        self, level_name: str = None, bond: str = None
    ) -> Tuple[np.ndarray, type]:
        """根据处理数据的方式和数据的波段加载数据，返回这个波段的数据和数据类型

        Args:
            level_name (str): 数据处理方式，可以不传，这时表示使用的是S1_angle等.
            bond (str): 波段名称.
        """
        # /data/zouxuechao/cloudseg/cloudsen12_high/train/L2A_B12.dat
        filename = ""
        if level_name:
            filename = os.path.join(self.root, self.phase, "_".join([level_name, bond]))
        else:
            filename = os.path.join(self.root, self.phase, bond)
        filename = filename + ".dat"
        file_dtype = np.int16 if level_name else np.float32
        file = np.memmap(filename=filename, dtype=file_dtype, mode="r").reshape(-1, *self.METAINFO["ann_size"])
        return file, file_dtype

    def __len__(self) -> int:
        return self.label_data.shape[0]

    def __to_tensor(self,images:np.ndarray,images_type:List[type])->np.ndarray:
        """对数据进行归一化操作，并将数据调整为(c,h,w)
        
        Args:
            images (np.ndarray): 输入的图片，形状应为(h,w,c)
            images_type (List[type]): 列表，记录了各个通道的数据类型
        """
        norm_images = None
        for i in range(len(images_type)):
            image = images[:,:,i][:,:,np.newaxis]
            if images_type[i] == np.int16:
                image = image / (2 ** 16 - 1)
            elif images_type[i] == np.float32:
                image = (image - image.min()) / (image.max() - image.min() + self.eps)
            else:
                raise ValueError(f"意外的数据类型:{images_type[i]}")
            if norm_images is None:
                norm_images = image
            else:
                norm_images = np.concatenate([norm_images,image],axis=-1)
        norm_images = np.transpose(norm_images,(2,0,1))
        return norm_images

    def __getitem__(self, idx):
        ann = self.label_data[idx]
        img = None
        for i in range(len(self.channel_data)):
            channel_data = self.channel_data[i][idx][:,:,np.newaxis]
            if img is None:
                img = channel_data
            else:
                img = np.concatenate([img,channel_data],axis=-1)
            

        if self.all_transform:
            albumention = self.all_transform(image=img, mask=ann)
            img = albumention["image"]
            ann = albumention["mask"]

        if self.img_transform:
            img = self.img_transform(image=img)["image"]

        if self.ann_transform:
            ann = self.ann_transform(image=img)["image"]
        
        img = self.__to_tensor(img,self.channel_dtype)
        

        return {
            "img": img,
            "ann": np.int64(ann),
            "lac_type": np.int8,
        }


if __name__ == "__main__":
    bands = ["B4", "B3", "B2"] # RGB
    
    all_transform = albu.Compose([
        albu.RandomCrop(512,512),
    ])

    for phase in ["train", "val", "test"]:
        dataset = CloudSEN12High(phase=phase, level='l2a', bands=bands, all_transform=all_transform)
        assert len(dataset)==CloudSEN12High.METAINFO[f"{phase}_size"]
        assert dataset[0]["img"].shape == (len(bands), *CloudSEN12High.METAINFO["ann_size"])
        assert dataset[0]["ann"].shape == CloudSEN12High.METAINFO["ann_size"]

    
    import matplotlib.pyplot as plt
    from src.utils.stretch import linear_stretch
    
    dataset = CloudSEN12High(phase="train", level='l1c', bands=bands, all_transform=all_transform)
    data = dataset[-1]['img']
    data = (np.transpose(data,(1,2,0))*255).astype(np.uint8)
    data = linear_stretch(data)

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.title("L1C_RGB")
    plt.axis("off")
    plt.imshow(data)

    # L1C L2A S1 ANN
    plt.subplot(1,4,2)

    dataset = CloudSEN12High(phase="train", level="l2a", bands=bands, all_transform=all_transform)
    data = dataset[-1]['img']
    data = (np.transpose(data,(1,2,0))*255).astype(np.uint8)
    data = linear_stretch(data)

    plt.title("L2A_RGB")
    plt.axis("off")
    plt.imshow(data)

    plt.subplot(1,4,3)

    dataset = CloudSEN12High(phase="train", level="l1c", bands=["S1_VV","S1_VH","S1_angle"], all_transform=all_transform)
    data = dataset[-1]['img']
    data = (np.transpose(data,(1,2,0))*255).astype(np.uint8)
    data = linear_stretch(data)


    plt.title("S1")
    plt.axis("off")
    plt.imshow(data)

    plt.subplot(1,4,4)

    ann = dataset[-1]['ann']

    plt.axis("off")
    plt.title("ANN")
    plt.imshow(ann, cmap="gray")
    plt.savefig("cloudsen12_high.png", bbox_inches="tight", pad_inches=0)
