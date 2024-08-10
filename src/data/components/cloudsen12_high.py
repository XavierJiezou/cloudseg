import os

import albumentations as albu
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from glob import glob
from typing import Literal, List, Tuple


class CloudSEN12High(Dataset):
    METAINFO = dict(
        classes=("clear", "thick cloud", "thin cloud", "cloud shadow"),
        palette=((31, 119, 18), (255, 127, 14), (44, 160, 44), (214, 39, 40)),
        img_size=(3, 512, 512),  # C, H, W
        ann_size=(512, 512),  # C, H, W
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
        self.channel_data, self.channel_type, self.label_data = self.load_data()
        self.eps = 1e-6 #最大最小归一化时防止分母为0

    def load_data(self):
        channel_list = []
        channel_type_list = []
        for bands in self.bands:
            level_name = None if bands.startswith("S1_") else self.level
            data, data_type = self.__load_channel_data(level_name, bands)
            channel_list.append(data)
            channel_type_list.append(data_type)
        label_path = os.path.join(self.root, self.phase, "LABEL_manual_hq.dat")
        label_shape = self.__get_file_shape()
        label_data = np.memmap(label_path, dtype=np.int8, mode="r", shape=label_shape)

        return channel_list, channel_type_list, label_data

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
        file_type = np.int16 if level_name else np.float32
        file_shape = self.__get_file_shape()
        file = np.memmap(filename=filename, dtype=file_type, mode="r", shape=file_shape)
        return file, file_type

    def __get_file_shape(self) -> Tuple[int, int, int]:
        """得到数据形状"""
        key = f"{self.phase}_size"
        if key in self.METAINFO:
            return (self.METAINFO[f"{self.phase}_size"], 512, 512)

        raise ValueError(
            f"不存在的phase.phase应当是'train'或'val'或'test',实际却是{self.phase}."
        )

    def __len__(self) -> int:
        key = f"{self.phase}_size"
        if key in self.METAINFO:
            return self.METAINFO[key]

        raise ValueError(
            f"不存在的phase.phase应当是'train'或'val'或'test',实际却是{self.phase}."
        )

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
                num = (2 ** 16) - 1
                image = image / num
            elif images_type[i] == np.float32:
                max_num = image.max()
                min_num = image.min()
                image = (image - min_num) / (max_num - min_num + self.eps)
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
        
        img = self.__to_tensor(img,self.channel_type)

        return {
            "img": img,
            "ann": np.int64(ann),
            "lac_type": np.int8,
        }


if __name__ == "__main__":
    bands = ["B1", "B2", "B3", "B4", "B5","S1_VV","S1_angle","S1_VH"]
    
    all_transform = albu.Compose([
        albu.RandomCrop(512,512),
        albu.RandomGamma()
    ])
    train_set = CloudSEN12High(phase='train',level='l2a',bands=bands,all_transform=all_transform)
    train_data = train_set[0]
    print(train_data['img'].shape,train_data['ann'].shape)

    test_set = CloudSEN12High(phase='test',level='l2a',bands=bands,all_transform=all_transform)
    test_data = test_set[0]
    print(test_data['img'].shape,test_data['ann'].shape)


    val_set = CloudSEN12High(phase='val',level='l2a',bands=bands,all_transform=all_transform)
    val_data = val_set[0]
    print(val_data['img'].shape,val_data['ann'].shape)

    # (8, 512, 512) (512, 512)
    # (8, 512, 512) (512, 512)
    # (8, 512, 512) (512, 512)
