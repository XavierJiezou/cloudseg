# -*- coding: utf-8 -*-
# @Time    : 2024/8/3 上午10:46
# @Author  : xiaoshun
# @Email   : 3038523973@qq.com
# @File    : wandb_vis.py
# @Software: PyCharm
import argparse
import os
import shutil
from glob import glob

import albumentations as albu
import numpy as np
import torch
import torchvision
import wandb
from PIL import Image
from matplotlib import pyplot as plt
from rich.progress import track
from thop import profile

from src.data.components.hrcwhu import HRCWHU
from src.data.hrcwhu_datamodule import HRCWHUDataModule
from src.models.components.cdnetv1 import CDnetV1
from src.models.components.cdnetv2 import CDnetV2
from src.models.components.dual_branch import Dual_Branch
from src.models.components.hrcloud import HRcloudNet
from src.models.components.mcdnet import MCDNet
from src.models.components.scnn import SCNNNet


class WandbVis:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.colors = ((255, 255, 255), (128, 192, 128))
        self.num_classes = 2
        self.model = self.load_model()
        self.dataloader = self.load_dataset()
        self.macs, self.params = None, None
        wandb.init(project='model_vis', name=self.model_name)

    def load_weight(self, weight_path: str):
        weight = torch.load(weight_path, map_location=self.device)
        state_dict = {}
        for key, value in weight["state_dict"].items():
            new_key = key[4:]
            state_dict[new_key] = value
        return state_dict

    def load_model_by_model_name(self):
        if self.model_name == 'dual_branch':
            return Dual_Branch(img_size=256, in_channels=3, num_classes=2).to(self.device)
        if self.model_name == "cdnetv1":
            return CDnetV1(num_classes=2).to(self.device)
        if self.model_name == "cdnetv2":
            return CDnetV2(num_classes=2).to(self.device)

        if self.model_name == "hrcloud":
            return HRcloudNet(num_classes=2).to(self.device)
        if self.model_name == "mcdnet":
            return MCDNet(in_channels=3, num_classes=2).to(self.device)

        if self.model_name == "scnn":
            return SCNNNet(num_classes=2).to(self.device)

        raise ValueError(f"{self.model_name}模型不存在")

    def load_model(self):
        weight_path = glob(f"logs/train/runs/hrcwhu_{self.model_name}/*/checkpoints/*.ckpt")[0]
        model = self.load_model_by_model_name()
        state_dict = self.load_weight(weight_path)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def load_dataset(self):

        all_transform = albu.Compose(
            [
                albu.Resize(
                    height=HRCWHU.METAINFO["img_size"][1],
                    width=HRCWHU.METAINFO["img_size"][2],
                    always_apply=True
                )
            ]
        )
        img_transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )
        ann_transform = None
        val_pipeline = dict(
            all_transform=all_transform,
            img_transform=img_transform,
            ann_transform=ann_transform,
        )
        dataloader = HRCWHUDataModule(
            root="/home/liujie/liumin/cloudseg/data/hrcwhu",
            train_pipeline=val_pipeline,
            val_pipeline=val_pipeline,
            test_pipeline=val_pipeline,
            batch_size=1,
        )
        dataloader.setup()
        test_dataloader = dataloader.test_dataloader()
        return test_dataloader
        # for data in test_dataloader:
        #     print(data['img'].shape,data['ann'].shape,data['img_path'],data['ann_path'],data['lac_type'])
        #     break
        # torch.Size([1, 3, 256, 256])
        # torch.Size([1, 256, 256])
        # ['/home/liujie/liumin/cloudseg/data/hrcwhu/img_dir/test/barren_30.tif']
        # ['/home/liujie/liumin/cloudseg/data/hrcwhu/ann_dir/test/barren_30.tif']
        # ['barren']

    def give_colors_to_mask(self, mask: np.ndarray):
        """
        赋予mask颜色
        """
        assert len(mask.shape) == 2, "Value Error,mask的形状为(height,width)"
        colors_mask = np.zeros((mask.shape[0], mask.shape[1], 3)).astype(np.float32)
        for color in range(2):
            segc = (mask == color)
            colors_mask[:, :, 0] += segc * (self.colors[color][0])
            colors_mask[:, :, 1] += segc * (self.colors[color][1])
            colors_mask[:, :, 2] += segc * (self.colors[color][2])
        return colors_mask

    @torch.no_grad
    def pred_mask(self, x: torch.Tensor):
        self.macs, self.params = profile(self.model, inputs=(x, ))
        x = x.to(self.device)
        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        fake_mask = torch.argmax(logits, 1).detach().cpu().squeeze(0).numpy()
        return fake_mask

    def np_pil_np(self, image: np.ndarray, filename="colors_ann"):
        colors_np = self.give_colors_to_mask(image)
        pil_np = Image.fromarray(np.uint8(colors_np))
        return np.array(pil_np)

    def run(self, delete_wadb_log=True):
        # print(len(self.dataloader))
        # 30
        for data in track(self.dataloader):
            img = data["img"]
            ann = data["ann"].squeeze(0).numpy()
            img_path = data["img_path"]
            fake_mask = self.pred_mask(img)

            colors_ann = self.np_pil_np(ann)
            colors_fake = self.np_pil_np(fake_mask, "colors_fake")
            image_name = img_path[0].split(os.path.sep)[-1].split(".")[0]

            plt.subplot(1, 2, 1)
            plt.axis("off")
            plt.title("groud true")
            plt.imshow(colors_ann)

            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.title("predict mask")
            plt.imshow(colors_fake)
            wandb.log({image_name: wandb.Image(plt)})
        wandb.log({"macs":self.macs,"params":self.params})
        wandb.finish()
        if delete_wadb_log and os.path.exists("wandb"):
            shutil.rmtree("wandb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="dual_branch")
    parser.add_argument("--delete-wadb-log", type=bool, default=True)
    args = parser.parse_args()
    vis = WandbVis(model_name=args.model_name)
    vis.run(delete_wadb_log=args.delete_wadb_log)
