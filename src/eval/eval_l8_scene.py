from glob import glob
import argparse
from rich.table import Table
from typing import Tuple, Dict
from rich.progress import track
import json
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2

from collections import OrderedDict
from src.metrics.metric import IoUMetric
from torchmetrics.utilities.data import to_onehot
import albumentations as albu
import torch
from torch import nn as nn
from src.data.hrc_whu_datamodule import HRC_WHU
from src.data.hrc_whu_datamodule import HRC_WHUDataModule
from src.data.cloudsen12_high_datamodule import CloudSEN12HighDataModule
from src.data.gf12ms_whu_datamodule import GF12MSWHUDataModule
from src.data.l8_biome_crop_datamodule import L8BiomeCropDataModule
from src.models.components.cdnetv1 import CDnetV1
from src.models.components.cdnetv2 import CDnetV2
from src.models.components.dbnet import DBNet
from src.models.components.hrcloudnet import HRCloudNet
from src.models.components.kappamask import KappaMask
from src.models.components.mcdnet import MCDNet
from src.models.components.scnn import SCNN
from src.models.components.unetmobv2 import UNetMobV2
from src.utils.model_order import model_order


def get_args():
    parser = argparse.ArgumentParser(description="获取实验名称和使用的显卡信息")
    parser.add_argument(
        "--root", type=str, help="使用的设备", default="data/l8_biome_crop"
    )
    parser.add_argument("--gpu", type=str, help="使用的设备", default="cuda:0")

    args = parser.parse_args()
    return args.root, args.gpu


class Eval:
    def __init__(self, root: str, device: str):
        self.num_classes = 4
        self.device = device
        self.root = root
        self.image_size = 512
        self.models = {
            "cdnetv1": CDnetV1(num_classes=self.num_classes).to(self.device),
            "cdnetv2": CDnetV2(num_classes=self.num_classes).to(self.device),
            "hrcloudnet": HRCloudNet(num_classes=self.num_classes).to(self.device),
            "mcdnet": MCDNet(in_channels=3, num_classes=self.num_classes).to(
                self.device
            ),
            "scnn": SCNN(num_classes=self.num_classes).to(self.device),
            "dbnet": DBNet(
                img_size=self.image_size, in_channels=3, num_classes=self.num_classes
            ).to(self.device),
            "unetmobv2": UNetMobV2(num_classes=self.num_classes).to(self.device),
            "kappamask": KappaMask(num_classes=self.num_classes, in_channels=3).to(
                self.device
            ),
        }
        self.model_names_mapping = {
            "KappaMask": "kappamask",
            "CDNetv1": "cdnetv1",
            "CDNetv2": "cdnetv2",
            "HRCloudNet": "hrcloudnet",
            "MCDNet": "mcdnet",
            "SCNN": "scnn",
            "DBNet": "dbnet",
            "UNetMobv2": "unetmobv2",
        }
        self.model_names_mapping = {
            value: key for key, value in self.model_names_mapping.items()
        }

        self.__load_weight()
        self.val_dataloader = self.__load_data()

    def __load_weight(self):
        """
        将模型权重加载进来
        """
        for model_name, model in self.models.items():
            weight_path = glob(
                f"checkpoints/l8_biome/{model_name}.bin"
            )[0]
            weight = torch.load(weight_path, map_location=self.device)
            model.load_state_dict(weight)
            model.eval()

    def __get_data_module(self):
        train_pipeline = val_pipeline = test_pipeline = dict(
            all_transform=None,
            img_transform=albu.Compose([ToTensorV2()]),
            ann_transform=None,
        )
        return L8BiomeCropDataModule(
            root=self.root,
            train_pipeline=train_pipeline,
            val_pipeline=val_pipeline,
            test_pipeline=test_pipeline,
            batch_size=1,
        )

    @torch.no_grad()
    def inference(self, img: torch.Tensor, model: nn.Module) -> torch.Tensor:
        logits = model(img)
        if isinstance(logits, tuple):
            logits = logits[0]
        pred = torch.argmax(logits, 1).detach()
        return pred

    def __load_data(self):

        data_loader = self.__get_data_module()
        data_loader.prepare_data()

        data_loader.setup("test")
        val_dataloader = data_loader.test_dataloader()
        return val_dataloader

    def run(self):

        metric = IoUMetric(
            num_classes=4,
            iou_metrics=["mIoU", "mDice", "mFscore"],
        )

        scenes_cls = [
            "Grass/Crops",
            "Urban",
            "Wetlands",
            "Snow/Ice",
            "Barren",
            "Forest",
            "Shrubland",
            "Water",
        ]
        scene_metrics = {scene: {} for scene in scenes_cls}
        for scene in list(scene_metrics.keys()):
            for model_name in list(self.model_names_mapping.values()):
                scene_metrics[scene][model_name] = {
                    "aAcc": [],
                    "mIoU": [],
                    "mAcc": [],
                    "mDice": [],
                    "mFscore": [],
                    "mPrecision": [],
                    "mRecall": [],
                }

        for data in track(
            self.val_dataloader,
            description="evaling...",
            total=len(self.val_dataloader),
        ):
            img = data["img"].to(self.device)
            ann = data["ann"].to(self.device)
            lac_type = data["lac_type"][0]
            for model_name, model in self.models.items():
                pred = self.inference(img, model)
                metric.results.append(
                    metric.intersect_and_union(
                        pred, ann, num_classes=4, ignore_index=255
                    )
                )
                result: dict = metric.compute_metrics(metric.results)
                for metrics_name, metrics_data in result.items():
                    scene_metrics[lac_type][self.model_names_mapping[model_name]][metrics_name].append(metrics_data)
                metric.results = []

        result = {}
        for scene in list(scene_metrics.keys()):
            result[scene] = {}
            for model_name in list(scene_metrics[scene].keys()):
                result[scene][model_name] = {}
                for metrics in list(scene_metrics[scene][model_name].keys()):
                    scene_metrics[scene][model_name][metrics] = [0.00 if math.isnan(x) else x for x in scene_metrics[scene][model_name][metrics]]
                    val = sum( scene_metrics[scene][model_name][metrics]) / len(scene_metrics[scene][model_name][metrics])
                    val = round(val,2)
                    result[scene][model_name][metrics] = val

        json_data = json.dumps(result, ensure_ascii=False, indent=4)
        with open("result.json", "w") as json_file:
            json_file.write(json_data)


if __name__ == "__main__":
    # example usage: python src/eval/eval_l8_scene.py --gpu cuda:0
    root, gpu = get_args()
    Eval(root, gpu).run()
