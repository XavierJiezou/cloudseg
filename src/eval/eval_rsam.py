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
from torch.nn import functional as F
from src.data.hrc_whu_datamodule import HRC_WHU
from src.data.hrc_whu_datamodule import HRC_WHUDataModule
from src.data.cloudsen12_high_datamodule import CloudSEN12HighDataModule
from src.data.gf12ms_whu_datamodule import GF12MSWHUDataModule
from src.data.l8_biome_crop_datamodule import L8BiomeCropDataModule
from src.models.components.rsam_seg.sam import SAM


def get_args():
    parser = argparse.ArgumentParser(description="获取实验名称和使用的显卡信息")
    parser.add_argument(
        "--dataset_name", type=str, help="数据集名称", default="hrc_whu"
    )
    parser.add_argument("--gpu", type=str, help="使用的设备", default="cuda:0")

    args = parser.parse_args()
    return args.dataset_name, args.gpu


class Eval:
    def __init__(self, dataset_name: str, device: str):
        self.num_classes = 2
        self.device = device
        self.dataset_name = dataset_name
        self.image_size = 256
        self.root = self.__get_root(self.dataset_name)
        encoder_mode = {
            "name": "sam",
            "img_size": 256,
            "mlp_ratio": 4,
            "patch_size": 16,
            "qkv_bias": True,
            "use_rel_pos": True,
            "window_size": 14,
            "out_chans": 256,
            "scale_factor": 32,
            "input_type": "fft",
            "freq_nums": 0.25,
            "prompt_type": "highpass",
            "prompt_embed_dim": 256,
            "tuning_stage": "1234",
            "handcrafted_tune": True,
            "embedding_tune": True,
            "adaptor": "adaptor",
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "global_attn_indexes": [5, 11, 17, 23],
        }

        self.model = SAM(
            inp_size=self.image_size, encoder_mode=encoder_mode, loss="iou"
        ).to(self.device)

        self.__load_weight()
        self.val_dataloader = self.__load_data(dataset_name=self.dataset_name)

    def __get_root(self, dataset_name: str):
        dataset_root_mapping = {
            "cloudsen12_high_l1c": "data/cloudsen12_high",
            "cloudsen12_high_l2a": "data/cloudsen12_high",
            "gf12ms_whu_gf1": "data/gf12ms_whu",
            "gf12ms_whu_gf2": "data/gf12ms_whu",
            "hrc_whu": "data/hrc_whu",
            "l8_biome_crop": "data/l8_biome_crop",
        }
        return dataset_root_mapping[dataset_name]

    def __load_weight(self):
        """
        将模型权重加载进来
        """
        weight_path = glob(
            f"logs/{self.dataset_name}/rsam_seg/*/checkpoints/*epoch*.ckpt"
        )[0]
        weight = torch.load(weight_path, map_location=self.device)
        state_dict = {}
        for key, value in weight["state_dict"].items():
            new_key = key[4:]
            state_dict[new_key] = value
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def __load_data(self, dataset_name: str):

        data_loader = self.__get_data_module(dataset_name)
        data_loader.prepare_data()
        if dataset_name == "l8_biome_crop":

            data_loader.setup("test")
        else:
            data_loader.setup()
        val_dataloader = data_loader.test_dataloader()
        return val_dataloader

    def __get_data_module(self, experiment_name):
        train_pipeline = val_pipeline = test_pipeline = dict(
            all_transform=albu.Compose(
                [
                    albu.PadIfNeeded(
                        self.image_size, self.image_size, p=1, always_apply=True
                    ),
                    albu.CenterCrop(self.image_size, self.image_size),
                ]
            ),
            img_transform=albu.Compose([ToTensorV2()]),
            ann_transform=None,
        )
        if experiment_name == "cloudsen12_high_l1c":
            return CloudSEN12HighDataModule(
                root=self.root,
                level="l1c",
                train_pipeline=train_pipeline,
                val_pipeline=val_pipeline,
                test_pipeline=test_pipeline,
                batch_size=1,
            )

        elif experiment_name == "cloudsen12_high_l2a":
            return CloudSEN12HighDataModule(
                root=self.root,
                level="l2a",
                train_pipeline=train_pipeline,
                val_pipeline=val_pipeline,
                test_pipeline=test_pipeline,
                batch_size=1,
            )
        elif experiment_name == "gf12ms_whu_gf1":
            return GF12MSWHUDataModule(
                root=self.root,
                train_pipeline=train_pipeline,
                val_pipeline=val_pipeline,
                test_pipeline=test_pipeline,
                batch_size=1,
                serial="gf1",
            )
        elif experiment_name == "gf12ms_whu_gf2":
            return GF12MSWHUDataModule(
                root=self.root,
                train_pipeline=train_pipeline,
                val_pipeline=val_pipeline,
                test_pipeline=test_pipeline,
                batch_size=1,
                serial="gf2",
            )
        elif experiment_name == "hrc_whu":
            train_pipeline = val_pipeline = test_pipeline = dict(
                all_transform=albu.Compose(
                    [albu.CenterCrop(self.image_size, self.image_size)]
                ),
                img_transform=albu.Compose([albu.ToFloat(255), ToTensorV2()]),
                ann_transform=None,
            )
            return HRC_WHUDataModule(
                root=self.root,
                train_pipeline=train_pipeline,
                val_pipeline=val_pipeline,
                test_pipeline=test_pipeline,
                batch_size=1,
            )
        elif experiment_name == "l8_biome_crop":
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
        raise ValueError(f"Experiment name {experiment_name} is not recognized.")

    @torch.no_grad()
    def inference(self, image: torch.Tensor) -> torch.tensor:
        logits = self.model.infer(image)
        logits = F.sigmoid(logits)
        preds = (logits >= 0.5).long().detach().squeeze(1)
        return preds

    def run(self):

        metric = IoUMetric(
            num_classes=2,
            iou_metrics=["mIoU", "mDice", "mFscore"],
            model_name=f"{self.dataset_name}_rsam_seg",
        )

        for data in track(
            self.val_dataloader,
            description="evaling...",
            total=len(self.val_dataloader),
        ):
            img: torch.Tensor = data["img"].to(self.device)
            ann: torch.Tensor = data["ann"].to(self.device)
            pred = self.inference(img)
            metric.results.append(
                metric.intersect_and_union(pred, ann, 2, ignore_index=255)
            )

        result = metric.compute_metrics(metric.results)
        with open(f"{self.dataset_name}_rsam.json", "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # example usage: python src/eval/eval_rsam.py --dataset_name hrc_whu --gpu cuda:0
    dataset_name, gpu = get_args()
    Eval(dataset_name, gpu).run()
