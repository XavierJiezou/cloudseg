from glob import glob
import argparse
from rich.table import Table
from typing import Tuple, Dict
from rich.progress import track
import csv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2

from collections import OrderedDict
from src.metrics.metric import IoUMetric
from torchmetrics.utilities.data import to_onehot
import albumentations as albu
import torch
from torch import nn as nn
from copy import deepcopy
import torchvision
import os
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
from src.utils.stretch import gaussian_stretch


def get_args():
    parser = argparse.ArgumentParser(description="获取实验名称和使用的显卡信息")
    parser.add_argument(
        "--experiment_name", type=str, help="实验的名称", default="hrc_whu"
    )
    parser.add_argument("--gpu", type=str, help="使用的设备", default="cuda:0")

    args = parser.parse_args()
    return args.experiment_name, args.gpu


class Eval:
    def __init__(self, experiment_name: str, device: str):
        self.device = device
        self.num_classes, self.image_size, self.colors = (
            self.__get_num_classes_image_shape_colors(experiment_name)
        )
        self.experiment_name = experiment_name
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
        self.models = OrderedDict(
             {key: self.models[key] for key in model_order[experiment_name] if key in self.models}
        )
        assert len(self.models) == 8,f"num of models is {len(self.models)},but actually is 8"
        self.root = self.__get_root(experiment_name)
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
        self.invert_model_mapping = {
            value: key for key, value in self.model_names_mapping.items()
        }
        self.__load_weight(experiment_name)
        self.val_dataloader = self.__load_data(experiment_name)
        self.model_metrics = {
            model_name: self.__get_metrics(self.num_classes, model_name=model_name)
            for model_name in self.model_names_mapping
        }

    def __get_root(self, experiment_name: str):
        experiment_root_mapping = {
            "cloudsen12_high_l1c": "data/cloudsen12_high",
            "cloudsen12_high_l2a": "data/cloudsen12_high",
            "gf12ms_whu_gf1": "data/gf12ms_whu",
            "gf12ms_whu_gf2": "data/gf12ms_whu",
            "hrc_whu": "data/hrc_whu",
            "l8_biome_crop": "data/l8_biome_crop",
        }
        return experiment_root_mapping[experiment_name]

    def __get_num_classes_image_shape_colors(self, experiment_name: str):
        if experiment_name in ["cloudsen12_high_l1c", "cloudsen12_high_l2a"]:
            return (
                4,
                512,
                (
                    (0, 0, 0),
                    (255, 255, 255),
                    (170, 170, 170),
                    (85, 85, 85),
                ),
            )
        elif experiment_name in ["gf12ms_whu_gf1", "gf12ms_whu_gf2"]:
            return 2, 256, ((0, 0, 0), (255, 255, 255))
        elif experiment_name in ["hrc_whu"]:
            return 2, 256, ((79, 253, 199), (77, 2, 115))
        elif experiment_name in ["l8_biome_crop"]:
            return (
                4,
                512,
                (
                    (0, 0, 0),
                    (85, 85, 85),
                    (170, 170, 170),
                    (255, 255, 255),
                ),
            )
        raise ValueError(f"Experiment name {experiment_name} is not recognized.")

    def __load_weight(self, experiment_name):
        """
        将模型权重加载进来
        """
        for model_name, model in self.models.items():
            try:
                weight_path = glob(
                    f"logs/{experiment_name}/{model_name}/*/checkpoints/*_*.ckpt"
                )[0]
            except IndexError:
                print(f"logs/{experiment_name}/{model_name}/*/checkpoints/*_*.ckpt")
                exit()
            weight = torch.load(weight_path, map_location=self.device)
            state_dict = {}
            for key, value in weight["state_dict"].items():
                new_key = key[4:]
                state_dict[new_key] = value
            model.load_state_dict(state_dict)
            model.eval()

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

    def __load_data(self, experiment_name: str):

        data_loader = self.__get_data_module(experiment_name)
        data_loader.prepare_data()
        if experiment_name == "l8_biome_crop":

            data_loader.setup("test")
        else:
            data_loader.setup()
        val_dataloader = data_loader.test_dataloader()
        return val_dataloader

    def __get_metrics(self, num_classes: int, model_name=None):
        metric = IoUMetric(
            iou_metrics=["mIoU", "mDice", "mFscore"],
            num_classes=num_classes,
            model_name=model_name,
            dataset_name=self.experiment_name
        )

        return metric

    def give_colors_to_mask(
        self, image: torch.Tensor, mask: torch.Tensor, num_classes=2
    ):
        image = deepcopy(image) * 255
        image = image.to(torch.uint8)
        mask = to_onehot(mask, num_classes=num_classes).to(torch.bool)[0]
        mask_colors = (
            torchvision.utils.draw_segmentation_masks(
                image, mask, colors=list(self.colors), alpha=1.0
            )
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        return mask_colors

    def visualize_img(self, show_images: np.ndarray, index=None, column_titles=None):
        show_images_tensor = torch.from_numpy(show_images).permute(0, 3, 1, 2)
        show_image = torchvision.utils.make_grid(
            show_images_tensor, nrow=1, padding=8
        )
        grid_image = torchvision.transforms.ToPILImage()(show_image)

        # 获取图像尺寸
        width, height = grid_image.size

        # 创建一个新的图像，为标题留出空间
        new_height = height + 50
        new_image = Image.new("RGB", (width, new_height), color="white")
        new_image.paste(grid_image, (0, 50))

        # 准备绘图
        draw = ImageDraw.Draw(new_image)
        font = ImageFont.truetype("resource/Times New Roman.ttf", 50)
        if column_titles is None:
            column_titles = ["Input", "Label"] + [
                "UNetMobv2",
                "DBNet",
                "CDNetv1",
                "HRCloudNet",
                "CDNetv2",
                "KappaMask",
                "SCNN",
                "MCDNet",
            ]
        num_cols = len(column_titles)
        col_width = width // num_cols

        for i, title in enumerate(column_titles):
            # 获取文本大小
            left, top, right, bottom = draw.textbbox((0, 0), title, font=font)
            text_width = right - left
            text_height = bottom - top

            # 计算文本位置（居中）
            x = (i * col_width) + (col_width - text_width) // 2
            y = (50 - text_height) // 2

            # 绘制文本
            draw.text((x, y - 10), title, fill="black", font=font)
        filename = os.path.join("images", f"{self.experiment_name}")
        os.makedirs(filename, exist_ok=True)
        new_image.save(f"{filename}{os.path.sep}{index}.pdf",dpi=(300,300))

    def visualize_img_no_text(self, show_images: np.ndarray, index=None):
        show_images_tensor = torch.from_numpy(show_images).permute(0, 3, 1, 2)
        show_image = torchvision.utils.make_grid(
            show_images_tensor, nrow=1, padding=8
        )
        grid_image = torchvision.transforms.ToPILImage()(show_image)

        # 获取图像尺寸
        filename = os.path.join("images", f"{self.experiment_name}")
        os.makedirs(filename, exist_ok=True)
        grid_image.save(f"{filename}{os.path.sep}{index}.png",dpi=(300,300))


    @torch.no_grad()
    def inference(self, img: torch.Tensor, model: nn.Module) -> torch.Tensor:
        logits = model(img)
        if isinstance(logits, tuple):
            logits = logits[0]
        pred = torch.argmax(logits, 1).detach()
        return pred

    def show_metrics(self, filename="hrc_metrics.csv"):
        filename = f"eval_results/{self.experiment_name}_metrics.csv"
        data = {
            model_name: model_metric.compute_metrics(model_metric.results)
            for model_name, model_metric in self.model_metrics.items()
        }
        metric_weights = {
            "aAcc": 0,
            "mIoU": 0.25,
            "mAcc": 0.25,
            "mDice": 0.25,
            "mFscore": 0.25,
            "mPrecision": 0,
            "mRecall": 0,
        }
        averages = {
            model: sum(
                value[metric].item() * metric_weights[metric] for metric in value
            )
            for model, value in data.items()
        }
        sorted_models = sorted(averages, key=averages.get)
        correct_flow = list(metric_weights.keys())
        with open(filename, "w", newline="") as csvfile:
            fieldnames = ["Method"] + list(next(iter(data.values())).keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for model in sorted_models:
                row = {"Method": model}
                row_data = {
                    metric: round(data[model][metric].item(), 4)
                    for metric in correct_flow
                }
                row.update(row_data)
                writer.writerow(row)

    def run(self):
        """
        评测模型
        """
        show_images = None
        index = 0
        for data in track(
            self.val_dataloader,
            description="evaling...",
            total=len(self.val_dataloader),
        ):
            img: torch.Tensor = data["img"]
            ann: torch.Tensor = data["ann"]

            img, ann = img.to(self.device), ann.to(self.device)
            model_masks = {}
            for model_name, model in self.models.items():
                model_name = self.invert_model_mapping[model_name]
                pred = self.inference(img, model)

                self.model_metrics[model_name].results.append(
                    self.model_metrics[model_name].intersect_and_union(
                        pred,
                        ann,
                        num_classes=self.num_classes,
                        ignore_index=self.model_metrics[model_name].ignore_index,
                    )
                )
                color_mask = self.give_colors_to_mask(
                    img[0], pred, num_classes=self.num_classes
                )
                model_masks[model_name] = color_mask
            image = (img[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gt = self.give_colors_to_mask(img[0], ann, num_classes=self.num_classes)
            image = gaussian_stretch(image)
            masks = [image] + [gt] + list(model_masks.values())
            masks = np.concatenate(masks,axis=1)
            masks = masks[None,]
            # self.visualize_img(masks,column_titles=["Input", "Label"] + list(model_masks.keys()),index=index)
            self.visualize_img_no_text(masks,index=index)
            index += 1

        self.show_metrics()


if __name__ == "__main__":
    # 使用示例 python src/eval/eval_on_experiment.py --experiment_name "hrc_whu" --gpu "cuda:3"
    experiment_name, device = get_args()
    eval_dataset = Eval(experiment_name, device)
    eval_dataset.run()
