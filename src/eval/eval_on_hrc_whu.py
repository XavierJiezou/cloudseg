from glob import glob
from rich.table import Table
from typing import Tuple, Dict
from rich.progress import track
import csv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from mmseg.evaluation.metrics.iou_metric import IoUMetric
from torchmetrics.utilities.data import to_onehot
import albumentations as albu
import torch
from torch import nn as nn
import torchvision
import os
from src.data.hrc_whu_datamodule import HRC_WHU
from src.data.hrc_whu_datamodule import HRC_WHUDataModule
from src.models.components.cdnetv1 import CDnetV1
from src.models.components.cdnetv2 import CDnetV2
from src.models.components.dbnet import DBNet
from src.models.components.hrcloudnet import HRCloudNet
from src.models.components.kappamask import KappaMask
from src.models.components.mcdnet import MCDNet
from src.models.components.scnn import SCNN
from src.models.components.unetmobv2 import UNetMobV2


class EvalOnHRC_WHU:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {
            "cdnetv1": CDnetV1(num_classes=2).to(self.device),
            "cdnetv2": CDnetV2(num_classes=2).to(self.device),
            "hrcloudnet": HRCloudNet(num_classes=2).to(self.device),
            "mcdnet": MCDNet(in_channels=3, num_classes=2).to(self.device),
            "scnn": SCNN(num_classes=2).to(self.device),
            "dbnet": DBNet(img_size=256, in_channels=3, num_classes=2).to(self.device),
            "unetmobv2": UNetMobV2(num_classes=2).to(self.device),
            "kappamask": KappaMask(num_classes=2, in_channels=3).to(self.device),
        }
        self.root = "data/hrc_whu"
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
        self.__load_weight()
        self.val_dataloader = self.__load_data()
        self.model_metrics = {
            model_name: self.__get_metrics("multiclass", 2)
            for model_name in self.model_names_mapping
        }

    def __load_weight(self):
        """
        将模型权重加载进来
        """
        for model_name, model in self.models.items():
            weight_path = glob(
                f"logs/train/runs/hrc_whu_{model_name}*/*/checkpoints/*epoch*.ckpt"
            )[0]
            weight = torch.load(weight_path, map_location=self.device)
            state_dict = {}
            for key, value in weight["state_dict"].items():
                new_key = key[4:]
                state_dict[new_key] = value
            model.load_state_dict(state_dict)
            model.eval()

    def __load_data(self):
        train_pipeline = val_pipeline = test_pipeline = dict(
            all_transform=albu.Compose([albu.CenterCrop(256, 256)]),
            img_transform=albu.Compose([albu.ToFloat(), ToTensorV2()]),
            ann_transform=None,
        )
        data_loader = HRC_WHUDataModule(
            root=self.root,
            train_pipeline=train_pipeline,
            val_pipeline=val_pipeline,
            test_pipeline=test_pipeline,
        )
        data_loader.prepare_data()
        data_loader.setup()
        val_dataloader = data_loader.test_dataloader()
        return val_dataloader

    def __get_metrics(self, task: str, num_classes: int):
        collect_device = "gpu" if self.device == "cuda" else "cpu"
        metric = IoUMetric(
            iou_metrics=["mIoU", "mDice", "mFscore"],
            num_classes=num_classes,
            collect_device=collect_device,
        )
        metric._dataset_meta = dict(classes=HRC_WHU.METAINFO["classes"])

        return metric

    def give_colors_to_mask(
        self, image: torch.Tensor, mask: torch.Tensor, num_classes=2
    ):
        mask = to_onehot(mask, num_classes=num_classes).to(torch.bool)[0]
        mask_colors = (
            torchvision.utils.draw_segmentation_masks(
                image, mask, colors=list(HRC_WHU.METAINFO["palette"]), alpha=1.0
            )
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        return mask_colors

    def visualize_img(self, show_images: np.ndarray):
        show_images_tensor = torch.from_numpy(show_images).permute(0, 3, 1, 2)
        show_image = torchvision.utils.make_grid(
            show_images_tensor, nrow=len(self.models) + 2, padding=8
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
        column_titles = ["Input", "Label"] + [
            "UNetMobv2",
            "DBNet",
            "CDNetv1",
            "HRCloudNet",
            "KappaMask",
            "CDNetv2",
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
        new_image.save("1.png", dpi=(300, 300))
        new_image.save("1.pdf", dpi=(300, 300))

    @torch.no_grad()
    def inference(self, img: torch.Tensor, model: nn.Module) -> torch.Tensor:
        logits = model(img)
        if isinstance(logits, tuple):
            logits = logits[0]
        pred = torch.argmax(logits, 1).detach()
        return pred

    def show_metrics(self, filename="hrc_metrics.csv"):
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

    def run(self, num_classes: int = 2):
        """
        评测模型
        """
        model_order = [
            "UNetMobv2",
            "DBNet",
            "CDNetv1",
            "HRCloudNet",
            "KappaMask",
            "CDNetv2",
            "SCNN",
            "MCDNet",
        ]
        show_images = None
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
                        num_classes=num_classes,
                        ignore_index=self.model_metrics[model_name].ignore_index,
                    )
                )
                color_mask = self.give_colors_to_mask(img[0], pred)
                model_masks[model_name] = color_mask
            image = img[0].detach().cpu().permute(1, 2, 0).numpy()
            gt = self.give_colors_to_mask(img[0], ann)
            masks = [model_masks[mask_name] for mask_name in model_order]
            masks = [image] + [gt] + masks
            masks = np.array(masks)
            if show_images is not None and show_images.shape[0] > 5:
                continue
            if show_images is None:
                show_images = masks
            else:
                show_images = np.concatenate((show_images, masks), axis=0)
        # self.show_metrics()
        self.visualize_img(show_images)


if __name__ == "__main__":
    EvalOnHRC_WHU().run(num_classes=2)
