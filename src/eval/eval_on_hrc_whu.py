from glob import glob
from rich.table import Table
from typing import Tuple, Dict
from rich.progress import track
import csv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torchvision
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, F1Score, Precision, Recall
from torchmetrics.utilities.data import to_onehot
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore
from torchvision import transforms
import torch
from torch import nn as nn
import os
from src.data.hrcwhu_datamodule import HRCWHU
from src.models.components.cdnetv1 import CDnetV1
from src.models.components.cdnetv2 import CDnetV2
from src.models.components.dbnet import DBNet
from src.models.components.hrcloudnet import HRCloudNet
from src.models.components.kappamask import KappaMask
from src.models.components.mcdnet import MCDNet
from src.models.components.scnn import SCNN
from src.models.components.unetmobv2 import UNetMobV2


class EvalOnHRCWHU:
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
        self.root = "data/hrcwhu"
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
        self.data_list = self.__load_data()
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
                f"logs/train/runs/hrcwhu_{model_name}*/*/checkpoints/*epoch*.ckpt"
            )[0]
            weight = torch.load(weight_path, map_location=self.device)
            state_dict = {}
            for key, value in weight["state_dict"].items():
                new_key = key[4:]
                state_dict[new_key] = value
            model.load_state_dict(state_dict)
            model.eval()

    def __load_data(self):
        data_list = []
        split = "test"
        split_file = os.path.join(self.root, f"{split}.txt")
        with open(split_file, "r") as f:
            for line in f:
                image_file = line.strip()
                img_path = os.path.join(self.root, "img_dir", split, image_file)
                ann_path = os.path.join(self.root, "ann_dir", split, image_file)
                data_list.append((img_path, ann_path))
        return data_list

    def __get_metrics(self, task: str, num_classes: int):
        return MetricCollection(
            {
                "Acc": MulticlassAccuracy(num_classes=num_classes, average="macro"),
                "Precision": Precision(
                    task=task, num_classes=num_classes, average="macro"
                ),
                "Recall": Recall(task=task, num_classes=num_classes, average="macro"),
                "F1-Score": F1Score(
                    task=task, num_classes=num_classes, average="macro"
                ),
                "MIoU": MeanIoU(num_classes=num_classes, per_class=False),
                "Dice": GeneralizedDiceScore(num_classes=num_classes, per_class=False),
            }
        ).to(self.device)

    def give_colors_to_mask(
        self, image: torch.Tensor, mask: torch.Tensor, num_classes=2
    ):
        mask = to_onehot(mask, num_classes=num_classes).to(torch.bool)[0]
        mask_colors = (
            torchvision.utils.draw_segmentation_masks(
                image, mask, colors=list(HRCWHU.METAINFO["palette"]), alpha=1.0
            )
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        return mask_colors

    def table_to_csv(self, table: Table, filename="hrc_metrics.csv"):
        # 将Table数据导出到CSV
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # 写入表头
            writer.writerow([column.header for column in table.columns])
            # 写入数据行
            for row in table.rows:
                print(row)
                writer.writerow([cell.plain for cell in row.cells])

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

    def __get_img_ann(
        self, img_path: str, ann_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        center_crop = transforms.CenterCrop((256, 256))
        img_to_tensor = transforms.ToTensor()
        img = np.array(Image.open(img_path))
        ann = np.array(Image.open(ann_path))[:, :, np.newaxis]

        img_ann = np.concatenate((img, ann), axis=-1)
        img_ann = Image.fromarray(img_ann)
        img_ann = center_crop(img_ann)
        img_ann = np.array(img_ann)
        img = img_ann[:,:,:3]
        ann = img_ann[:,:,-1]

        img = img_to_tensor(img)
        img = img.unsqueeze(0)

        ann = torch.tensor(ann).long().unsqueeze(0)

        return img.to(self.device), ann.to(self.device)

    @torch.no_grad()
    def inference(
        self, img: torch.Tensor, model: nn.Module
    ) -> torch.Tensor:
        logits = model(img)
        if isinstance(logits, tuple):
            logits = logits[0]
        pred = torch.argmax(logits, 1).detach()
        return pred

    def make_table_and_csv(self, filename="hrc_metrics.csv"):
        data = {
            model_name: model_metric.compute()
            for model_name, model_metric in self.model_metrics.items()
        }
        metric_weights = {
            "Acc": 0.2,
            "Dice": 0.2,
            "F1-Score": 0.2,
            "MIoU": 0.2,
            "Precision": 0.1,
            "Recall": 0.1,
        }
        averages = {
            model: sum(
                value[metric].item() * metric_weights[metric] for metric in value
            )
            for model, value in data.items()
        }
        sorted_models = sorted(averages, key=averages.get)
        correct_flow = ["Acc", "Precision", "Recall", "F1-Score", "MIoU", "Dice"]
        with open("tmp.csv", "w", newline="") as csvfile:
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
        for img_path, ann_path in track(
            self.data_list, description="evaling...", total=len(self.data_list)
        ):
            img, ann = self.__get_img_ann(img_path, ann_path)
            model_masks = {}
            for model_name, model in self.models.items():
                model_name = self.invert_model_mapping[model_name]
                pred = self.inference(img, model)
                self.model_metrics[model_name].update(pred, ann)
                color_mask = self.give_colors_to_mask(img[0], pred)
                model_masks[model_name] = color_mask
            image = img[0].detach().cpu().permute(1, 2, 0).numpy()
            gt = self.give_colors_to_mask(img[0], ann)
            masks = [model_masks[mask_name] for mask_name in model_order]
            masks = [image] + [gt] + masks
            masks = np.array(masks)
            if show_images is None:
                show_images = masks
            else:
                show_images = np.concatenate((show_images, masks), axis=0)
        # self.make_table_and_csv()
        self.visualize_img(show_images)


if __name__ == "__main__":
    EvalOnHRCWHU().run()
