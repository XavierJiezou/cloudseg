from glob import glob
from rich.table import Table
from typing import Tuple, Dict
from rich.progress import track
import csv
from rich.console import Console
import matplotlib.pyplot as plt
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, F1Score, Precision, Recall
from torchmetrics.utilities.data import to_onehot
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore
import torch
from torch.nn import functional as F
from torch import nn as nn
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from src.data.hrcwhu_datamodule import HRCWHUDataModule
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
        self.__load_weight()
        self.eval_dataset = self.__load_dataset(root="data/hrcwhu", phase="test")
        self.model_metrics = {
            model_name: self.__get_metrics("multiclass", 2, HRCWHU.METAINFO["classes"])
            for model_name in self.models
        }
        self.metrics_loss = {
            model_name: MeanMetric().to(self.device) for model_name in self.models
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

    def __load_dataset(self, root: str, phase: str):
        val_pipeline = dict(
            all_transform=albu.Compose([albu.Resize(256, 256, always_apply=True)]),
            img_transform=albu.Compose([albu.ToFloat(), ToTensorV2()]),
            ann_transform=None,
        )
        data_module = HRCWHUDataModule(
            root=root,
            train_pipeline=val_pipeline,
            val_pipeline=val_pipeline,
            test_pipeline=val_pipeline,
        )
        data_module.prepare_data()
        data_module.setup()
        dataset = data_module.test_dataloader()
        return dataset

    def __get_metrics(self, task: str, num_classes: int, classes: tuple):
        return MetricCollection(
            {
                "accuracy": ClasswiseWrapper(
                    MulticlassAccuracy(num_classes=num_classes, average="none"),
                    labels=list(classes),
                ),
                "precision": ClasswiseWrapper(
                    Precision(task=task, num_classes=num_classes, average="none"),
                    labels=list(classes),
                ),
                "recall": ClasswiseWrapper(
                    Recall(task=task, num_classes=num_classes, average="none"),
                    labels=list(classes),
                ),
                "f1Score": ClasswiseWrapper(
                    F1Score(task=task, num_classes=num_classes, average="none"),
                    labels=list(classes),
                ),
                "iou": ClasswiseWrapper(
                    MeanIoU(num_classes=num_classes, per_class=True),
                    labels=list(classes),
                ),
                "dice": ClasswiseWrapper(
                    GeneralizedDiceScore(num_classes=num_classes, per_class=True),
                    labels=list(classes),
                ),
            }
        ).to(self.device)

    def make_table_and_csv(self, model_name: str, filename="hrc_metrics.csv"):
        res = {
            model_name: model_metric.compute()
            for model_name, model_metric in self.model_metrics.items()
        }
        # res.update(
        #     {
        #         f"{model_name}_loss": self.metrics_loss[model_name].compute()
        #         for model_name in self.model_metrics
        #     }
        # )
        columns = list(res[model_name].keys())
        width = 50
        table = Table(title="方法与指标对比")
        table.add_column(
            "方法\\指标", justify="center", style="bold", no_wrap=True, width=width
        )
        for column in columns:
            table.add_column(
                column, justify="center", style="bold", no_wrap=True, width=width
            )
        table.add_column(
            "loss", justify="center", style="bold", no_wrap=True, width=width
        )
        csv_data = []
        csv_data.append(["方法\\指标"] + columns + ["loss"])
        model_res = {}
        for method, metrics in res.items():
            row_data = (
                [method]
                + [f"{round(metrics[column].item(),4)}" for column in columns]
                + [f"{round(self.metrics_loss[method].compute().item(),4)}"]
            )
            table.add_row(*row_data)
            csv_data.append(row_data)
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)

        return table

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

    def visualize_img(self):
        pass

    @torch.no_grad()
    def inference(
        self, img: torch.Tensor, target: torch.Tensor, model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = model(img)
        if isinstance(logits, tuple):
            logits = logits[0]
        pred = torch.argmax(logits, 1).detach()
        loss = F.cross_entropy(logits, target).detach()
        return pred, loss

    def run(self):
        """
        评测模型
        """
        for data in track(
            self.eval_dataset, description="evaling...", total=len(self.eval_dataset)
        ):
            img = data["img"].to(self.device)
            ann = data["ann"].to(self.device)
            img_path = data["img_path"][0]
            for model_name, model in self.models.items():
                pred, loss = self.inference(img, ann, model)
                self.model_metrics[model_name].update(
                    to_onehot(pred, num_classes=2), to_onehot(ann, num_classes=2)
                )
                self.metrics_loss[model_name].update(loss)
        self.make_table_and_csv(model_name)


if __name__ == "__main__":
    EvalOnHRCWHU().run()
