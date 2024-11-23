from glob import glob
import argparse
from rich.progress import track
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2

from collections import OrderedDict
from torchmetrics.utilities.data import to_onehot
import albumentations as albu
import torch
from torch import nn as nn
from copy import deepcopy
import torchvision
import os
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
        "--root", type=str, help="使用的设备", default="data/l8_biome_crop"
    )
    parser.add_argument("--gpu", type=str, help="使用的设备", default="cuda:0")

    args = parser.parse_args()
    return args.root, args.gpu


class Visualize:
    def __init__(self, root: str, device: str):
        self.device = device
        self.root = root
        self.colors = (
            (0, 0, 0),
            (85, 85, 85),
            (170, 170, 170),
            (255, 255, 255),
        )
        self.num_classes = 4
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
        self.invert_model_mapping = {
            value: key for key, value in self.model_names_mapping.items()
        }
        self.models = OrderedDict(
            {
                key: self.models[key]
                for key in model_order["l8_biome_crop"]
                if key in self.models
            }
        )
        self.__load_weight()
        self.val_dataloader = self.__load_data()

    def __load_weight(self):
        """
        将模型权重加载进来
        """
        for model_name, model in self.models.items():
            weight_path = glob(
                f"logs/l8_biome_crop/{model_name}/*/checkpoints/*epoch*.ckpt"
            )[0]
            weight = torch.load(weight_path, map_location=self.device)
            state_dict = {}
            for key, value in weight["state_dict"].items():
                new_key = key[4:]
                state_dict[new_key] = value
            model.load_state_dict(state_dict)
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

    def __load_data(self):

        data_loader = self.__get_data_module()
        data_loader.prepare_data()

        data_loader.setup("test")
        val_dataloader = data_loader.test_dataloader()
        return val_dataloader

    @torch.no_grad()
    def inference(self, img: torch.Tensor, model: nn.Module) -> torch.Tensor:
        logits = model(img)
        if isinstance(logits, tuple):
            logits = logits[0]
        pred = torch.argmax(logits, 1).detach()
        return pred

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

    def visualize_img(
        self, show_images: np.ndarray, index=None, column_titles=None, filename=None
    ):
        show_images_tensor = torch.from_numpy(show_images).permute(0, 3, 1, 2)
        show_image = torchvision.utils.make_grid(show_images_tensor, nrow=1, padding=8)
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
        assert filename, f"filename is None"
        new_image.save(filename, dpi=(300, 300))

    def visualize_img_no_text(
        self, show_images: np.ndarray, index=None, column_titles=None, filename=None
    ):
        show_images_tensor = torch.from_numpy(show_images).permute(0, 3, 1, 2)
        show_image = torchvision.utils.make_grid(show_images_tensor, nrow=1, padding=8)
        grid_image = torchvision.transforms.ToPILImage()(show_image)

        assert filename, f"filename is None"
        grid_image.save(filename, dpi=(300, 300))

    def vis(self):
        index = 0
        for data in track(
            self.val_dataloader,
            description="vis...",
            total=len(self.val_dataloader),
        ):
            img = data["img"].to(self.device)
            ann = data["ann"].to(self.device)
            lac_type = data["lac_type"][0]
            os.makedirs(
                os.path.join("images", "l8_biome_crop", lac_type), exist_ok=True
            )
            model_masks = {}
            for model_name, model in self.models.items():
                model_name = self.invert_model_mapping[model_name]
                pred = self.inference(img, model)

                color_mask = self.give_colors_to_mask(
                    img[0], pred, num_classes=self.num_classes
                )
                model_masks[model_name] = color_mask
            image = (img[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(
                np.uint8
            )
            gt = self.give_colors_to_mask(img[0], ann, num_classes=self.num_classes)
            image = gaussian_stretch(image)
            masks = [image] + [gt] + list(model_masks.values())
            masks = np.concatenate(masks, axis=1)
            masks = masks[None,]
            self.visualize_img_no_text(
                masks,
                index=index,
                filename=os.path.join("images", "l8_biome_crop", lac_type)
                + os.path.sep
                + f"{index}.png",
            )
            index += 1


if __name__ == "__main__":
    # 使用示例 python src/eval/vis_l8.py --gpu "cuda:3"
    root, device = get_args()
    Visualize(root, device).vis()
