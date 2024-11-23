import argparse
from glob import glob
import os
from rich.progress import track
import torch
from torch import nn as nn
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from PIL import Image
from src.models.components.cdnetv1 import CDnetV1
from src.models.components.cdnetv2 import CDnetV2
from src.models.components.dbnet import DBNet
from src.models.components.hrcloudnet import HRCloudNet
from src.models.components.kappamask import KappaMask
from src.models.components.mcdnet import MCDNet
from src.models.components.scnn import SCNN
from src.models.components.unetmobv2 import UNetMobV2


def get_args():
    parser = argparse.ArgumentParser(
        description="Parse command line arguments for dataset and device."
    )
    parser.add_argument("--dataset_name", help="The name of the dataset to use.")
    parser.add_argument("--root", default="mmseg-data")
    parser.add_argument("--device", help="The device to use for computation.")

    args = parser.parse_args()
    return args.dataset_name, args.root, args.device


class VisSingleDataset:
    def __init__(self, dataset_name, device,root):
        self.dataset_name = dataset_name
        self.device = device
        self.root = root

        self.num_classes = self.get_classes()
        self.image_size = self.get_img_size()
        self.colors = self.get_colors()
        self.file_list = self.get_file_list()

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
        self.load_weight()

    def load_weight(self):
        """
        将模型权重加载进来
        """
        for model_name, model in self.models.items():
            weight_path = f"logs/{self.dataset_name}/{model_name}/*/checkpoints/*_*.ckpt"
            if self.dataset_name == "l8_biome":
                weight_path = f"logs/l8_biome_crop/{model_name}/*/checkpoints/*_*.ckpt"
            weight_path = glob(weight_path)[0]
            weight = torch.load(weight_path, map_location=self.device)
            state_dict = {}
            for key, value in weight["state_dict"].items():
                new_key = key[4:]
                state_dict[new_key] = value
            model.load_state_dict(state_dict)
            model.eval()
            print(f"{model_name} load {weight_path} successfully")

    def get_classes(self):
        if self.dataset_name in [
            "cloudsen12_high_l1c",
            "cloudsen12_high_l2a",
            "l8_biome",
        ]:
            return 4
        return 2

    def get_colors(self):
        if self.dataset_name in ["cloudsen12_high_l1c", "cloudsen12_high_l2a"]:
            return [79, 253, 199, 77, 2, 115, 251, 255, 41, 221, 53, 223]
        if self.dataset_name == "l8_biome":
            return [79, 253, 199, 221, 53, 223, 251, 255, 41, 77, 2, 115]
        if self.dataset_name in ["gf12ms_whu_gf1", "gf12ms_whu_gf2", "hrc_whu"]:
            return [79, 253, 199, 77, 2, 115]
        raise Exception("dataset_name not supported")

    def draw(self, mask: np.ndarray, save_path):
        im = Image.fromarray(mask)
        im.putpalette(self.colors)
        im.save(save_path)

    def get_img_size(self):
        if self.dataset_name in [
            "cloudsen12_high_l1c",
            "cloudsen12_high_l2a",
            "l8_biome",
        ]:
            return 512
        return 256

    def get_image_sub_path(self) -> str:
        if dataset_name in [
            "cloudsen12_high_l1c",
            "cloudsen12_high_l2a",
            "l8_biome",
            "hrc_whu",
        ]:
            return "test"
        return "val"

    def get_file_list(self):
        sub_path = self.get_image_sub_path()
        file_path = os.path.join(
            self.root, self.dataset_name, "img_dir", sub_path, "*"
        )
        file_list = glob(file_path)
        print(f"共计有{len(file_list)}张图片.")
        return file_list

    @torch.no_grad()
    def inference(self, img: torch.Tensor, model: nn.Module) -> torch.Tensor:
        logits = model(img)
        if isinstance(logits, tuple):
            logits = logits[0]
        pred = torch.argmax(logits, 1).detach()
        return pred

    def get_transformer(self):
        transform = albu.Compose(
                [
                    albu.Resize(
                        self.image_size, self.image_size, always_apply=True, p=1
                    ),
                    albu.ToFloat(256),
                    ToTensorV2(),
                ]
            )
        return transform

    def run(self):
        transform = self.get_transformer()

        for model_name in list(self.models.keys()):
            os.makedirs(
                os.path.join("visualization", self.dataset_name, model_name),
                exist_ok=True,
            )

        for file_path in track(self.file_list, total=len(self.file_list)):
            img = np.array(Image.open(file_path))

            img: torch.Tensor = transform(image=img)["image"].to(self.device)
            img = img.unsqueeze(0)
            filename = os.path.basename(file_path).split(".")[0] + ".png"
            for model_name, model in self.models.items():
                pred = self.inference(img, model)
                pred_mask = pred.cpu().squeeze().numpy().astype(np.uint8)
                self.draw(
                    pred_mask,
                    os.path.join(
                        "visualization", self.dataset_name, model_name, filename
                    ),
                )


if __name__ == "__main__":
    # example usage: python src/tools/vis_single_model_dataset.py --dataset_name hrc_whu --device cuda:1 --root data
    dataset_name, root, device = get_args()
    vis = VisSingleDataset(dataset_name=dataset_name, device=device,root=root)
    vis.run()
