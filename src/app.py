# -*- coding: utf-8 -*-
# @Time    : 2024/8/4 下午2:38
# @Author  : xiaoshun
# @Email   : 3038523973@qq.com
# @File    : app.py
# @Software: PyCharm

from glob import glob

import albumentations as albu
import gradio as gr
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2

from src.models.components.cdnetv1 import CDnetV1
from src.models.components.cdnetv2 import CDnetV2
from src.models.components.dbnet import DBNet
from src.models.components.hrcloudnet import HRCloudNet
from src.models.components.kappamask import KappaMask
from src.models.components.mcdnet import MCDNet
from src.models.components.scnn import SCNN
from src.models.components.unetmobv2 import UNetMobV2


class Application:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {
            "cdnetv1": CDnetV1(num_classes=2).to(self.device),
            "cdnetv2": CDnetV2(num_classes=2).to(self.device),
            "hrcloudnet": HRCloudNet(num_classes=2).to(self.device),
            "mcdnet": MCDNet(in_channels=3, num_classes=2).to(self.device),
            "scnn": SCNN(num_classes=2).to(self.device),
            "dbnet": DBNet(img_size=256, in_channels=3, num_classes=2).to(
                self.device
            ),
            "unetmobv2": UNetMobV2(num_classes=2).to(self.device),
            "kappamask":KappaMask(num_classes=2,in_channels=3).to(self.device)
        }
        self.__load_weight()
        self.transform = albu.Compose(
            [
                albu.Resize(256, 256, always_apply=True),
                albu.ToFloat(),
                ToTensorV2(),
            ]
        )

    def __load_weight(self):
        """
        将模型权重加载进来
        """
        for model_name, model in self.models.items():
            weight_path = glob(
                f"logs/train/runs/*{model_name}*/*/checkpoints/*epoch*.ckpt"
            )[0]
            weight = torch.load(weight_path, map_location=self.device)
            state_dict = {}
            for key, value in weight["state_dict"].items():
                new_key = key[4:]
                state_dict[new_key] = value
            model.load_state_dict(state_dict)
            model.eval()
            print(f"{model_name} weight loaded!")

    @torch.no_grad
    def inference(self, image: torch.Tensor, model_name: str):
        x = image.float()
        x = x.unsqueeze(0)
        x = x.to(self.device)
        logits = self.models[model_name](x)
        if isinstance(logits, tuple):
            logits = logits[0]
        fake_mask = torch.argmax(logits, 1).detach().cpu().squeeze(0).numpy()
        return fake_mask

    def give_colors_to_mask(self, mask: np.ndarray):
        """
        赋予mask颜色
        """
        assert len(mask.shape) == 2, "Value Error,mask的形状为(height,width)"
        colors_mask = np.zeros((mask.shape[0], mask.shape[1], 3)).astype(np.float32)
        colors = ((255, 255, 255), (128, 192, 128))
        for color in range(2):
            segc = mask == color
            colors_mask[:, :, 0] += segc * (colors[color][0])
            colors_mask[:, :, 1] += segc * (colors[color][1])
            colors_mask[:, :, 2] += segc * (colors[color][2])
        return colors_mask

    def to_pil(self, image: np.ndarray, width=None, height=None):
        colors_np = self.give_colors_to_mask(image)
        pil_np = Image.fromarray(np.uint8(colors_np))
        if width and height:
            pil_np = pil_np.resize((width, height))
        return pil_np

    def flip(self, image_pil: Image.Image, model_name: str):
        if image_pil is None:
            return Image.fromarray(np.uint8(np.random.random((32, 32, 3)) * 255)), "请上传一张图片"
        if model_name is None:
            return Image.fromarray(np.uint8(np.random.random((32, 32, 3)) * 255)), "请选择模型名称"
        image = np.array(image_pil)
        raw_height, raw_width = image.shape[0], image.shape[1]
        print("image type:",image.dtype)
        transform = self.transform(image=image)
        image = transform["image"]
        fake_image = self.inference(image, model_name)
        fake_image = self.to_pil(fake_image, raw_width, raw_height)
        return fake_image, "success"

    def tiff_to_png(image: Image.Image):
        if image.format == "TIFF":
            image = image.convert("RGB")
        return np.array(image)

    def run(self):
        app = gr.Interface(
            self.flip,
            [
                gr.Image(sources=["clipboard", "upload"], type="pil"),
                gr.Radio(
                    ["cdnetv1", "cdnetv2", "hrcloudnet", "mcdnet", "scnn", "dbnet", "unetmobv2","kappamask"],
                    label="model_name",
                    info="选择使用的模型",
                ),
            ],
            [gr.Image(), gr.Textbox(label="提示信息")],
            examples=[
                ["images/app_examples/barren_11.png", "dbnet"],
                ["images/app_examples/snow_10.png", "scnn"],
                ["images/app_examples/vegetation_21.png", "cdnetv2"],
                ["images/app_examples/water_22.png", "hrcloudnet"],
            ],
            title="云检测模型在线演示",
            submit_btn=gr.Button("Submit", variant="primary")
        )
        app.launch(share=True)


if __name__ == "__main__":
    app = Application()
    app.run()
