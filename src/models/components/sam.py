# -*- coding: utf-8 -*-
# @Time    : 2024/8/8 下午3:41
# @Author  : xiaoshun
# @Email   : 3038523973@qq.com
# @File    : sam.py
# @Software: PyCharm

from typing import Literal
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F


class SAM(nn.Module):
    def __init__(self, model_type="vit_h", points_per_size=32):
        super().__init__()
        self.model = sam_model_registry[model_type]()
        self.points_per_size = 32
        self.transformer = ResizeLongestSide(self.model.image_encoder.img_size)

    def forward(self, images: torch.Tensor):
        B, _, H, W = images.shape

        images = F.interpolate(
            images,
            (self.model.image_encoder.img_size, self.model.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )

        x = np.linspace(0, H - 1, self.points_per_size)
        y = np.linspace(0, W - 1, self.points_per_size)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X, Y], axis=-1).reshape(-1, 2)
        point_labels = np.ones(shape=(images.shape[0], points.shape[0]))

        points = self.transformer.apply_coords(points, (H, W))

        batch_points = np.repeat(points[np.newaxis, :, :], B, axis=0)  # (batch_size, num_points, 2)

        points_pt = torch.as_tensor(batch_points, device=images.device, dtype=torch.float)
        point_labels_pt = torch.as_tensor(
            point_labels, device=images.device, dtype=torch.int
        )

        # exit()

        image_embeddings = self.model.image_encoder(images)
        pred_masks = []
        ious = []
        for embedding in image_embeddings:
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=(points_pt, point_labels_pt),
                boxes=None,
                masks=None,
            )

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks)
            ious.append(iou_predictions)
        return pred_masks, ious


if __name__ == "__main__":
    device = "cuda:7"
    model = SAM().to(device)
    fake_data = torch.randn(2, 3, 512, 512).to(device)
    pred_masks, ious = model(fake_data)
    print(len(pred_masks), len(ious))
    print(pred_masks[0].shape)
