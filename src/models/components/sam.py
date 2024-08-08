# -*- coding: utf-8 -*-
# @Time    : 2024/8/8 下午3:41
# @Author  : xiaoshun
# @Email   : 3038523973@qq.com
# @File    : sam.py
# @Software: PyCharm

from typing import Literal
from ultralytics.models import sam
import torch
from torch import nn as nn
from torch.nn import functional as F

class SAM(nn.Module):
    def __init__(self,model:Literal['sam_b.pt','sam_l.pt','sam2_t.pt','sam2_s.pt','sam2_l.pt','sam2_b.pt']="sam_b.pt"):
        super().__init__()
        self.model = sam.SAM(model).model
        self.setup()
        
    def setup(self):
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False
    
    def forward(self,images):
        _, _, H, W = images.shape
        image_embeddings = self.model.image_encoder(images)
        pred_masks = []
        ious = []
        for embedding in image_embeddings:
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
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
            pred_masks.append(masks.squeeze())
            ious.append(iou_predictions)
        return torch.stack(pred_masks,dim=0)
        # return pred_masks, ious
    
if __name__ == "__main__":
    device = "cuda:5"
    model = SAM().to(device)
    fake_data = torch.randn(2,3,1024,1024).to(device)
    pred_masks = model(fake_data)
    print(pred_masks.shape)
    print(pred_masks[0])