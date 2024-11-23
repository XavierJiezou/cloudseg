# -*- coding: utf-8 -*-
# @Time    : 2024/8/1 下午2:45
# @Author  : xiaoshun
# @Email   : 3038523973@qq.com
# @File    : cdnetv2_loss.py
# @Software: PyCharm
import torch
import torch.nn as nn


class CDnetv2Loss(nn.Module):
    def __init__(self, loss_fn: nn.Module) -> None:
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, logits: torch.Tensor, logits_aux,target: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(logits, target)
        loss_aux = self.loss_fn(logits_aux, target)
        total_loss = loss + loss_aux
        return total_loss
