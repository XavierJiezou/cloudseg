from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore
from src.models.base_module import BaseLitModule


class L8BiomeLitModule(BaseLitModule):

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch["image"], batch["mask"]
        logits = self.forward(x)
        loss = self.hparams.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    

