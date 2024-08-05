from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore


class BaseLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        num_classes: int,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net'])

        self.net = net

        # metric objects for calculating and averaging accuracy across batches
        task = "binary" if self.hparams.num_classes==2 else "multiclass"
        
        self.train_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.train_precision = Precision(task=task, num_classes=num_classes)
        self.train_recall = Recall(task=task, num_classes=num_classes)
        self.train_f1score = F1Score(task=task, num_classes=num_classes)
        self.train_miou = MeanIoU(num_classes=num_classes)
        self.train_dice = GeneralizedDiceScore(num_classes=num_classes)
        
        self.val_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.val_precision = Precision(task=task, num_classes=num_classes)
        self.val_recall = Recall(task=task, num_classes=num_classes)
        self.val_f1score = F1Score(task=task, num_classes=num_classes)
        self.val_miou = MeanIoU(num_classes=num_classes)
        self.val_dice = GeneralizedDiceScore(num_classes=num_classes)
        
        self.test_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.test_precision = Precision(task=task, num_classes=num_classes)
        self.test_recall = Recall(task=task, num_classes=num_classes)
        self.test_f1score = F1Score(task=task, num_classes=num_classes)
        self.test_miou = MeanIoU(num_classes=num_classes)
        self.test_dice = GeneralizedDiceScore(num_classes=num_classes)
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_miou_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1score.reset()
        self.val_miou.reset()
        self.val_dice.reset()
        
        self.val_miou_best.reset()

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
        x, y = batch["img"], batch["ann"]
        logits = self.forward(x)
        loss = self.hparams.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)
        
        # print(preds.shape) # (8, 256, 256)
        # print(targets.shape) # (8, 256, 256)

        # update and log metrics
        self.train_loss(loss)
        
        self.train_accuracy(preds, targets)
        self.train_precision(preds, targets)
        self.train_recall(preds, targets)
        self.train_f1score(preds, targets)
        self.train_miou(preds, targets)
        self.train_dice(preds, targets)
        
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log("train/accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision", self.train_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1score", self.train_f1score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/miou", self.train_miou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dice", self.train_dice, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        
        self.val_accuracy(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_f1score(preds, targets)
        self.val_miou(preds, targets)
        self.val_dice(preds, targets)
        
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log("val/accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1score", self.val_f1score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/miou", self.val_miou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        miou = self.val_miou.compute()  # get current val acc
        self.val_miou_best(miou)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/miou_best", self.val_miou_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)
        
        # update and log metrics
        self.test_loss(loss)

        # update and log metrics
        self.test_accuracy(preds, targets)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.test_f1score(preds, targets)
        self.test_miou(preds, targets)
        self.test_dice(preds, targets)
        
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log("test/accuracy", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1score", self.test_f1score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/miou", self.test_miou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/dice", self.test_dice, on_step=False, on_epoch=True, prog_bar=True)
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = BaseLitModule(None, None, None, None)
