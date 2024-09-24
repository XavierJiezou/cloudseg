import torch
from torch import nn as nn
from src.models.components.dbnet import DBNet
from src.data.l8_biome_datamodule import L8BiomeDataModule
from src.metrics.metric import IoUMetric
import albumentations as albu
from rich.progress import track
import numpy as np
from albumentations.pytorch import ToTensorV2


@torch.no_grad()
def inference(model: nn.Module, img: torch.Tensor) -> np.ndarray:
    logits = model(img)
    preds = torch.argmax(logits, dim=1)
    return preds


if __name__ == "__main__":
    device = "cuda:7"
    model = DBNet(img_size=512, in_channels=3, num_classes=5).to(device)
    ckpt = torch.load(
        "epoch_012.ckpt",
        map_location=device,
    )
    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        new_k = k[4:]
        state_dict[new_k] = v
    model.load_state_dict(state_dict)
    model.eval()
    img_transform = albu.Compose([ToTensorV2(), albu.ToFloat(255)])
    all_transform = None
    ann_transform = None
    train_pipeline = val_pipeline = test_pipeline = dict(
        img_transform=img_transform,
        all_transform=all_transform,
        ann_transform=ann_transform,
    )
    data_module = L8BiomeDataModule(
        train_pipeline=train_pipeline,
        val_pipeline=val_pipeline,
        test_pipeline=test_pipeline,
        batch_size=4,
        pin_memory=True,
    )
    data_module.setup("fit")
    data_module.setup("validate")
    data_module.setup("test")
    data_loader = data_module.test_dataloader()
    metrics = IoUMetric(5, iou_metrics=["mIoU", "mDice", "mFscore"])

    for data in track(data_loader,total=len(data_loader)):
        img = data["img"].to(device)
        ann = data["ann"].to(device)
        preds = inference(model, img)
        print(preds.shape, ann.shape)
        print(preds.dtype, ann.dtype)
        metrics.results.append(
            IoUMetric.intersect_and_union(preds, ann, num_classes=5, ignore_index=255)
        )
    result = metrics.compute_metrics(metrics.results)
    print(result)
