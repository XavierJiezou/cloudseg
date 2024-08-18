from typing import Dict, Optional, Tuple

import albumentations as albu
import torch
import torch.utils
from albumentations.pytorch.transforms import ToTensorV2

from src.data.base_datamodule import BaseDataModule
from src.data.components.cloud38 import Cloud38


class Cloud38DataModule(BaseDataModule):
    def __init__(
        self,
        root: str = "/data/zouxuechao/cloudseg/38-cloud/38-Cloud_training",
        train_pipeline: Dict = None,
        val_pipeline: Dict = None,
        test_pipeline: Dict = None,
        batch_size: int = 2,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_val_split: Tuple[int, int, int] = (5800, 2600),
    ) -> None:
        super().__init__(
            root,
            train_pipeline,
            val_pipeline,
            test_pipeline,
            batch_size,
            num_workers,
            pin_memory,
            persistent_workers,
            train_val_split=train_val_split,
        )

    @property
    def num_classes(self) -> int:
        return len(Cloud38.METAINFO["classes"])

    def prepare_data(self) -> None:
        Cloud38(
            root=self.hparams.root,
            **self.hparams.train_pipeline,
        )

    def setup(self, stage: Optional[str] = None) -> None:

        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            dataset = Cloud38(
                root=self.hparams.root,
                **self.hparams.train_pipeline,
            )
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                dataset,
                self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(42),
            )
            self.test_dataset = self.val_dataset


if __name__ == "__main__":
    root = "/data/zouxuechao/cloudseg/38-cloud/38-Cloud_training"
    train_pipeline = val_pipeline = test_pipeline = dict(
        all_transform=albu.Compose([albu.RandomCrop(384, 384)]),
        img_transform=albu.Compose([albu.ToFloat(), ToTensorV2()]),
        ann_transform=None,
    )
    datamodule = Cloud38DataModule(
        root=root,
        train_pipeline=train_pipeline,
        val_pipeline=val_pipeline,
        test_pipeline=test_pipeline,
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    print(len(datamodule.train_dataloader()))
    print(len(datamodule.test_dataloader()))
    print(len(datamodule.val_dataloader()))
    for data in datamodule.train_dataloader():
        print(data['img'].shape,data['ann'].shape,data['img_path'],data['ann_path'])
        break
