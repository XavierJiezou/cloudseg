from typing import Dict, Optional, Literal, List
from src.data.components.cloudsen12_high import CloudSEN12High
from src.data.base_datamodule import BaseDataModule


class CloudSEN12HighDataModule(BaseDataModule):
    def __init__(
        self,
        root: str = "data/cloudsen12_high",
        level: Literal["l1c", "l2a"] = "l1c",
        bands: List[str] = ["B4", "B3", "B2"],
        train_pipeline: Dict = {"all_transform": None, "img_transform": None, "ann_transform": None},
        val_pipeline: Dict = {"all_transform": None, "img_transform": None, "ann_transform": None},
        test_pipeline: Dict = {"all_transform": None, "img_transform": None, "ann_transform": None},
        batch_size: int = 2,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            train_pipeline=train_pipeline,
            val_pipeline=val_pipeline,
            test_pipeline=test_pipeline,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            level=level,
            bands=bands,
        )

    @property
    def num_classes(self) -> int:
        return len(CloudSEN12High.METAINFO["classes"])

    def prepare_data(self) -> None:
        CloudSEN12High(
            root=self.hparams.root,
            phase="train",
            **self.hparams.train_pipeline,
            level=self.hparams.level,
            bands=self.hparams.bands,
        )

        CloudSEN12High(
            root=self.hparams.root,
            phase="val",
            **self.hparams.val_pipeline,
            level=self.hparams.level,
            bands=self.hparams.bands,
        )

        CloudSEN12High(
            root=self.hparams.root,
            phase="test",
            **self.hparams.test_pipeline,
            level=self.hparams.level,
            bands=self.hparams.bands,
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
            self.train_dataset = CloudSEN12High(
                root=self.hparams.root,
                phase="train",
                **self.hparams.train_pipeline,
                level=self.hparams.level,
                bands=self.hparams.bands,
            )

            self.val_dataset = CloudSEN12High(
                root=self.hparams.root,
                phase="val",
                **self.hparams.train_pipeline,
                level=self.hparams.level,
                bands=self.hparams.bands,
            )

            self.test_dataset = CloudSEN12High(
                root=self.hparams.root,
                phase="test",
                **self.hparams.train_pipeline,
                level=self.hparams.level,
                bands=self.hparams.bands,
            )


if __name__ == "__main__":
#     train_pipeline:
#   all_transform:
#     _target_: albumentations.Compose
#     transforms:
#       - _target_: albumentations.OneOf
#         transforms:
#           - _target_: albumentations.HorizontalFlip
#             p: 0.5
#           - _target_: albumentations.VerticalFlip
#             p: 0.5
#           - _target_: albumentations.RandomRotate90
#             p: 0.5
#           - _target_: albumentations.Transpose
#             p: 0.5
#         p: 1
#     is_check_shapes: False

#   img_transform: null
#   ann_transform: null
    import albumentations as A
    train_pipeline = {
        "all_transform": A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
            ], p=1)
        ]),
        "img_transform": None,
        "ann_transform": None,
    }
    datamodule = CloudSEN12HighDataModule(train_pipeline=train_pipeline)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    for batch in train_dataloader:
        assert batch["img"].shape == (2, 3, 512, 512)
        assert batch["ann"].shape == (2, 512, 512)
    
