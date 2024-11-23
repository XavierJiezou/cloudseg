from typing import Optional, Literal, Dict, List

from src.data.base_datamodule import BaseDataModule
from src.data.components.gf12ms_whu import GF12MSWHU


class GF12MSWHUDataModule(BaseDataModule):
    def __init__(
            self,
            root: str="data/gf12ms_whu",
            train_pipeline: Dict = {"all_transform": None, "img_transform": None, "ann_transform": None},
            val_pipeline: Dict = {"all_transform": None, "img_transform": None, "ann_transform": None},
            test_pipeline: Dict = {"all_transform": None, "img_transform": None, "ann_transform": None},
            batch_size: int = 2,
            num_workers: int = 0,
            pin_memory: bool = False,
            persistent_workers: bool = False,
            serial: Literal["gf1", "gf2", "all"] = "all",
            bands: List[str] = ["B3", "B2", "B1"],
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
            serial=serial,
            bands=bands,
        )

    @property
    def num_classes(self) -> int:
        return len(GF12MSWHU.METAINFO["classes"])

    def prepare_data(self) -> None:
        # train
        GF12MSWHU(
            root=self.hparams.root,
            phase="train",
            serial=self.hparams.serial,
            bands=self.hparams.bands,
            **self.hparams.train_pipeline,
        )

        # val or test
        GF12MSWHU(
            root=self.hparams.root,
            serial=self.hparams.serial,
            phase="test",
            bands=self.hparams.bands,
            **self.hparams.test_pipeline,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            self.train_dataset = GF12MSWHU(
                root=self.hparams.root,
                phase="train",
                **self.hparams.train_pipeline,
                serial=self.hparams.serial,
                bands=self.hparams.bands,
            )

            self.val_dataset = self.test_dataset = GF12MSWHU(
                root=self.hparams.root,
                phase="test",
                **self.hparams.test_pipeline,
                serial=self.hparams.serial,
                bands=self.hparams.bands,
            )


if __name__ == "__main__":
    _ = GF12MSWHUDataModule()
