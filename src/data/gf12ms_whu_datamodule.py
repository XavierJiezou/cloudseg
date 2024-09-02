from typing import Optional, Literal

from src.data.base_datamodule import BaseDataModule
from src.data.components.gaofen12 import Gaofen12


class GF12MSWHUDataModule(BaseDataModule):
    def __init__(
            self,
            root: str,
            train_pipeline: None,
            val_pipeline: None,
            test_pipeline: None,
            batch_size: int = 1,
            num_workers: int = 0,
            pin_memory: bool = False,
            persistent_workers: bool = False,
            serial: Literal["gaofen1", "gaofen2", "all"] = "all",
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
            serial=serial
        )

    @property
    def num_classes(self) -> int:
        return len(Gaofen12.METAINFO["classes"])

    def prepare_data(self) -> None:
        # train
        Gaofen12(
            root=self.hparams.root,
            phase="train",
            serial=self.hparams.serial,
            **self.hparams.train_pipeline,
        )

        # val or test
        Gaofen12(
            root=self.hparams.root,
            serial=self.hparams.serial,
            phase="test",
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
            self.train_dataset = Gaofen12(
                root=self.hparams.root,
                phase="train",
                **self.hparams.train_pipeline,
            )

            self.val_dataset = self.test_dataset = Gaofen12(
                root=self.hparams.root,
                phase="test",
                **self.hparams.test_pipeline,
            )


if __name__ == "__main__":
    _ = GF12MSWHUDataModule()
