from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components.hrcwhu import HRCWHU
from src.data.base_datamodule import BaseDataModule


class HRCWHUDataModule(BaseDataModule):
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
        )


    @property
    def num_classes(self) -> int:
        return len(HRCWHU.METAINFO["classes"])

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # train
        HRCWHU(
            root=self.hparams.root,
            phase="train",
            **self.hparams.train_pipeline,
            seed=self.hparams.seed,
        )
        
        # val or test
        HRCWHU(
            root=self.hparams.root,
            phase="test",
            **self.hparams.test_pipeline,
            seed=self.hparams.seed,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            self.train_dataset = HRCWHU(
                root=self.hparams.root,
                phase="train",
                **self.hparams.train_pipeline,
                seed=self.hparams.seed,
            )
            
            self.val_dataset = self.test_dataset = HRCWHU(
                root=self.hparams.root,
                phase="test",
                **self.hparams.test_pipeline,
                seed=self.hparams.seed,
            )

    


if __name__ == "__main__":
    _ = HRCWHUDataModule()
