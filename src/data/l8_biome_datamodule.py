from typing import Any, Dict, Optional, List

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components.l8_biome import L8Biome
from src.data.base_datamodule import BaseDataModule


class L8BiomeDataModule(BaseDataModule):
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
        bands: List[str] = ["2", "3", "4"],
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
            bands=bands
        )


    @property
    def num_classes(self) -> int:
        return len(L8Biome.METAINFO["classes"])

    def prepare_data(self) -> None:
        # train
        L8Biome(
            root=self.hparams.root,
            phase="train",
            bands=self.hparams.bands,
            **self.hparams.train_pipeline,
        )
        
        # val or test
        L8Biome(
            root=self.hparams.root,
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
            self.train_dataset = L8Biome(
                root=self.hparams.root,
                bands=self.hparams.bands,
                phase="train",
                **self.hparams.train_pipeline,
            )
            
            self.val_dataset = self.test_dataset = L8Biome(
                root=self.hparams.root,
                bands=self.hparams.bands,
                phase="test",
                **self.hparams.test_pipeline,
            )

    


if __name__ == "__main__":
    _ = L8BiomeDataModule()
