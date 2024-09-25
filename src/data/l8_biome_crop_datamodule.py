from typing import Optional, Dict, List, Sequence

from src.data.base_datamodule import BaseDataModule
from src.data.components.l8_biome_crop import L8BiomeCrop


class L8BiomeCropDataModule(BaseDataModule):
    def __init__(
            self,
            root: str="data/l8_biome_crop",
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
            bands=bands,
        )
        self.root = root
        self.bands = bands
        self.train_pipeline = train_pipeline
        self.test_pipeline = test_pipeline
        self.val_pipeline = val_pipeline

    @property
    def num_classes(self) -> int:
        return len(L8BiomeCrop.METAINFO["classes"])

    def prepare_data(self) -> None:
        L8BiomeCrop(
            root=self.hparams.root,
            bands=self.hparams.bands,
            phase="train",
            **self.hparams.train_pipeline,
        )
        
        L8BiomeCrop(
            root=self.hparams.root,
            bands=self.hparams.bands,
            phase="val",
            **self.hparams.val_pipeline,
        )
        
        L8BiomeCrop(
            root=self.hparams.root,
            bands=self.hparams.bands,
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
            self.train_dataset = L8BiomeCrop(
                root=self.hparams.root,
                bands=self.hparams.bands,
                phase="train",
                **self.hparams.train_pipeline,
            )
            
            self.val_dataset = L8BiomeCrop(
                root=self.hparams.root,
                bands=self.hparams.bands,
                phase="val",
                **self.hparams.val_pipeline,
            )
            
            self.test_dataset = L8BiomeCrop(
                root=self.hparams.root,
                bands=self.hparams.bands,
                phase="test",
                **self.hparams.test_pipeline,
            )
            

if __name__ == "__main__":
    datamodule = L8BiomeCropDataModule()
    datamodule.prepare_data()
    datamodule.setup()
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    test_dataset = datamodule.test_dataset
    print(f"len(train_dataset) = {len(train_dataset)}") # 7931
    print(f"len(val_dataset) = {len(val_dataset)}") # 2643
    print(f"len(test_dataset) = {len(test_dataset)}") # 2643