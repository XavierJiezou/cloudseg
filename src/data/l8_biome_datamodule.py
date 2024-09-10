from typing import Any, Dict, Optional, List

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
from torchgeo.datasets import random_bbox_assignment
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler, BatchGeoSampler
from src.data.components.l8_biome import L8Biome
from torchgeo.datamodules import GeoDataModule
from torch import Tensor


class L8BiomeDataModule(GeoDataModule):
    def __init__(
        self,
        root: str = "data/l8_biome",
        bands: List[str] = ["B4", "B3", "B2"],
        split=[0.6, 0.2, 0.2],
        patch_size=512,
        seed=42,
        train_pipeline = {"all_transform": None, "img_transform": None, "ann_transform": None},
        val_pipeline = {"all_transform": None, "img_transform": None, "ann_transform": None},
        test_pipeline = {"all_transform": None, "img_transform": None, "ann_transform": None},
        batch_size: int = 2,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__(
            L8Biome,
            batch_size=batch_size,
            patch_size=patch_size,
            num_workers=num_workers,
            pin_memory = pin_memory,
            persistent_workers = persistent_workers
        )

        self.root = root
        self.bands = bands
        self.split = split
        self.seed = seed
        self.train_pipeline = train_pipeline
        self.test_pipeline = test_pipeline
        self.val_pipeline = val_pipeline

    @property
    def num_classes(self) -> int:
        return len(L8Biome.METAINFO["classes"])

    def prepare_data(self) -> None:
        # train
        L8Biome(root=self.root, bands=self.bands)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        dataset = L8Biome(root=self.root, bands=self.bands)
        generator = torch.Generator().manual_seed(self.seed)
        (self.train_dataset, self.val_dataset, self.test_dataset) = (
            random_bbox_assignment(dataset, self.split, generator)
        )

        self.train_dataset.all_transform = self.train_pipeline["all_transform"]
        self.train_dataset.img_transform = self.train_pipeline["img_transform"]
        self.train_dataset.ann_transform = self.train_pipeline["ann_transform"]

        self.val_dataset.all_transform = self.val_pipeline["all_transform"]
        self.val_dataset.img_transform = self.val_pipeline["img_transform"]
        self.val_dataset.ann_transform = self.val_pipeline["ann_transform"]

        self.test_dataset.all_transform = self.test_pipeline["all_transform"]
        self.test_dataset.img_transform = self.test_pipeline["img_transform"]
        self.test_dataset.ann_transform = self.test_pipeline["ann_transform"]

        if stage in ["fit"]:
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset, self.patch_size, self.batch_size, self.length
            )
        if stage in ["fit", "validate"]:
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.patch_size, self.patch_size
            )
        if stage in ["test"]:
            self.test_sampler = GridGeoSampler(
                self.test_dataset, self.patch_size, self.patch_size
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        sampler = self._valid_attribute(
            f"{split}_batch_sampler", f"{split}_sampler", "batch_sampler", "sampler"
        )
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")

        if isinstance(sampler, BatchGeoSampler):
            batch_size = 1
            batch_sampler = sampler
            sampler = None
        else:
            batch_sampler = None

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.kwargs["pin_memory"],
            persistent_workers=self.kwargs["persistent_workers"],
        )


if __name__ == "__main__":
    import albumentations as A
    import albumentations.pytorch
    train_pipeline = {
        "all_transform": A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
            ], p=1),
            # A.pytorch.transforms.ToTensorV2(),
        ]),
        "img_transform": None,
        "ann_transform": None,
    }
    data_module = L8BiomeDataModule(root="data/l8_biome",batch_size=1)
    data_module.setup("fit")
    data_module.setup("test")
    train_daloader = data_module.train_dataloader()

    for index, data in enumerate(train_daloader):
        print(f"data['img'].dtype: {data['img'].dtype}, data['img'].shape: {data['img'].shape}, data['img'].min: {data['img'].min()}, data['img'].max: {data['img'].max()}")
        print(f"data['ann'].dtype: {data['ann'].dtype}, data['ann'].shape: {data['ann'].shape}, data['ann'].min: {data['ann'].min()}, data['ann'].max: {data['ann'].max()}")
        break
