from typing import List
from torch.utils.data import DataLoader
import torch
from torchgeo.datasets import random_bbox_assignment
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler, BatchGeoSampler
from src.data.components.l8_biome import L8Biome
from torchgeo.datamodules import GeoDataModule
import matplotlib.pyplot as plt
import albumentations as A
import albumentations.pytorch
from torch import Tensor
from src.utils.stretch import gaussian_stretch
import numpy as np


class L8BiomeDataModule(GeoDataModule):
    def __init__(
        self,
        root: str = "data/l8_biome",
        bands: List[str] = ["B4", "B3", "B2"],
        split=[0.6, 0.2, 0.2],
        length=None, # 29328
        patch_size=512,
        seed=42,
        train_pipeline = {"all_transform": None, "img_transform": None, "ann_transform": None},
        val_pipeline = {"all_transform": None, "img_transform": None, "ann_transform": None},
        test_pipeline = {"all_transform": None, "img_transform": None, "ann_transform": None},
        batch_size: int = 2,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        cache = False,
    ) -> None:
        super().__init__(
            L8Biome,
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
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
        self.cache = cache

    @property
    def num_classes(self) -> int:
        return len(L8Biome.METAINFO["classes"])

    def prepare_data(self) -> None:
        # train
        L8Biome(root=self.root, bands=self.bands, cache=self.cache)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        dataset = L8Biome(root=self.root, bands=self.bands, cache=self.cache)
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


def show_l8_biome():
    train_pipeline = {
        "all_transform": A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
            ], p=1),
            A.pytorch.transforms.ToTensorV2(),
        ]),
        "img_transform": A.Compose([
            A.ToFloat(255),
        ]),
        "ann_transform": None,
    }
    datamodule = L8BiomeDataModule(batch_size=1, train_pipeline=train_pipeline)
    datamodule.setup("fit")
    datamodule.setup("test")
    train_daloader = datamodule.train_dataloader()
    val_daloader = datamodule.val_dataloader()
    test_daloader = datamodule.test_dataloader()
    
    print(len(train_daloader))
    print(len(val_daloader))
    print(len(test_daloader))
    
    import pdb; pdb.set_trace()

    for data in train_daloader:
        img = data["img"].squeeze(0)
        ann = data["ann"].squeeze(0)
        ldc = data["ldc"][0]
    
        plt.figure(figsize=(16, 8))
        
        plt.title(f"{ldc}")
        plt.axis("off")
        
        plt.subplot(1, 2, 1)
        plt.title(f"img")
        plt.axis("off")
        img = (img.permute(1, 2, 0).numpy()*255).astype(np.uint8)
        img = gaussian_stretch(img)
        plt.imshow(img)
        
        plt.subplot(1, 2, 2)
        plt.title(f"ann")
        plt.axis("off")
        ann = ann.numpy()
        palette=(
            (0, 0, 0),
            (85, 85, 85),
            (170, 170, 170),
            (255, 255, 255),
        )
        color_map = np.array(palette)
        color_ann = color_map[ann]
        print(color_ann.shape)
        plt.imshow(color_ann)
        
        plt.savefig("l8_biome.png", bbox_inches="tight", pad_inches=0)
        
        import time
        time.sleep(3)


if __name__ == "__main__":
    show_l8_biome()
