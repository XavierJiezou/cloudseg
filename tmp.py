from torchgeo.datasets import random_bbox_assignment
from torchgeo.datamodules import L8BiomeDataModule

from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from torchgeo.datamodules import InriaAerialImageLabelingDataModule
from torchgeo.datasets import CDL, Landsat7, Landsat8, VHR10, stack_samples
from torchgeo.samplers import RandomGeoSampler, RandomBatchGeoSampler
from torchgeo.trainers import SemanticSegmentationTask
from src.data.components.l8_biome import L8Biome
# from torchgeo.datasets.l8biome import L8Biome
# Initialize the dataset
dataset = L8Biome(
    root="data/l8_biome",
    bands=("B4", "B3", "B2") # default: ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11')
)

import torch
generator = torch.Generator().manual_seed(0)
(train_dataset, val_dataset, test_dataset) = (
    random_bbox_assignment(dataset, [0.8, 0.1, 0.1], generator)
)

sampler = RandomBatchGeoSampler(train_dataset, 512,1)

dataloader = DataLoader(train_dataset, batch_size=1, sampler=None, batch_sampler=sampler, collate_fn=stack_samples)

print(len(dataloader))

for batch in dataloader:
    image = batch["image"]
    mask = batch["mask"]
    landcover = batch["landcover"]
    print(image.shape)
    print(image.dtype)
    print(mask.shape)
    print(mask.dtype)
    print(landcover)
    break 

    # train a model, or make predictions using a pre-trained model


datamodule = L8BiomeDataModule(
    patch_size=512,
    paths="/data/zouxuechao/cloudseg/l8_biome",
    batch_size=1,           # Adjust batch size as needed
    num_workers=0,           # Adjust the number of workers for parallel data loading
)
datamodule.setup("fit")
datamodule.setup("test")
print(len(datamodule.train_dataloader()))
print(len(datamodule.val_dataloader()))
print(len(datamodule.test_dataloader()))


# datamodule.prepare_data()

# datamodule.setup(stage='fit')  # 'fit' for training, 'test' for testing

# train_loader = datamodule.train_dataloader()
# print(len(train_loader))
# for batch in train_loader:
#     image = batch["image"]  # list of images
#     mask = batch["mask"]  # list of boxes
#     print(image.shape)
#     print(image.dtype)
#     print(image.min(), image.max())
#     print(mask.shape)
#     print(mask.dtype)
#     print(mask.min(), mask.max())
#     import matplotlib.pyplot as plt
#     plt.subplot(121)
#     image = (image-image.min())/(image.max()-image.min()+1e-6)
#     plt.imshow(image[0][:3].permute(1, 2, 0))
#     plt.subplot(122)
#     plt.imshow(mask[0].permute(1, 2, 0))
#     plt.savefig("test.png")