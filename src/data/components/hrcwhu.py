import os

import albumentations
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class HRCWHU(Dataset):
    METAINFO = dict(
        classes=('clear sky', 'cloud'),
        palette=((128, 192, 128), (255, 255, 255)),
        img_size=(3, 256, 256),  # C, H, W
        ann_size=(256, 256),  # C, H, W
        train_size=120,
        test_size=30,
    )

    def __init__(self, root, phase, all_transform: albumentations.Compose = None,
                 img_transform: albumentations.Compose = None,
                 ann_transform: albumentations.Compose = None, seed: int = 42):
        self.root = root
        self.phase = phase
        self.all_transform = all_transform
        self.img_transform = img_transform
        self.ann_transform = ann_transform
        self.seed = seed
        self.data = self.load_data()


    def load_data(self):
        data_list = []
        split = 'train' if self.phase == 'train' else 'test'
        split_file = os.path.join(self.root, f'{split}.txt')
        with open(split_file, 'r') as f:
            for line in f:
                image_file = line.strip()
                img_path = os.path.join(self.root, 'img_dir', split, image_file)
                ann_path = os.path.join(self.root, 'ann_dir', split, image_file)
                lac_type = image_file.split('_')[0]
                data_list.append((img_path, ann_path, lac_type))
        return data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, ann_path, lac_type = self.data[idx]
        img = Image.open(img_path)
        ann = Image.open(ann_path)

        img = np.array(img)
        ann = np.array(ann)

        if self.all_transform:
            albumention = self.all_transform(image=img, mask=ann)
            img = albumention['image']
            ann = albumention['mask']

        if self.img_transform:
            img = self.img_transform(image=img)['image']

        if self.ann_transform:
            ann = self.ann_transform(image=img)['image']

        # if self.img_transform is not None:
        #     img = self.img_transform(img)
        # if self.ann_transform is not None:
        #     ann = self.ann_transform(ann)
        # if self.all_transform is not None:
        #     # 对img和ann实现相同的随机变换操作
        #     # seed_everything(self.seed, workers=True)
        #     # random.seed(self.seed)
        #     # img= self.all_transform(img)
        #     # seed_everything(self.seed, workers=True)
        #     # random.seed(self.seed)
        #     # ann= self.all_transform(ann)
        #     merge = torch.cat((img, ann), dim=0)
        #     merge = self.all_transform(merge)
        #     img = merge[:-1]
        #     ann = merge[-1]

        return {
            'img': img,
            'ann': np.int64(ann),
            'img_path': img_path,
            'ann_path': ann_path,
            'lac_type': lac_type,
        }


if __name__ == '__main__':
    import torchvision.transforms as transforms
    import torch

    # all_transform = transforms.Compose([
    #     transforms.RandomCrop((256, 256)),
    # ])
    all_transform = transforms.RandomCrop((256, 256))

    # img_transform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])

    img_transform = transforms.ToTensor()

    # ann_transform = transforms.Compose([
    #     transforms.PILToTensor(),
    # ])
    ann_transform = transforms.PILToTensor()

    train_dataset = HRCWHU(root='data/hrcwhu', phase='train', all_transform=all_transform, img_transform=img_transform,
                           ann_transform=ann_transform)
    test_dataset = HRCWHU(root='data/hrcwhu', phase='test', all_transform=all_transform, img_transform=img_transform,
                          ann_transform=ann_transform)

    assert len(train_dataset) == train_dataset.METAINFO['train_size']
    assert len(test_dataset) == test_dataset.METAINFO['test_size']

    train_sample = train_dataset[0]
    test_sample = test_dataset[0]

    assert train_sample['img'].shape == test_sample['img'].shape == train_dataset.METAINFO['img_size']
    assert train_sample['ann'].shape == test_sample['ann'].shape == train_dataset.METAINFO['ann_size']

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for train_sample in train_dataset:
        axs[0].imshow(train_sample['img'].permute(1, 2, 0))
        axs[0].set_title('Image')
        axs[1].imshow(torch.tensor(train_dataset.METAINFO['palette'])[train_sample['ann']])
        axs[1].set_title('Annotation')
        plt.suptitle(f'Land Cover Type: {train_sample["lac_type"].capitalize()}', y=0.8)
        plt.tight_layout()
        plt.savefig('HRCWHU_sample.png', bbox_inches="tight")
        # break
