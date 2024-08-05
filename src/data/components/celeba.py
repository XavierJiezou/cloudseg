import json
import os
import random

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DalleTransformerPreprocessor(object):
    def __init__(self,
                 size=256,
                 phase='train',
                 additional_targets=None):

        self.size = size
        self.phase = phase
        # ddc: following dalle to use randomcrop
        self.train_preprocessor = albumentations.Compose([albumentations.RandomCrop(height=size, width=size)],
                                                   additional_targets=additional_targets)
        self.val_preprocessor = albumentations.Compose([albumentations.CenterCrop(height=size, width=size)],
                                                   additional_targets=additional_targets)


    def __call__(self, image, **kargs):
        """
        image: PIL.Image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        w, h = image.size
        s_min = min(h, w)

        if self.phase == 'train':
            off_h = int(random.uniform(3*(h-s_min)//8, max(3*(h-s_min)//8+1, 5*(h-s_min)//8)))
            off_w = int(random.uniform(3*(w-s_min)//8, max(3*(w-s_min)//8+1, 5*(w-s_min)//8)))

            image = image.crop((off_w, off_h, off_w + s_min, off_h + s_min))

            # resize image
            t_max = min(s_min, round(9/8*self.size))
            t_max = max(t_max, self.size)
            t = int(random.uniform(self.size, t_max+1))
            image = image.resize((t, t))
            image = np.array(image).astype(np.uint8)
            image = self.train_preprocessor(image=image)
        else:
            if w < h:
                w_ = self.size
                h_ = int(h * w_/w)
            else:
                h_ = self.size
                w_ = int(w * h_/h)
            image = image.resize((w_, h_))
            image = np.array(image).astype(np.uint8)
            image = self.val_preprocessor(image=image)
        return image


class CelebA(Dataset):

    """
    This Dataset can be used for:
    - image-only: setting 'conditions' = []
    - image and multi-modal 'conditions': setting conditions as the list of modalities you need

    To toggle between 256 and 512 image resolution, simply change the 'image_folder'
    """

    def __init__(
        self,
        phase='train',
        size=512,
        test_dataset_size=3000,
        conditions=['seg_mask', 'text', 'sketch'],
        image_folder='data/celeba/image/image_512_downsampled_from_hq_1024',
        text_file='data/celeba/text/captions_hq_beard_and_age_2022-08-19.json',
        mask_folder='data/celeba/mask/CelebAMask-HQ-mask-color-palette_32_nearest_downsampled_from_hq_512_one_hot_2d_tensor',
        sketch_folder='data/celeba/sketch/sketch_1x1024_tensor',
    ):
        self.transform = DalleTransformerPreprocessor(size=size, phase=phase)
        self.conditions = conditions

        self.image_folder = image_folder

        # conditions directory
        self.text_file = text_file
        with open(self.text_file, 'r') as f:
            self.text_file_content = json.load(f)
        if 'seg_mask' in self.conditions:
            self.mask_folder = mask_folder
        if 'sketch' in self.conditions:
            self.sketch_folder = sketch_folder

        # list of valid image names & train test split
        self.image_name_list = list(self.text_file_content.keys())

        # train test split
        if phase == 'train':
            self.image_name_list = self.image_name_list[:-test_dataset_size]
        elif phase == 'test':
            self.image_name_list = self.image_name_list[-test_dataset_size:]
        else:
            raise NotImplementedError
        self.num = len(self.image_name_list)

    def __len__(self):
        return self.num

    def __getitem__(self, index):

        # ---------- (1) get image ----------
        image_name = self.image_name_list[index]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        image = np.array(image).astype(np.uint8)
        image = self.transform(image=image)['image']
        image = image.astype(np.float32)/127.5 - 1.0

        # record into data entry
        if len(self.conditions) == 1:
            data = {
                'image': image,
            }
        else:
            data = {
                'image': image,
                'conditions': {}
            }

        # ---------- (2) get text ----------
        if 'text' in self.conditions:
            text = self.text_file_content[image_name]["Beard_and_Age"].lower()
            # record into data entry
            if len(self.conditions) == 1:
                data['caption'] = text
            else:
                data['conditions']['text'] = text

        # ---------- (3) get mask ----------
        if 'seg_mask' in self.conditions:
            mask_idx = image_name.split('.')[0]
            mask_name = f'{mask_idx}.pt'
            mask_path = os.path.join(self.mask_folder, mask_name)
            mask_one_hot_tensor = torch.load(mask_path)

            # record into data entry
            if len(self.conditions) == 1:
                data['seg_mask'] = mask_one_hot_tensor
            else:
                data['conditions']['seg_mask'] = mask_one_hot_tensor

        # ---------- (4) get sketch ----------
        if 'sketch' in self.conditions:
            sketch_idx = image_name.split('.')[0]
            sketch_name = f'{sketch_idx}.pt'
            sketch_path = os.path.join(self.sketch_folder, sketch_name)
            sketch_one_hot_tensor = torch.load(sketch_path)

            # record into data entry
            if len(self.conditions) == 1:
                data['sketch'] = sketch_one_hot_tensor
            else:
                data['conditions']['sketch'] = sketch_one_hot_tensor
        data["image_name"] = image_name.split('.')[0]
        return data


if __name__ == '__main__':
    # The caption file only has 29999 captions: https://github.com/ziqihuangg/CelebA-Dialog/issues/1
    
    # Testing for `phase`
    train_dataset = CelebA(phase="train")
    test_dataset = CelebA(phase="test")
    assert len(train_dataset)==26999
    assert len(test_dataset)==3000
    
    # Testing for `size`
    size_512 = CelebA(size=512)
    assert size_512[0]['image'].shape == (512, 512, 3)
    assert size_512[0]["conditions"]['seg_mask'].shape == (19, 1024)
    assert size_512[0]["conditions"]['sketch'].shape == (1, 1024)
    size_512 = CelebA(size=256)
    assert size_512[0]['image'].shape == (256, 256, 3)
    assert size_512[0]["conditions"]['seg_mask'].shape == (19, 1024)
    assert size_512[0]["conditions"]['sketch'].shape == (1, 1024)
    
    # Testing for `conditions`
    dataset = CelebA(conditions = ['seg_mask', 'text', 'sketch'])
    image = dataset[0]["image"]
    seg_mask= dataset[0]["conditions"]['seg_mask']
    sketch = dataset[0]["conditions"]['sketch']
    text = dataset[0]["conditions"]['text']
    # show image, seg_mask, sketch in 3x3 grid, and text in title
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    
    # Show image
    ax[0].imshow((image + 1) / 2)
    ax[0].set_title('Image')
    ax[0].axis('off')

    # # Show segmentation mask
    seg_mask = torch.argmax(seg_mask, dim=0).reshape(32, 32).numpy().astype(np.uint8)
    # resize to 512x512 using nearest neighbor interpolation
    seg_mask = Image.fromarray(seg_mask).resize((512, 512), Image.NEAREST)
    seg_mask = np.array(seg_mask)
    ax[1].imshow(seg_mask, cmap='tab20')
    ax[1].set_title('Segmentation Mask')
    ax[1].axis('off')
    
    # # # Show sketch
    sketch = sketch.reshape(32, 32).numpy().astype(np.uint8)
    # resize to 512x512 using nearest neighbor interpolation
    sketch = Image.fromarray(sketch).resize((512, 512), Image.NEAREST)
    sketch = np.array(sketch)
    ax[2].imshow(sketch, cmap='gray')
    ax[2].set_title('Sketch')
    ax[2].axis('off')

    # Add title with text
    fig.suptitle(text, fontsize=16)
    plt.tight_layout()
    plt.savefig('celeba_sample.png')

    # save seg_mask with name such as "27000.png, 270001.png, ..., 279999.png" of test dataset to "/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/zouxuechao/Collaborative-Diffusion/evaluation/CollDiff/real_mask"
    from tqdm import tqdm
    for data in tqdm(test_dataset):
        mask = torch.argmax(data["conditions"]['seg_mask'], dim=0).reshape(32, 32).numpy().astype(np.uint8)
        mask = Image.fromarray(mask).resize((512, 512), Image.NEAREST)
        mask.save(f"/mnt/chongqinggeminiceph1fs/geminicephfs/wx-mm-spr-xxxx/zouxuechao/Collaborative-Diffusion/evaluation/CollDiff/real_mask/{data['image_name']}.png")