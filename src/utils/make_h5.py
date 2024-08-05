import os

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


class MNISTH5Creator:
    def __init__(self, data_dir, h5_file):
        self.data_dir = data_dir
        self.h5_file = h5_file

    def create_h5_file(self):
        """创建HDF5文件，包含从0到9的子目录中的所有图像数据。"""
        with h5py.File(self.h5_file, 'w') as h5f:
            for i in range(10):
                images = self.load_images_for_digit(i)
                h5f.create_dataset(name=str(i), data=images)
        print("HDF5文件已创建.")

    def load_images_for_digit(self, digit):
        """为给定的数字加载所有图像，并将它们转换为numpy数组。"""
        digit_folder = os.path.join(self.data_dir, str(digit))
        images = []
        for img_name in tqdm(os.listdir(digit_folder), desc=f"Loading images for digit {digit}"):
            img_path = os.path.join(digit_folder, img_name)
            img = Image.open(img_path).convert('L')
            img_array = np.array(img)
            images.append(img_array)
        return images

if __name__ == "__main__":
    data_dir = 'data/mnist'
    h5_file = 'data/mnist.h5'
    mnist_h5_creator = MNISTH5Creator(data_dir, h5_file)
    mnist_h5_creator.create_h5_file()
