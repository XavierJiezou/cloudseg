from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import h5py
import numpy as np
import matplotlib.pyplot as plt


class MNIST(Dataset):
    def __init__(self, h5_file, transform=ToTensor()):
        self.h5_file = h5_file
        self.transform = transform
        # 读取HDF5文件
        with h5py.File(self.h5_file, 'r') as file:
            self.data = []
            self.labels = []
            for i in range(10):
                images = file[str(i)][()]
                for img in images:
                    self.data.append(img)
                    self.labels.append(i)
            self.data = np.array(self.data)
            self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == '__main__':
    mnist_h5_dataset = MNIST('data/mnist.h5')

    assert len(mnist_h5_dataset) == 70000

    # Display the first 10 images of each digit, along with their labels, in a 10x10 grid
    fig, axs = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        images = mnist_h5_dataset.data[mnist_h5_dataset.labels == i]
        for j in range(10):
            axs[i, j].imshow(images[j], cmap='gray')
            axs[i, j].axis('off')
            axs[i, j].set_title(i)
    plt.tight_layout()
    plt.savefig("mnist_h5_dataset.png")
