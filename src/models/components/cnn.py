import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, dim=32):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, dim, 5)
        self.conv2 = nn.Conv2d(dim, dim * 2, 5)
        self.fc1 = nn.Linear(dim * 2 * 4 * 4, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    input = torch.randn(2, 1, 28, 28)
    model = CNN()
    output = model(input)
    assert output.shape == (2, 10)