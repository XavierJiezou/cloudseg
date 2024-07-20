import torch
from torch import nn


class LNN(nn.Module):
    # 创建一个全连接网络用于手写数字识别，并通过一个参数dim控制中间层的维度
    def __init__(self, dim=32):
        super(LNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, dim)
        self.fc2 = nn.Linear(dim, 10)
    
    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    input = torch.randn(2, 1, 28, 28)
    model = LNN()
    output = model(input)
    assert output.shape == (2, 10)
