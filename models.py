import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """LeNet-5 模型用于MNIST分类"""
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, return_features=False):
        """
        前向传播

        Args:
            x: 输入数据
            return_features: 是否返回中间特征（用于特征蒸馏）

        Returns:
            如果 return_features=False: 返回 logits
            如果 return_features=True: 返回 (logits, features_dict)
        """
        x1 = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x2 = F.max_pool2d(F.relu(self.conv2(x1)), 2)
        x2_flat = x2.view(-1, 16 * 4 * 4)
        x3 = F.relu(self.fc1(x2_flat))
        x4 = F.relu(self.fc2(x3))
        x5 = self.fc3(x4)

        if return_features:
            features = {
                'conv1': x1,    # (batch, 6, 12, 12)
                'conv2': x2,    # (batch, 16, 4, 4)
                'fc1': x3,      # (batch, 120)
                'fc2': x4,      # (batch, 84) - 通常用于特征蒸馏
            }
            return x5, features
        return x5


def get_model():
    """返回一个新的模型实例"""
    return LeNet()
