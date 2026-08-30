import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """LeNet-5 模型用于MNIST分类（支持特征提取）"""
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
            return_features: 是否返回中间层特征（用于知识蒸馏）

        Returns:
            如果 return_features=False: 返回输出 logits
            如果 return_features=True: 返回 (logits, features_dict)
        """
        # 第一个卷积层
        x1 = F.max_pool2d(F.relu(self.conv1(x)), 2)

        # 第二个卷积层
        x2 = F.max_pool2d(F.relu(self.conv2(x1)), 2)

        # 展平
        x_flat = x2.view(-1, 16 * 4 * 4)

        # 全连接层
        x3 = F.relu(self.fc1(x_flat))
        x4 = F.relu(self.fc2(x3))
        x5 = self.fc3(x4)

        if return_features:
            # 返回多个层的特征用于蒸馏
            features = {
                'conv1': x1,      # 第一个卷积层输出
                'conv2': x2,      # 第二个卷积层输出
                'fc1': x3,        # 第一个全连接层输出
                'fc2': x4,        # 第二个全连接层输出（倒数第二层）
            }
            return x5, features
        else:
            return x5


def get_model(model_name='lenet'):
    """
    获取模型

    Args:
        model_name: 模型名称（目前只支持 'lenet'）

    Returns:
        model: 模型实例
    """
    if model_name.lower() == 'lenet':
        return LeNet()
    else:
        raise ValueError(f"Unknown model: {model_name}")
