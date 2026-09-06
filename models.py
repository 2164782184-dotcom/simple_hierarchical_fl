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


class MnistTeacher(nn.Module):
    """MNIST教师模型 - 用于知识蒸馏"""
    def __init__(self, num_classes=10):
        super(MnistTeacher, self).__init__()
        # 两个卷积层：通道数为10和20
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # 两个全连接层
        # 第一个FC层生成50维特征表示用于知识蒸馏
        self.fc1 = nn.Linear(20 * 4 * 4, 50)
        # 第二个FC层用于分类
        self.fc2 = nn.Linear(50, num_classes)

        # Dropout正则化以提高泛化能力
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x, return_features=False):
        """
        前向传播

        Args:
            x: 输入数据 (batch, 1, 28, 28)
            return_features: 是否返回中间特征用于知识蒸馏

        Returns:
            如果 return_features=False: 返回 logits
            如果 return_features=True: 返回 (logits, features_dict)
        """
        # 卷积层 + 池化 + 激活
        x1 = F.relu(self.conv1(x))
        x1 = F.max_pool2d(x1, 2)
        x1 = self.dropout1(x1)

        x2 = F.relu(self.conv2(x1))
        x2 = F.max_pool2d(x2, 2)
        x2 = self.dropout1(x2)

        # 展平
        x2_flat = x2.view(-1, 20 * 4 * 4)

        # 第一个全连接层：生成50维蒸馏特征
        distill_features = F.relu(self.fc1(x2_flat))
        distill_features = self.dropout2(distill_features)

        # 第二个全连接层：分类
        logits = self.fc2(distill_features)

        if return_features:
            features = {
                'conv1': x1,                          # (batch, 10, 12, 12)
                'conv2': x2,                          # (batch, 20, 4, 4)
                'distill_features': distill_features,  # (batch, 50) - 用于知识蒸馏
            }
            return logits, features
        return logits


class MnistStudent(nn.Module):
    """MNIST学生模型 - 更紧凑的设计，参数更少"""
    def __init__(self, num_classes=10):
        super(MnistStudent, self).__init__()
        # 两个卷积层：通道数为6和16（更少的参数）
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # 两个全连接层
        # 保持与教师模型兼容的50维蒸馏特征
        self.fc1 = nn.Linear(16 * 4 * 4, 50)
        self.fc2 = nn.Linear(50, num_classes)

        # 为了架构简单性，移除了Dropout层

    def forward(self, x, return_features=False):
        """
        前向传播

        Args:
            x: 输入数据 (batch, 1, 28, 28)
            return_features: 是否返回中间特征用于知识蒸馏

        Returns:
            如果 return_features=False: 返回 logits
            如果 return_features=True: 返回 (logits, features_dict)
        """
        # 卷积层 + 池化 + 激活
        x1 = F.relu(self.conv1(x))
        x1 = F.max_pool2d(x1, 2)

        x2 = F.relu(self.conv2(x1))
        x2 = F.max_pool2d(x2, 2)

        # 展平
        x2_flat = x2.view(-1, 16 * 4 * 4)

        # 第一个全连接层：生成50维蒸馏特征（与教师模型兼容）
        distill_features = F.relu(self.fc1(x2_flat))

        # 第二个全连接层：分类
        logits = self.fc2(distill_features)

        if return_features:
            features = {
                'conv1': x1,                          # (batch, 6, 12, 12)
                'conv2': x2,                          # (batch, 16, 4, 4)
                'distill_features': distill_features,  # (batch, 50) - 用于知识蒸馏
            }
            return logits, features
        return logits


class CifarTeacher(nn.Module):
    """CIFAR-10教师模型 - 用于知识蒸馏"""
    def __init__(self, num_classes=10):
        super(CifarTeacher, self).__init__()
        # 两个卷积层：通道数为32和64
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 特征适配模块：将特征扩展到128维空间
        self.feature_adaptation = nn.Linear(64 * 8 * 8, 128)

        # 最终全连接层：10类概率分布
        self.fc = nn.Linear(128, num_classes)

        # Dropout正则化
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, return_features=False):
        """
        前向传播

        Args:
            x: 输入数据 (batch, 3, 32, 32)
            return_features: 是否返回中间特征用于知识蒸馏

        Returns:
            如果 return_features=False: 返回 logits
            如果 return_features=True: 返回 (logits, features_dict)
        """
        # 卷积层 + 池化 + 激活
        x1 = F.relu(self.conv1(x))
        x1 = F.max_pool2d(x1, 2)  # (batch, 32, 16, 16)

        x2 = F.relu(self.conv2(x1))
        x2 = F.max_pool2d(x2, 2)  # (batch, 64, 8, 8)

        # 展平
        x2_flat = x2.view(-1, 64 * 8 * 8)

        # 特征适配模块：线性变换 + ReLU激活，扩展到128维
        distill_features = F.relu(self.feature_adaptation(x2_flat))
        distill_features = self.dropout(distill_features)

        # 最终分类层
        logits = self.fc(distill_features)

        if return_features:
            features = {
                'conv1': x1,                          # (batch, 32, 16, 16)
                'conv2': x2,                          # (batch, 64, 8, 8)
                'distill_features': distill_features,  # (batch, 128) - 用于知识蒸馏
            }
            return logits, features
        return logits


class CifarStudent(nn.Module):
    """CIFAR-10学生模型 - 更紧凑的设计"""
    def __init__(self, num_classes=10):
        super(CifarStudent, self).__init__()
        # 两个卷积层：通道数为16和32（更少的参数）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # 特征适配模块：扩展到128维空间（与教师模型兼容）
        self.feature_adaptation = nn.Linear(32 * 8 * 8, 128)

        # 最终全连接层：10类概率分布
        self.fc = nn.Linear(128, num_classes)

        # 为了计算效率，移除了Dropout

    def forward(self, x, return_features=False):
        """
        前向传播

        Args:
            x: 输入数据 (batch, 3, 32, 32)
            return_features: 是否返回中间特征用于知识蒸馏

        Returns:
            如果 return_features=False: 返回 logits
            如果 return_features=True: 返回 (logits, features_dict)
        """
        # 卷积层 + 池化 + 激活
        x1 = F.relu(self.conv1(x))
        x1 = F.max_pool2d(x1, 2)  # (batch, 16, 16, 16)

        x2 = F.relu(self.conv2(x1))
        x2 = F.max_pool2d(x2, 2)  # (batch, 32, 8, 8) - 最终特征图为8×8空间分辨率

        # 展平
        x2_flat = x2.view(-1, 32 * 8 * 8)

        # 特征适配模块：线性变换 + ReLU激活，扩展到128维（与教师模型兼容）
        distill_features = F.relu(self.feature_adaptation(x2_flat))

        # 最终分类层
        logits = self.fc(distill_features)

        if return_features:
            features = {
                'conv1': x1,                          # (batch, 16, 16, 16)
                'conv2': x2,                          # (batch, 32, 8, 8)
                'distill_features': distill_features,  # (batch, 128) - 用于知识蒸馏
            }
            return logits, features
        return logits


def get_model(model_name='lenet'):
    """
    返回一个新的模型实例

    Args:
        model_name: 模型名称，可选值：
            - 'lenet': 原始LeNet模型
            - 'mnist_teacher': MNIST教师模型
            - 'mnist_student': MNIST学生模型
            - 'cifar_teacher': CIFAR-10教师模型
            - 'cifar_student': CIFAR-10学生模型

    Returns:
        模型实例
    """
    if model_name == 'lenet':
        return LeNet()
    elif model_name == 'mnist_teacher':
        return MnistTeacher()
    elif model_name == 'mnist_student':
        return MnistStudent()
    elif model_name == 'cifar_teacher':
        return CifarTeacher()
    elif model_name == 'cifar_student':
        return CifarStudent()
    else:
        raise ValueError(f"未知的模型名称: {model_name}")
