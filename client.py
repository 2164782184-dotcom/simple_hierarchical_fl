import torch
import torch.nn as nn
import copy
from privacy import Flatten_gradients, local_process, reshape_gradients


class Client:
    """客户端类（支持差分隐私）"""

    def __init__(self, client_id, data_loader, device='cpu'):
        """
        初始化客户端

        Args:
            client_id: 客户端ID
            data_loader: 该客户端的数据加载器
            device: 计算设备
        """
        self.client_id = client_id
        self.data_loader = data_loader
        self.device = device
        self.model = None
        self.initial_params = None  # 保存初始参数用于计算梯度

    def set_model(self, global_model):
        """接收来自边缘服务器的全局模型"""
        self.model = copy.deepcopy(global_model)
        self.model.to(self.device)
        # 保存初始参数
        self.initial_params = copy.deepcopy(self.model.state_dict())

    def train(self, epochs, learning_rate, momentum=0, weight_decay=0,
              lr_decay=1.0, lr_decay_epoch=1, use_dp=False, dp_config=None):
        """
        在本地数据上训练模型

        Args:
            epochs: 本地训练轮数
            learning_rate: 初始学习率
            momentum: SGD动量参数（0表示不使用动量）
            weight_decay: 权重衰减/L2正则化系数（0表示不使用）
            lr_decay: 学习率衰减系数
            lr_decay_epoch: 每多少轮衰减一次学习率
            use_dp: 是否使用差分隐私
            dp_config: 差分隐私配置参数

        Returns:
            avg_loss: 平均训练损失
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")

        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(),
                                     lr=learning_rate,
                                     momentum=momentum,
                                     weight_decay=weight_decay)

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                     step_size=lr_decay_epoch,
                                                     gamma=lr_decay)

        total_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # 每个epoch后更新学习率
            scheduler.step()

            total_loss += epoch_loss / len(self.data_loader)

        avg_loss = total_loss / epochs
        return avg_loss

    def get_model_parameters(self, use_dp=False, dp_config=None):
        """
        返回模型参数（支持差分隐私处理）

        Args:
            use_dp: 是否使用差分隐私
            dp_config: 差分隐私配置参数

        Returns:
            如果不使用差分隐私：返回完整的模型参数字典
            如果使用差分隐私：返回 (处理后的梯度, top-k索引, 梯度形状)
        """
        if not use_dp:
            # 不使用差分隐私，直接返回模型参数
            return copy.deepcopy(self.model.state_dict())
        else:
            # 使用差分隐私，返回处理后的梯度
            return self._get_private_gradients(dp_config)

    def _get_private_gradients(self, dp_config):
        """
        计算并处理差分隐私梯度

        Args:
            dp_config: 差分隐私配置

        Returns:
            processed_gradient: 处理后的梯度向量
            choices: top-k 索引
            shapes: 梯度形状列表
        """
        # 计算梯度（当前参数 - 初始参数）
        gradients = []
        current_params = self.model.state_dict()

        for key in current_params.keys():
            grad = current_params[key] - self.initial_params[key]
            gradients.append(grad)

        # 展平梯度
        flattened_grad, shapes = Flatten_gradients(gradients)

        # 应用差分隐私处理（裁剪 + 稀疏化 + 加噪）
        dimension = flattened_grad.numel()
        processed_gradient, choices = local_process(
            flattened_grad,
            dp_config,
            dimension
        )

        return processed_gradient, choices, shapes


class DPConfig:
    """差分隐私配置类"""
    def __init__(self, epsilon=1.0, delta=1e-5, clip_C=1.0, rate=50, mechanism='laplace'):
        """
        Args:
            epsilon: 隐私预算，越小隐私保护越强（典型值：0.1-10）
            delta: 失败概率（典型值：1e-5 到 1e-7）
            clip_C: 梯度裁剪阈值
            rate: 稀疏化率（rate=50 表示保留 2% 的梯度）
            mechanism: 噪声机制（'laplace' 或 'gaussian'）
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_C = clip_C
        self.rate = rate
        self.mechanism = mechanism
