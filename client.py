import torch
import torch.nn as nn
import copy


class Client:
    """客户端类"""

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

    def set_model(self, global_model):
        """接收来自边缘服务器的全局模型"""
        self.model = copy.deepcopy(global_model)
        self.model.to(self.device)

    def train(self, epochs, learning_rate, step_size, gamma, momentum, weight_decay):
        """
        在本地数据上训练模型

        Args:
            epochs: 本地训练轮数
            learning_rate: 学习率

        Returns:
            训练损失
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")

        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                num_batches += 1

            total_loss += epoch_loss / len(self.data_loader)

        avg_loss = total_loss / epochs
        return avg_loss

    def get_model_parameters(self):
        """返回模型参数"""
        return copy.deepcopy(self.model.state_dict())
