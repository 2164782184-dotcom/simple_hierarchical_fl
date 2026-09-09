import torch
import torch.nn as nn
import copy
from privacy import Flatten_gradients, dp_process_full, reshape_gradients, topindex


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
              lr_decay=1.0, lr_decay_epoch=1, use_dp=False, dp_config=None,
              mu=0.01):
        """
        在本地数据上训练模型（FedProx算法）

        Args:
            epochs: 本地训练轮数
            learning_rate: 初始学习率
            momentum: SGD动量参数（0表示不使用动量）
            weight_decay: 权重衰减/L2正则化系数（0表示不使用）
            lr_decay: 学习率衰减系数
            lr_decay_epoch: 每多少轮衰减一次学习率
            use_dp: 是否使用差分隐私
            dp_config: 差分隐私配置参数
            mu: FedProx近端项系数（0表示退化为FedAvg）

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

        # FedProx: 保存全局模型参数（近端项的参考点）
        global_params = copy.deepcopy(list(self.model.parameters()))

        total_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)

                # 标准交叉熵损失
                loss = criterion(output, target)

                # FedProx近端项: (mu/2) * ||w - w_global||^2
                if mu > 0:
                    proximal_term = 0.0
                    for local_param, global_param in zip(self.model.parameters(), global_params):
                        proximal_term += torch.sum((local_param - global_param) ** 2)
                    loss += (mu / 2) * proximal_term

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # 每个epoch后更新学习率
            scheduler.step()

            total_loss += epoch_loss / len(self.data_loader)

        avg_loss = total_loss / epochs
        return avg_loss

    def get_model_parameters(self, use_dp=False, dp_config=None, use_compression=False, compression_rate=50):
        """
        返回模型参数（支持差分隐私处理和梯度压缩）

        Args:
            use_dp: 是否使用差分隐私
            dp_config: 差分隐私配置参数
            use_compression: 是否使用梯度压缩（Top-k稀疏化）
            compression_rate: 压缩率（rate=50 表示保留 2% 的梯度）

        Returns:
            根据配置返回不同格式：
            - 无DP无压缩：完整模型参数字典
            - 只DP：(加噪后的完整梯度, None, 梯度形状)
            - 只压缩：(非零值, top-k索引, 梯度形状)
            - DP+压缩：(非零值, top-k索引, 梯度形状) - 先DP再Top-k
        """
        if not use_dp and not use_compression:
            # 场景1：不使用任何技术，直接返回完整模型参数
            return copy.deepcopy(self.model.state_dict())

        elif use_dp and not use_compression:
            # 场景2：只使用DP（对所有梯度加噪）
            return self._get_dp_gradients(dp_config)

        elif use_compression and not use_dp:
            # 场景3：只使用压缩（Top-k），不加噪
            return self._get_compressed_gradients(compression_rate)

        else:
            # 场景4：DP + 压缩（先DP加噪，再Top-k选择）
            return self._get_dp_then_compressed_gradients(dp_config, compression_rate)

    def _get_dp_gradients(self, dp_config):
        """
        纯差分隐私处理：对所有梯度加噪（不做Top-k）

        Args:
            dp_config: 差分隐私配置

        Returns:
            noisy_gradient: 加噪后的完整梯度向量
            None: 无索引（因为是完整梯度）
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

        # 纯DP处理：裁剪 + 加噪（不做Top-k）
        noisy_gradient = dp_process_full(flattened_grad, dp_config, self.device)

        # 返回完整的加噪梯度
        return noisy_gradient, None, shapes

    def _get_dp_then_compressed_gradients(self, dp_config, compression_rate):
        """
        先DP加噪，再Top-k压缩

        Args:
            dp_config: 差分隐私配置
            compression_rate: 压缩率

        Returns:
            non_zero_values: 选中的非零值
            choices: top-k 索引
            shapes: 梯度形状列表
        """
        # 步骤1：DP处理（得到加噪后的完整梯度）
        noisy_gradient, _, shapes = self._get_dp_gradients(dp_config)

        # 步骤2：Top-k选择
        dimension = noisy_gradient.numel()
        topk = int(dimension / compression_rate)

        # 找出加噪后梯度中绝对值最大的top-k个
        abs_grad = torch.abs(noisy_gradient)
        _, choices = torch.topk(abs_grad, topk)
        choices = choices.tolist()

        # 只传输选中的值
        non_zero_values = noisy_gradient[choices]

        return non_zero_values, choices, shapes


    def _get_compressed_gradients(self, compression_rate):
        """
        计算并压缩梯度（只使用 Top-k 稀疏化，不加噪）

        Args:
            compression_rate: 压缩率（rate=50 表示保留 2% 的梯度）

        Returns:
            non_zero_values: 只包含非零值的向量（压缩传输）
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

        # 只进行 Top-k 稀疏化，不加噪
        dimension = flattened_grad.numel()
        topk = int(dimension / compression_rate)

        # 找出 top-k 索引
        abs_grad = torch.abs(flattened_grad)
        _, choices = torch.topk(abs_grad, topk)
        choices = choices.tolist()

        # Top-k压缩：只上传选中的值及其位置
        non_zero_values = flattened_grad[choices]

        return non_zero_values, choices, shapes


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
