import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from privacy import Flatten_gradients, local_process, reshape_gradients


class Client:
    """客户端类（支持差分隐私 + 知识蒸馏）"""

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
        self.teacher_model = None  # 教师模型（用于知识蒸馏）

    def set_model(self, global_model):
        """接收来自边缘服务器的全局模型"""
        self.model = copy.deepcopy(global_model)
        self.model.to(self.device)
        # 保存初始参数
        self.initial_params = copy.deepcopy(self.model.state_dict())

    def set_teacher_model(self, teacher_model_state):
        """设置教师模型（用于知识蒸馏）"""
        if self.teacher_model is None:
            from models import get_model
            self.teacher_model = get_model()
        self.teacher_model.load_state_dict(teacher_model_state)
        self.teacher_model.to(self.device)
        self.teacher_model.eval()

    def train(self, epochs, learning_rate, momentum=0, weight_decay=0,
              lr_decay=1.0, lr_decay_epoch=1, use_dp=False, dp_config=None,
              use_distillation=False, temperature=3.0, alpha=0.5, beta_feat=0.3,
              use_dp_distillation=False, weight_adjuster=None):
        """
        在本地数据上训练模型（标准训练+知识蒸馏）

        Args:
            epochs: 本地训练轮数
            learning_rate: 学习率
            momentum: SGD动量参数（0表示不使用动量）
            weight_decay: 权重衰减/L2正则化系数（0表示不使用）
            lr_decay: 学习率衰减系数
            lr_decay_epoch: 每多少轮衰减一次学习率
            use_dp: 是否使用差分隐私（用于梯度上传）
            dp_config: 差分隐私配置参数
            use_distillation: 是否使用知识蒸馏
            temperature: 蒸馏温度参数
            alpha: KL蒸馏损失权重
            beta_feat: 特征蒸馏损失权重（MSE）
            use_dp_distillation: 是否对蒸馏过程使用差分隐私
            weight_adjuster: 动态权重调整器

        Returns:
            avg_loss: 平均训练损失
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")

        return self._train_standard(epochs, learning_rate, momentum,
                                   weight_decay, lr_decay, lr_decay_epoch,
                                   use_distillation, temperature, alpha, beta_feat,
                                   use_dp_distillation, dp_config, weight_adjuster)

    def _train_standard(self, epochs, learning_rate, momentum,
                       weight_decay, lr_decay, lr_decay_epoch,
                       use_distillation=False, temperature=3.0, alpha=0.5, beta_feat=0.3,
                       use_dp_distillation=False, dp_config=None, weight_adjuster=None):
        """标准训练方法（支持知识蒸馏）"""
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        mse_criterion = nn.MSELoss()
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

                # 学生模型前向传播
                if use_distillation and self.teacher_model is not None:
                    student_output, student_features = self.model(data, return_features=True)
                else:
                    student_output = self.model(data, return_features=False)

                # 1. 交叉熵损失（硬标签）
                ce_loss = criterion(student_output, target)

                # 2. KL散度 + 3. 特征MSE（如果启用蒸馏）
                if use_distillation and self.teacher_model is not None:
                    with torch.no_grad():
                        teacher_output, teacher_features = self.teacher_model(data, return_features=True)

                        # 对教师模型的输出添加差分隐私噪声
                        if use_dp_distillation and dp_config is not None:
                            teacher_output = self._add_noise_to_teacher_output(teacher_output, dp_config)
                            teacher_features['fc2'] = self._add_noise_to_teacher_features(
                                teacher_features['fc2'], dp_config
                            )

                    # KL散度损失（软标签）
                    kl_loss = self._compute_kl_loss(student_output, teacher_output, temperature)

                    # 特征MSE损失（使用fc2层特征）
                    # 使用标准化而非归一化，保留幅度信息但统一尺度
                    teacher_feat_mean = teacher_features['fc2'].mean()
                    teacher_feat_std = teacher_features['fc2'].std() + 1e-8

                    student_feat_scaled = (student_features['fc2'] - student_features['fc2'].mean()) / (student_features['fc2'].std() + 1e-8)
                    teacher_feat_scaled = (teacher_features['fc2'] - teacher_feat_mean) / teacher_feat_std

                    feat_loss = mse_criterion(student_feat_scaled, teacher_feat_scaled)

                    # 动态调整权重（如果启用）
                    if weight_adjuster is not None:
                        alpha, beta_feat = weight_adjuster.update(
                            ce_loss.item(), kl_loss.item(), feat_loss.item()
                        )

                    # 三组件组合损失：loss = (1-α-β)*CE + α*KL + β*MSE
                    loss = (1 - alpha - beta_feat) * ce_loss + alpha * kl_loss + beta_feat * feat_loss

                    # 诊断：打印各损失分量（只在第一个batch打印）
                    if batch_idx == 0 and epoch == 0:
                        if weight_adjuster is not None:
                            print(f"[客户端{self.client_id}] CE: {ce_loss.item():.4f}, KL: {kl_loss.item():.4f}, MSE: {feat_loss.item():.4f}, α: {alpha:.3f}, β: {beta_feat:.3f}, Total: {loss.item():.4f}")
                        else:
                            print(f"[客户端{self.client_id}] CE: {ce_loss.item():.4f}, KL: {kl_loss.item():.4f}, MSE: {feat_loss.item():.4f}, Total: {loss.item():.4f}")
                else:
                    loss = ce_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # 每个epoch后更新学习率
            scheduler.step()

            total_loss += epoch_loss / len(self.data_loader)

        avg_loss = total_loss / epochs
        return avg_loss

    def _compute_kl_loss(self, student_logits, teacher_logits, temperature):
        """计算KL散度损失"""
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        return kl_loss * (temperature ** 2)

    def _add_noise_to_teacher_output(self, teacher_output, dp_config):
        """
        对教师模型的输出logits添加拉普拉斯噪声

        Args:
            teacher_output: 教师模型的logits (batch, num_classes)
            dp_config: 差分隐私配置

        Returns:
            加噪后的teacher_output
        """
        if dp_config.mechanism == 'laplace':
            # 拉普拉斯噪声：scale = sensitivity / epsilon
            # 对于logits，sensitivity可以设为一个较小的值
            sensitivity = 1.0
            scale = sensitivity / dp_config.epsilon
            noise = torch.from_numpy(
                np.random.laplace(0, scale, teacher_output.shape)
            ).float().to(self.device)
        elif dp_config.mechanism == 'gaussian':
            # 高斯噪声
            sensitivity = 1.0
            sigma = sensitivity * np.sqrt(2 * np.log(1.25 / dp_config.delta)) / dp_config.epsilon
            noise = torch.randn_like(teacher_output) * sigma
        else:
            noise = 0

        return teacher_output + noise

    def _add_noise_to_teacher_features(self, teacher_features, dp_config):
        """
        对教师模型的特征添加拉普拉斯噪声

        Args:
            teacher_features: 教师模型的特征 (batch, feature_dim)
            dp_config: 差分隐私配置

        Returns:
            加噪后的特征
        """
        # 先对特征进行L2范数裁剪
        feature_norm = torch.norm(teacher_features, p=2, dim=1, keepdim=True)
        clipped_features = teacher_features * torch.clamp(
            dp_config.clip_C / (feature_norm + 1e-8), max=1.0
        )

        # 添加噪声
        if dp_config.mechanism == 'laplace':
            sensitivity = dp_config.clip_C
            scale = sensitivity / dp_config.epsilon
            noise = torch.from_numpy(
                np.random.laplace(0, scale, clipped_features.shape)
            ).float().to(self.device)
        elif dp_config.mechanism == 'gaussian':
            sensitivity = dp_config.clip_C
            sigma = sensitivity * np.sqrt(2 * np.log(1.25 / dp_config.delta)) / dp_config.epsilon
            noise = torch.randn_like(clipped_features) * sigma
        else:
            noise = 0

        return clipped_features + noise

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
            dimension, self.device
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
