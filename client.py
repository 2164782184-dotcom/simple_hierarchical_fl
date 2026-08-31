import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from privacy import Flatten_gradients, local_process, reshape_gradients


class Adam:
    """
    全局Adam优化器（用于MAML外层更新）
    """
    def __init__(self, lr=0.01, betas=(0.9, 0.999), eps=1e-08):
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.m = dict()
        self.v = dict()
        self.n = 0

    def __call__(self, params, grads, i):
        if i not in self.m:
            self.m[i] = torch.zeros_like(params)
        if i not in self.v:
            self.v[i] = torch.zeros_like(params)

        self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads
        self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * torch.square(grads)

        alpha = self.lr * np.sqrt(1 - np.power(self.beta2, self.n))
        alpha = alpha / (1 - np.power(self.beta1, self.n))

        params.sub_(alpha * self.m[i] / (torch.sqrt(self.v[i]) + self.eps))

    def increase_n(self):
        self.n += 1


class Client:
    """客户端类（支持差分隐私 + MAML元学习 + KL蒸馏）"""

    def __init__(self, client_id, data_loader, device='cpu',
                 support_loader=None, query_loader=None,
                 train_inner_step=0, test_inner_step=0):
        """
        初始化客户端

        Args:
            client_id: 客户端ID
            data_loader: 该客户端的数据加载器（标准模式）
            device: 计算设备
            support_loader: support集加载器（MAML模式）
            query_loader: query集加载器（MAML模式）
            train_inner_step: support集采样batch数（0=遍历全部）
            test_inner_step: query集采样batch数（0=遍历全部）
        """
        self.client_id = client_id
        self.data_loader = data_loader
        self.device = device
        self.model = None
        self.initial_params = None  # 保存初始参数用于计算梯度

        # MAML相关
        self.support_loader = support_loader
        self.query_loader = query_loader
        self.teacher_model = None  # 教师模型（用于KL蒸馏）
        self.train_inner_step = train_inner_step
        self.test_inner_step = test_inner_step
        self.outer_opt = None  # Adam优化器（外层更新）

        # 初始化mini-batch迭代器
        if self.support_loader and self.is_train_mini_batch:
            self.train_dataset_loader_iterator = iter(self.support_loader)
        if self.query_loader and self.is_test_mini_batch:
            self.test_dataset_loader_iterator = iter(self.query_loader)

    def set_model(self, global_model):
        """接收来自边缘服务器的全局模型"""
        self.model = copy.deepcopy(global_model)

        # 多GPU支持
        if torch.cuda.device_count() > 1 and self.device.type == 'cuda':
            print(f"客户端 {self.client_id} 使用 {torch.cuda.device_count()} 个GPU进行训练")
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        # 保存初始参数
        self.initial_params = copy.deepcopy(self.model.state_dict())

    def set_teacher_model(self, teacher_model_state):
        """设置教师模型（用于KL蒸馏）"""
        if self.teacher_model is None:
            from models import get_model
            self.teacher_model = get_model()
        self.teacher_model.load_state_dict(teacher_model_state)
        self.teacher_model.to(self.device)
        self.teacher_model.eval()

    @property
    def is_train_mini_batch(self):
        return self.train_inner_step > 0

    @property
    def is_test_mini_batch(self):
        return self.test_inner_step > 0

    def gen_support_batches(self):
        """生成support集的mini-batch"""
        for i in range(self.train_inner_step):
            try:
                data, target = next(self.train_dataset_loader_iterator)
            except StopIteration:
                self.train_dataset_loader_iterator = iter(self.support_loader)
                data, target = next(self.train_dataset_loader_iterator)
            yield data, target

    def gen_query_batches(self):
        """生成query集的mini-batch"""
        for i in range(self.test_inner_step):
            try:
                data, target = next(self.test_dataset_loader_iterator)
            except StopIteration:
                self.test_dataset_loader_iterator = iter(self.query_loader)
                data, target = next(self.test_dataset_loader_iterator)
            yield data, target

    def train(self, epochs, learning_rate, momentum=0, weight_decay=0,
              lr_decay=1.0, lr_decay_epoch=1, use_dp=False, dp_config=None,
              use_maml=False, beta=0.001, use_distillation=False,
              temperature=3.0, alpha=0.5, beta_feat=0.3, use_dp_distillation=False):
        """
        在本地数据上训练模型（支持标准训练和MAML元学习）

        Args:
            epochs: 本地训练轮数
            learning_rate: 初始学习率（标准模式或MAML内层学习率）
            momentum: SGD动量参数（0表示不使用动量）
            weight_decay: 权重衰减/L2正则化系数（0表示不使用）
            lr_decay: 学习率衰减系数
            lr_decay_epoch: 每多少轮衰减一次学习率
            use_dp: 是否使用差分隐私（用于梯度上传）
            dp_config: 差分隐私配置参数
            use_maml: 是否使用MAML元学习
            beta: MAML外层学习率
            use_distillation: 是否使用KL蒸馏
            temperature: KL蒸馏温度参数
            alpha: KL蒸馏损失权重
            beta_feat: 特征蒸馏损失权重（MSE）
            use_dp_distillation: 是否对蒸馏过程使用差分隐私

        Returns:
            avg_loss: 平均训练损失
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")

        # 根据是否使用MAML选择训练方法
        if use_maml and self.support_loader and self.query_loader:
            return self._train_maml(epochs, learning_rate, beta,
                                   use_distillation, temperature, alpha, beta_feat,
                                   use_dp_distillation, dp_config)
        else:
            return self._train_standard(epochs, learning_rate, momentum,
                                       weight_decay, lr_decay, lr_decay_epoch)

    def _train_standard(self, epochs, learning_rate, momentum,
                       weight_decay, lr_decay, lr_decay_epoch):
        """标准训练方法（保持原有逻辑不变）"""
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

    def _train_maml(self, num_iter, inner_lr, outer_lr,
                   use_distillation, temperature, alpha, beta_feat,
                   use_dp_distillation, dp_config):
        """MAML元学习训练方法（支持三组件互蒸馏：CE + KL + MSE + 蒸馏差分隐私）"""
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        mse_criterion = nn.MSELoss()

        # 初始化外层Adam优化器
        if self.outer_opt is None:
            self.outer_opt = Adam(lr=outer_lr)

        total_loss = 0.0

        for iteration in range(num_iter):
            # 确定数据加载器
            if self.is_train_mini_batch:
                support_data_loader = self.gen_support_batches()
            else:
                support_data_loader = self.support_loader

            if self.is_test_mini_batch:
                query_data_loader = self.gen_query_batches()
            else:
                query_data_loader = self.query_loader

            # 保存当前模型参数
            temp_model = copy.deepcopy(self.model.state_dict())

            # === 内层更新：在support集上快速适应 ===
            loss_sum = 0.0
            support_num_samples = 0

            for data, target in support_data_loader:
                data, target = data.to(self.device), target.to(self.device)
                num_sample = target.size(0)

                output = self.model(data)
                loss = criterion(output, target)

                support_num_samples += num_sample
                loss_sum += loss * num_sample

            # 计算support集梯度
            grads = torch.autograd.grad(
                loss_sum / support_num_samples,
                list(self.model.parameters()),
                create_graph=True,
                retain_graph=True
            )

            # 内层更新
            for p, g in zip(self.model.parameters(), grads):
                p.data.add_(g.data, alpha=-inner_lr)

            # 清空梯度
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            # === 外层更新：在query集上元学习 + 三组件互蒸馏 ===
            query_loss_sum = 0.0
            query_num_samples = 0

            for data, target in query_data_loader:
                data, target = data.to(self.device), target.to(self.device)
                num_sample = target.size(0)

                # 学生模型前向传播（获取logits和特征）
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

                    # 特征MSE损失（使用fc2层特征，通常是最后一个隐藏层）
                    feat_loss = mse_criterion(student_features['fc2'], teacher_features['fc2'])

                    # 三组件组合损失：loss = (1-α-β)*CE + α*KL + β*MSE
                    loss = (1 - alpha - beta_feat) * ce_loss + alpha * kl_loss + beta_feat * feat_loss
                else:
                    loss = ce_loss

                query_num_samples += num_sample
                query_loss_sum += loss * num_sample

            # 计算query集梯度
            grads = torch.autograd.grad(
                query_loss_sum / query_num_samples,
                list(self.model.parameters())
            )

            # 使用Adam外层更新
            self.outer_opt.increase_n()
            for i, (key, value) in enumerate(temp_model.items()):
                self.outer_opt(value, grads[i], i=i)

            # 加载更新后的参数
            self.model.load_state_dict(temp_model)

            # 清空梯度
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            total_loss += query_loss_sum.item()

        avg_loss = total_loss / (num_iter * query_num_samples) if query_num_samples > 0 else 0
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
