import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from privacy import Flatten_gradients, local_process, reshape_gradients


class Client:
    """客户端类（支持差分隐私 + 知识蒸馏）"""

    def __init__(self, client_id, data_loader, device='cpu', teacher_model=None, student_model=None):
        """
        初始化客户端

        Args:
            client_id: 客户端ID
            data_loader: 该客户端的数据加载器
            device: 计算设备
            teacher_model: 教师模型实例（可选，用于不同架构的教师-学生）
            student_model: 学生模型实例（可选，用于不同架构的教师-学生）
        """
        self.client_id = client_id
        self.data_loader = data_loader
        self.device = device
        self.model = None  # 当前正在训练的模型
        self.initial_params = None  # 保存初始参数用于计算梯度
        self.teacher_model = teacher_model  # 本地教师模型
        self.student_model = student_model  # 本地学生模型
        self.distillation_teacher = None  # 用于蒸馏的目标教师模型
        self.feature_transform = None  # 特征转换层（用于对齐不同维度的特征）

    def set_model(self, global_model):
        """接收来自边缘服务器的全局模型"""
        self.model = copy.deepcopy(global_model)
        self.model.to(self.device)
        # 保存初始参数
        self.initial_params = copy.deepcopy(self.model.state_dict())

        # 如果使用不同架构的教师-学生模型，需要同步学生模型的参数
        # （全局模型使用的是学生模型架构）
        if self.student_model is not None:
            self.student_model.load_state_dict(self.model.state_dict())
            self.student_model.to(self.device)

    def set_teacher_model_state(self, teacher_model_state):
        """设置教师模型的参数（用于从全局模型同步）"""
        if self.teacher_model is not None:
            self.teacher_model.load_state_dict(teacher_model_state)
            self.teacher_model.to(self.device)

    def set_student_model_state(self, student_model_state):
        """设置学生模型的参数（用于从全局模型同步）"""
        if self.student_model is not None:
            self.student_model.load_state_dict(student_model_state)
            self.student_model.to(self.device)

    def set_distillation_teacher(self, teacher_model_state):
        """设置用于蒸馏的目标教师模型，并创建特征转换层（如果需要）"""
        # 根据传入的state_dict判断应该创建什么类型的模型
        # 通过检查第一个卷积层的形状来判断模型类型
        conv1_weight_shape = teacher_model_state['conv1.weight'].shape

        # 根据conv1的通道数判断模型类型
        in_channels = conv1_weight_shape[1]  # 输入通道
        out_channels = conv1_weight_shape[0]  # 输出通道

        # 检查fc1的形状来区分LeNet和MnistStudent/MnistTeacher
        fc1_out_features = teacher_model_state['fc1.weight'].shape[0]

        # 确定需要创建的模型类型
        model_name = None
        if in_channels == 1:  # MNIST
            if out_channels == 10:
                model_name = 'mnist_teacher'
            elif out_channels == 6:
                # 区分LeNet和MnistStudent
                if fc1_out_features == 120:  # LeNet: fc1输出120
                    model_name = 'lenet'
                elif fc1_out_features == 50:  # MnistStudent: fc1输出50
                    model_name = 'mnist_student'
                else:
                    model_name = 'lenet'  # 默认
            else:
                model_name = 'lenet'
        elif in_channels == 3:  # CIFAR-10
            if out_channels == 32:
                model_name = 'cifar_teacher'
            elif out_channels == 16:
                model_name = 'cifar_student'
            else:
                model_name = 'lenet'
        else:
            # 默认情况
            model_name = 'lenet'

        # 检查是否需要重新创建模型（类型改变了）
        need_recreate = False
        if self.distillation_teacher is None:
            need_recreate = True
        else:
            # 检查现有模型类型是否匹配
            current_model_type = type(self.distillation_teacher).__name__
            if model_name == 'mnist_teacher' and current_model_type != 'MnistTeacher':
                need_recreate = True
            elif model_name == 'mnist_student' and current_model_type != 'MnistStudent':
                need_recreate = True
            elif model_name == 'cifar_teacher' and current_model_type != 'CifarTeacher':
                need_recreate = True
            elif model_name == 'cifar_student' and current_model_type != 'CifarStudent':
                need_recreate = True
            elif model_name == 'lenet' and current_model_type != 'LeNet':
                need_recreate = True

        # 如果需要，重新创建模型
        if need_recreate:
            from models import get_model
            self.distillation_teacher = get_model(model_name)
            self.distillation_teacher.to(self.device)

        self.distillation_teacher.load_state_dict(teacher_model_state)
        self.distillation_teacher.eval()

        # 创建特征转换层φ：用于对齐学生和教师的特征维度
        # 获取一个样本来推断特征维度
        sample_input = next(iter(self.data_loader))[0][:1].to(self.device)

        with torch.no_grad():
            _, student_features = self.model(sample_input, return_features=True)
            _, teacher_features = self.distillation_teacher(sample_input, return_features=True)

        # 确定使用哪个特征键
        feature_key_student = 'distill_features' if 'distill_features' in student_features else 'fc2'
        feature_key_teacher = 'distill_features' if 'distill_features' in teacher_features else 'fc2'

        student_feat_dim = student_features[feature_key_student].shape[1]
        teacher_feat_dim = teacher_features[feature_key_teacher].shape[1]

        # 如果特征维度不同，创建线性变换层
        if student_feat_dim != teacher_feat_dim:
            self.feature_transform = nn.Linear(student_feat_dim, teacher_feat_dim).to(self.device)
            print(f"  [客户端{self.client_id}] 创建特征转换层: {student_feat_dim} -> {teacher_feat_dim}")
        else:
            self.feature_transform = None

    def switch_to_teacher(self):
        """切换到教师模型进行训练"""
        if self.teacher_model is not None:
            self.model = self.teacher_model
            self.initial_params = copy.deepcopy(self.model.state_dict())

    def switch_to_student(self):
        """切换到学生模型进行训练"""
        if self.student_model is not None:
            self.model = self.student_model
            self.initial_params = copy.deepcopy(self.model.state_dict())

    def get_teacher_state_dict(self):
        """获取教师模型的state_dict"""
        if self.teacher_model is not None:
            return copy.deepcopy(self.teacher_model.state_dict())
        return None

    def get_student_state_dict(self):
        """获取学生模型的state_dict"""
        if self.student_model is not None:
            return copy.deepcopy(self.student_model.state_dict())
        return None

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
                if use_distillation and self.distillation_teacher is not None:
                    student_output, student_features = self.model(data, return_features=True)
                else:
                    student_output = self.model(data, return_features=False)

                # 1. 交叉熵损失（硬标签）
                ce_loss = criterion(student_output, target)

                # 2. KL散度 + 3. 特征MSE（如果启用蒸馏）
                if use_distillation and self.distillation_teacher is not None:
                    with torch.no_grad():
                        teacher_output, teacher_features = self.distillation_teacher(data, return_features=True)

                        # 对教师模型的输出添加差分隐私噪声
                        if use_dp_distillation and dp_config is not None:
                            teacher_output = self._add_noise_to_teacher_output(teacher_output, dp_config)
                            # 根据模型类型选择特征层
                            feature_key = 'distill_features' if 'distill_features' in teacher_features else 'fc2'
                            teacher_features[feature_key] = self._add_noise_to_teacher_features(
                                teacher_features[feature_key], dp_config
                            )

                    # KL散度损失（软标签）
                    kl_loss = self._compute_kl_loss(student_output, teacher_output, temperature)

                    # 特征MSE损失（使用蒸馏特征层）
                    # 根据模型类型选择特征层
                    feature_key = 'distill_features' if 'distill_features' in student_features else 'fc2'

                    # 获取学生和教师的特征
                    student_feat = student_features[feature_key]
                    teacher_feat = teacher_features[feature_key]

                    # 如果存在特征转换层，应用φ变换对齐维度
                    if self.feature_transform is not None:
                        student_feat = self.feature_transform(student_feat)

                    # 使用标准化而非归一化，保留幅度信息但统一尺度
                    teacher_feat_mean = teacher_feat.mean()
                    teacher_feat_std = teacher_feat.std() + 1e-8

                    student_feat_scaled = (student_feat - student_feat.mean()) / (student_feat.std() + 1e-8)
                    teacher_feat_scaled = (teacher_feat - teacher_feat_mean) / teacher_feat_std

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
