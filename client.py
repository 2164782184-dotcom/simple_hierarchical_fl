"""
客户端模块 (Client Module)
========================================
功能：
1. 客户端初始化：加载本地数据和模型
2. 本地训练：使用元学习（MAML）+ KL蒸馏的混合方式
   - 第一步：在 support 集上计算梯度并更新（MAML内层）
   - 第二步：在 query 集上计算梯度并使用 Adam 优化器更新（MAML外层）
   - 蒸馏：在 query 集上添加与教师模型的 KL 散度损失
3. 模型通信：与边缘服务器同步模型参数
4. 差分隐私：对梯度进行稀疏化和加噪处理
"""

import torch
import torch.nn.functional as F
from models import LeNet
from privacy import Flatten_gradients, reshape_gradients, local_process
import copy
import numpy as np
from collections import OrderedDict


class Adam:
    """
    全局 Adam 优化器
    用于基于从客户端收集的梯度来更新全局网络的参数
    """

    def __init__(self, lr=0.01, betas=(0.9, 0.999), eps=1e-08):
        """
        初始化 Adam 优化器

        Args:
            lr: 学习率
            betas: 一阶和二阶矩估计的指数衰减率 (beta1, beta2)
            eps: 防止除零的小常数
        """
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.m = dict()  # 一阶矩估计
        self.v = dict()  # 二阶矩估计
        self.n = 0  # 计数器
        self.created_momentum_grad_index = set()

    def __call__(self, params, grads, i):
        """
        执行一步 Adam 更新

        Args:
            params: 要更新的参数
            grads: 参数的梯度
            i: 参数的索引
        """
        # 如果是第一次遇到这个参数，初始化其动量
        if i not in self.m:
            self.m[i] = torch.zeros_like(params)
        if i not in self.v:
            self.v[i] = torch.zeros_like(params)

        # 更新一阶矩估计
        self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads

        # 更新二阶矩估计
        self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * torch.square(grads)

        # 计算偏差修正后的学习率
        alpha = self.lr * np.sqrt(1 - np.power(self.beta2, self.n))
        alpha = alpha / (1 - np.power(self.beta1, self.n))

        # 使用 Adam 公式更新参数
        params.sub_(alpha * self.m[i] / (torch.sqrt(self.v[i]) + self.eps))

    def increase_n(self):
        """增加优化步数计数器"""
        self.n += 1


class Client:
    """
    联邦学习客户端类 - MAML + KL蒸馏混合版本

    每个客户端拥有：
    1. 本地数据：support_loader、query_loader、test_loader
    2. 本地模型：使用 MAML 元学习方式训练
    3. 教师模型：用于知识蒸馏
    4. 与边缘服务器通信的缓冲区
    """

    def __init__(self, client_id, support_loader, query_loader, test_loader, device):
        """
        初始化客户端

        Args:
            client_id: 客户端唯一标识符
            support_loader: 支持集数据加载器（用于元学习的内层更新）
            query_loader: 查询集数据加载器（用于元学习的外层更新）
            test_loader: 测试集数据加载器
            device: 计算设备
        """
        self.id = client_id
        self.support_loader = support_loader
        self.query_loader = query_loader
        self.test_loader = test_loader
        self.device = device

        # 初始化模型
        self.model = LeNet().to(device)

        # 教师模型（用于蒸馏）
        self.teacher_model = None

        # 学习率设置
        self.inner_lr = 0.01   # 内层学习率（support 集更新）
        self.outer_lr = 0.001  # 外层学习率（query 集更新）

        # 损失函数
        self.criterion = torch.nn.CrossEntropyLoss()

        self.receiver_buffer = {}  # 用于接收边缘服务器下发的模型参数

        # 训练状态记录
        self.epoch = 0

        # 初始化外层 Adam 优化器
        self.outer_opt = Adam(lr=self.outer_lr)

    def set_teacher_model(self, teacher_state_dict):
        """
        设置教师模型（从边缘服务器或云服务器获取）

        Args:
            teacher_state_dict: 教师模型的状态字典
        """
        if self.teacher_model is None:
            self.teacher_model = LeNet().to(self.device)
        self.teacher_model.load_state_dict(teacher_state_dict)
        self.teacher_model.eval()  # 设置为评估模式

    def _compute_kl_loss(self, student_logits, teacher_logits, temperature=3.0):
        """
        计算 KL 散度损失（知识蒸馏）

        Args:
            student_logits: 学生模型的输出 logits
            teacher_logits: 教师模型的输出 logits
            temperature: 温度参数，用于软化概率分布

        Returns:
            KL 散度损失
        """
        # 使用温度缩放软化概率分布
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)

        # 计算 KL 散度
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')

        # 根据温度缩放损失（保持梯度尺度）
        return kl_loss * (temperature ** 2)

    def train(self, num_iter, use_distillation=False, temperature=3.0, alpha=0.5,
              use_dp=False, dp_epsilon=1.0, dp_delta=1e-5, dp_clip_c=1.0, dp_rate=0.1):
        """
        客户端本地更新（MAML + KL蒸馏）

        训练流程：
        1. 在 support 集上计算梯度并进行内层更新（快速适应）
        2. 在 query 集上计算梯度（交叉熵 + KL蒸馏）并使用 Adam 进行外层更新
        3. 返回模型差值（用于隐私保护和通信效率）

        Args:
            num_iter: 本地更新迭代次数
            use_distillation: 是否使用知识蒸馏
            temperature: KL 散度的温度参数
            alpha: 蒸馏损失的权重（最终损失 = (1-α)*CE + α*KL）
            use_dp: 是否使用差分隐私
            dp_epsilon: 隐私预算 epsilon
            dp_delta: 隐私预算 delta
            dp_clip_c: 梯度裁剪阈值
            dp_rate: top-k 稀疏化比例

        Returns:
            loss: 平均损失
            num_samples: 训练样本数
        """
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        # 进行 num_iter 次本地更新
        for iteration in range(num_iter):
            # === 第一步：保存当前模型参数 ===
            temp_model = copy.deepcopy(self.model.state_dict())

            # === 第二步：在 Support 集上进行内层更新 ===
            loss_sum = 0.0
            support_num_samples = 0

            for data in self.support_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                num_sample = labels.size(0)

                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                support_num_samples += num_sample
                loss_sum += loss * num_sample

            # 计算 support 集上的平均梯度
            grads = torch.autograd.grad(
                loss_sum / support_num_samples,
                list(self.model.parameters()),
                create_graph=True,   # 创建计算图用于二阶导数
                retain_graph=True    # 保留计算图
            )

            # 使用梯度进行内层更新：params = params - inner_lr * grads
            for p, g in zip(self.model.parameters(), grads):
                p.data.add_(g.data, alpha=-self.inner_lr)

            # 清空梯度
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            # === 第三步：在 Query 集上计算外层梯度 ===
            query_loss_sum = 0.0
            query_num_samples = 0

            for data in self.query_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                num_sample = labels.size(0)

                # 学生模型前向传播
                student_outputs = self.model(inputs)

                # 计算交叉熵损失
                ce_loss = self.criterion(student_outputs, labels)

                # 如果使用蒸馏且教师模型存在
                if use_distillation and self.teacher_model is not None:
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(inputs)

                    # 计算 KL 散度损失
                    kl_loss = self._compute_kl_loss(student_outputs, teacher_outputs, temperature)

                    # 组合损失：(1-α)*CE + α*KL
                    loss = (1 - alpha) * ce_loss + alpha * kl_loss
                else:
                    loss = ce_loss

                query_num_samples += num_sample
                query_loss_sum += loss * num_sample

            total_samples = query_num_samples

            # 计算 query 集上的梯度（用于外层更新）
            grads = torch.autograd.grad(
                query_loss_sum / query_num_samples,
                list(self.model.parameters())
            )

            # === 第四步：使用 Adam 进行外层更新 ===
            self.outer_opt.increase_n()

            # 对每个参数使用 Adam 更新
            for i, (key, value) in enumerate(temp_model.items()):
                self.outer_opt(value, grads[i], i=i)

            # 加载更新后的参数
            self.model.load_state_dict(temp_model)

            # 清空梯度
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            # 记录损失
            total_loss += query_loss_sum.item()

        # 计算平均损失
        avg_loss = total_loss / (num_iter * query_num_samples) if query_num_samples > 0 else 0

        return avg_loss, total_samples

    def test(self):
        """
        在测试集上评估模型

        Returns:
            accuracy: 测试准确率
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # 统计准确率
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0
        return accuracy

    def get_model_params(self):
        """获取模型参数"""
        return copy.deepcopy(self.model.state_dict())

    def set_model_params(self, model_params):
        """设置模型参数"""
        self.model.load_state_dict(model_params)

    def send_to_edgeserver(self, edgeserver, use_dp=False, dp_epsilon=1.0, dp_delta=1e-5,
                          dp_clip_c=1.0, dp_rate=0.1):
        """
        将本地模型发送给边缘服务器（支持差分隐私）

        Args:
            edgeserver: 边缘服务器对象
            use_dp: 是否使用差分隐私
            dp_epsilon: 隐私预算
            dp_delta: 隐私预算
            dp_clip_c: 梯度裁剪阈值
            dp_rate: top-k 稀疏化比例
        """
        if use_dp:
            # 计算模型差值
            delta = OrderedDict()
            for (name, p1), p0 in zip(self.model.named_parameters(), self.receiver_buffer.values()):
                delta[name] = p1.detach() - p0

            delta_list = list(delta.values())

            # 展平梯度
            flatten_grads, shape_of_grads = Flatten_gradients(delta_list)
            dimension = flatten_grads.numel()

            # 差分隐私处理（top-k 稀疏化 + 拉普拉斯加噪）
            class DPArgs:
                def __init__(self):
                    self.epsilon = dp_epsilon
                    self.delta = dp_delta
                    self.clip_C = dp_clip_c
                    self.rate = dp_rate

            dp_args = DPArgs()
            processed_update, choices = local_process(flatten_grads, dp_args, dimension)

            # 重塑梯度
            reshape_grads = reshape_gradients(processed_update, shape_of_grads)

            edgeserver.receive_from_client(
                client_id=self.id,
                cshared_state_dict=reshape_grads,
                use_dp=True
            )
        else:
            edgeserver.receive_from_client(
                client_id=self.id,
                cshared_state_dict=copy.deepcopy(self.model.state_dict()),
                use_dp=False
            )

    def receive_from_edgeserver(self, shared_state_dict):
        """
        从边缘服务器接收全局模型

        Args:
            shared_state_dict: 边缘服务器下发的模型参数
        """
        self.receiver_buffer = shared_state_dict

    def sync_with_edgeserver(self):
        """
        与边缘服务器同步模型
        将接收到的全局模型参数加载到本地模型
        """
        self.model.load_state_dict(self.receiver_buffer)
