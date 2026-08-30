"""
差分隐私模块 (Differential Privacy Module)
========================================
功能：
1. 梯度展平和重塑
2. 梯度裁剪（L2 范数裁剪）
3. 梯度稀疏化（Top-k 选择）
4. 噪声添加（拉普拉斯/高斯噪声）
5. 坐标变换（归一化和反归一化）

差分隐私保证：
通过梯度裁剪 + 噪声添加，保证模型参数的隐私性
epsilon 越小，隐私保护越强，但模型效用下降

"""

import torch
import numpy as np
import math


def Flatten_gradients(grads):
    """
    将梯度列表展平为一维张量

    Args:
        grads: 梯度列表，每个元素是一个张量（对应模型的一层）
               例如：[conv1_grad (shape: [64, 3, 3, 3]),
                      fc_grad (shape: [10, 64])]

    Returns:
        flatten_grads: 展平后的一维张量
                      例如：tensor([0.1, 0.2, ..., 0.9])  (长度 = 所有参数数量)
        shape_of_grads: 每个梯度的原始形状列表
                       例如：[[64, 3, 3, 3], [10, 64]]
    """
    # 将所有梯度展平并拼接成一维向量
    flatten_grads = torch.cat([g.flatten() for g in grads])

    # 记录每个梯度的原始形状（用于后续恢复）
    shape_of_grads = [g.shape for g in grads]

    return flatten_grads, shape_of_grads


def Private_gradients2(gradient, args):
    """
    对梯度进行差分隐私处理（高斯机制）

    流程：
    1. 计算梯度的 L2 范数
    2. 如果范数超过阈值 clip_C，则进行裁剪
    3. 添加高斯噪声

    Args:
        gradient: 梯度张量
        args: 配置参数（包含 clip_C, epsilon, delta）

    Returns:
        noisy_grads: 加噪后的梯度张量
    """
    device = gradient.device

    # 计算梯度的 L2 范数（欧几里得范数）
    norm_2 = torch.norm(gradient, p=2)

    # 梯度裁剪：如果 ||gradient|| > clip_C，则缩放到 clip_C
    # grads = gradient * min(1, clip_C / ||gradient||)
    grads = gradient / torch.max(norm_2 / args.clip_C, torch.tensor(1.0, device=device))

    # 计算概率 pr（用于随机化方向）
    pr = 0.5 + (torch.norm(grads) / (2 * args.clip_C))
    pr = torch.min(pr, torch.tensor(1.0))

    # 随机决定梯度方向（伯努利分布）
    x = 2 * torch.from_numpy(np.random.binomial(1, pr.item(), 1)).float().to(device) - 1
    grads = x * grads * args.clip_C / torch.norm(grads)

    # 计算噪声向量的尺度 M
    d = grads.numel()  # 梯度维度
    C = (torch.exp(torch.tensor(args.epsilon, device=device)) + 1) / \
        (torch.exp(torch.tensor(args.epsilon, device=device)) - 1)
    M = C * args.clip_C * torch.sqrt(torch.tensor(np.pi * d / 2, device=device))

    # 生成并规范化噪声向量 V
    V = torch.randn(d, device=device)
    V = V / torch.norm(V)

    # 确保 V 与 grads 在同一方向
    if torch.dot(V, grads) < 0:
        V = -V

    # 再次计算概率并调整 V 的方向
    pr = torch.exp(torch.tensor(args.epsilon, device=device)) / \
         (torch.exp(torch.tensor(args.epsilon)) + 1)
    x = 2 * torch.from_numpy(np.random.binomial(1, pr.item(), 1)).float().to(device) - 1
    V = V * x * M

    return V


def Private_gradients(gradient, args):
    """
    对梯度进行差分隐私处理（坐标采样机制）

    流程：
    1. 随机选择一个坐标
    2. 对该坐标的值进行裁剪
    3. 添加拉普拉斯噪声
    4. 放大 D_len 倍（无偏估计）

    Args:
        gradient: 梯度张量
        args: 配置参数

    Returns:
        H: 处理后的梯度张量（只有一个坐标非零）
    """
    D_len = gradient.numel()  # 梯度总维度
    Z = np.random.choice(D_len, 1).item()  # 随机选择一个坐标

    # 计算该坐标的范数
    norm = torch.abs(gradient[Z])

    # 梯度裁剪
    grads = gradient[Z] / max(norm / args.norm_clip, 1)

    # 计算常数 C
    C = (torch.exp(torch.tensor(args.epsilon)) + 1) / \
        (torch.exp(torch.tensor(args.epsilon)) - 1)

    # 计算概率 pr
    pr = 0.5 + (grads / (2 * C * args.norm_clip))

    # 随机化方向
    x = 2 * np.random.binomial(1, pr.item(), 1).item() - 1

    # 创建稀疏梯度向量
    H = torch.zeros_like(gradient)
    H[Z] = D_len * x * C * args.norm_clip

    return H


def reshape_gradients(flat_gradient, shape_of_gradients):
    """
    将展平的一维梯度重塑回原始形状

    Args:
        flat_gradient: 展平的一维梯度张量
        shape_of_gradients: 原始梯度形状列表

    Returns:
        grads: 重塑后的梯度列表
    """
    grads = []
    index = 0

    for shape in shape_of_gradients:
        # 计算当前梯度的元素数量
        num_elements = torch.prod(torch.tensor(shape)).item()

        # 从展平梯度中切片并重塑为原始形状
        grad = flat_gradient[index:index + num_elements].view(shape)
        grads.append(grad)

        # 更新索引
        index += num_elements

    return grads


def topindex(updates, topk):
    """
    找出梯度中绝对值最大的 top-k 个元素的索引（稀疏化）

    Args:
        updates: 梯度张量
        topk: 要保留的元素数量

    Returns:
        indices: top-k 索引列表
    """
    # 计算元素的绝对值
    abs_updates = torch.abs(updates)

    # 找到绝对值最大的 k 个元素的索引
    _, indices = torch.topk(abs_updates, topk)

    return indices.tolist()


def local_process(flattened, args, dimension, device):
    """
    客户端本地处理：稀疏化 + 加噪

    流程：
    1. 计算 top-k 索引（稀疏化）
    2. 对选中的元素进行裁剪和加噪（差分隐私）
    3. 未选中的元素置零

    Args:
        flattened: 展平的梯度张量
        args: 配置参数
        dimension: 梯度总维度

    Returns:
        vector: 处理后的梯度向量
        choices: 被选中的 top-k 索引
    """
    # 计算要保留的元素数量
    topk = int(dimension / args.rate)  # rate 是压缩率，例如 rate=50 表示保留 2%

    # 找出 top-k 索引
    choices = topindex(flattened, topk)

    # 对选中的元素进行裁剪和加噪
    vector = sampling_randomizer(
        flattened, choices, args.clip_C,
        args.epsilon, args.delta, args.mechanism, device
    )

    return vector, choices


def sampling_randomizer(vector, choices, clip_C, eps, delta, mechanism, device, left=0, right=1):
    """
    对选中的梯度元素进行采样随机化（差分隐私）

    流程：
    1. 将梯度裁剪到 [-clip_C, clip_C]
    2. 将选中的元素归一化到 [left, right]
    3. 添加拉普拉斯噪声
    4. 反归一化回原始范围
    5. 未选中的元素置零

    Args:
        vector: 梯度向量
        choices: 被选中的索引列表
        clip_C: 裁剪阈值
        eps: 隐私预算 epsilon
        delta: 隐私参数 delta
        mechanism: 噪声机制（'gaussian' 或 'laplace'）
        device: 计算设备（'cpu' 或 torch.device 对象）
        left: 归一化后的左边界
        right: 归一化后的右边界

    Returns:
        result: 处理后的梯度向量
    """
    # 确保 device 是 torch.device 对象
    if isinstance(device, str):
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        device = torch.device('cpu')

    # 步骤 1：梯度裁剪
    vector = torch.clamp(vector, -clip_C, clip_C)

    # 初始化结果向量（全零）
    result = torch.zeros_like(vector)

    # 步骤 2：提取被选中的元素
    chosen_elements = vector[choices]

    # 步骤 3：归一化到 [left, right]
    normalized_elements = transform(chosen_elements, -clip_C, clip_C, left, right)

    # 步骤 4：添加拉普拉斯噪声
    noise = one_laplace(eps, right - left, normalized_elements)

    # 步骤 5：加噪后反归一化回原始范围
    noisy_elements = normalized_elements + noise.to(device)
    result[choices] = transform(noisy_elements, left, right, -clip_C, clip_C)

    return result


def transform(v, left, right, new_left, new_right):
    """
    线性变换：将值从 [left, right] 映射到 [new_left, new_right]

    公式：
        new_v = new_left + (new_right - new_left) * (v - left) / (right - left)

    Args:
        v: 原始值或张量
        left: 原始范围的左边界
        right: 原始范围的右边界
        new_left: 目标范围的左边界
        new_right: 目标范围的右边界

    Returns:
        变换后的值或张量
    """
    return new_left + (new_right - new_left) * (v - left) / (right - left)


def one_gaussian(eps, delta, sensitivity):
    """
    生成一个高斯噪声样本

    高斯机制的标准差公式：
        sigma = (sensitivity / eps) * sqrt(2 * log(1.25 / delta))

    Args:
        eps: 隐私预算 epsilon
        delta: 隐私参数 delta
        sensitivity: 敏感度（梯度的最大变化量）

    Returns:
        噪声样本（标量）
    """
    sigma = (sensitivity / eps) * math.sqrt(2 * math.log(1.25 / delta))
    return np.random.normal(0, sigma)


def one_laplace(eps, sensitivity, chosen_elements):
    """
    生成拉普拉斯噪声样本

    拉普拉斯机制的尺度参数：
        b = sensitivity / eps

    Args:
        eps: 隐私预算 epsilon
        sensitivity: 敏感度
        chosen_elements: 需要加噪的元素（用于确定噪声的形状）

    Returns:
        噪声张量（与 chosen_elements 形状相同）
    """
    return torch.distributions.Laplace(0, sensitivity / eps).sample(sample_shape=chosen_elements.size())


"""
差分隐私关键概念：
====================

1. 隐私预算 (epsilon)：
   - 越小隐私保护越强，但模型效用下降
   - 典型值：0.1 到 10
   - epsilon=0 表示完全隐私（但无法学习）
   - epsilon=∞ 表示无隐私保护

2. 失败概率 (delta)：
   - 差分隐私失败的概率上界
   - 通常设置为 1/n^2，其中 n 是数据集大小
   - 典型值：1e-5 到 1e-7

3. 敏感度 (sensitivity)：
   - 单个样本对结果的最大影响
   - 通过梯度裁剪（clip_C）来控制
   - 越小隐私保护成本越低

4. 稀疏化 (top-k)：
   - 只传输最重要的梯度，减少通信量
   - 压缩率 = 1 / rate
   - rate=50 表示只保留 2% 的梯度

5. 噪声机制：
   - 拉普拉斯噪声：适合 L1 敏感度
   - 高斯噪声：适合 L2 敏感度
   - 本项目主要使用拉普拉斯噪声
"""
