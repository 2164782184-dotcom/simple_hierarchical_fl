"""
动态权重调整模块
根据CE、KL、MSE三个损失的相对大小，自适应调整权重
原则：损失值越大，说明该部分需要更多关注，增加其权重
"""

import torch


class DynamicWeightAdjuster:
    """动态蒸馏权重调整器"""

    def __init__(self, initial_alpha=0.4, initial_beta=0.2, adjust_rate=0.05,
                 min_weight=0.1, max_weight=0.6):
        """
        初始化动态权重调整器

        Args:
            initial_alpha: KL损失的初始权重
            initial_beta: MSE损失的初始权重
            adjust_rate: 每次调整的幅度
            min_weight: 单个权重的最小值
            max_weight: 单个权重的最大值
        """
        self.alpha = initial_alpha
        self.beta = initial_beta
        self.adjust_rate = adjust_rate
        self.min_weight = min_weight
        self.max_weight = max_weight

        # 历史记录（用于平滑调整）
        self.loss_history = {
            'ce': [],
            'kl': [],
            'mse': []
        }
        self.history_size = 5  # 保留最近5次的损失值

    def update(self, ce_loss, kl_loss, mse_loss):
        """
        根据三个损失值更新权重

        策略：
        1. 计算三个损失的归一化比例
        2. 损失越大，权重越大
        3. 保证 alpha + beta < 1（留给CE的空间）

        Args:
            ce_loss: 交叉熵损失值
            kl_loss: KL散度损失值
            mse_loss: MSE损失值

        Returns:
            (alpha, beta): 更新后的权重
        """
        # 记录历史
        self.loss_history['ce'].append(ce_loss)
        self.loss_history['kl'].append(kl_loss)
        self.loss_history['mse'].append(mse_loss)

        # 只保留最近几次
        for key in self.loss_history:
            if len(self.loss_history[key]) > self.history_size:
                self.loss_history[key].pop(0)

        # 计算平滑后的损失（避免单次波动）
        avg_ce = sum(self.loss_history['ce']) / len(self.loss_history['ce'])
        avg_kl = sum(self.loss_history['kl']) / len(self.loss_history['kl'])
        avg_mse = sum(self.loss_history['mse']) / len(self.loss_history['mse'])

        # 归一化：计算每个损失占总损失的比例
        total_loss = avg_ce + avg_kl + avg_mse
        if total_loss < 1e-8:  # 避免除零
            return self.alpha, self.beta

        ce_ratio = avg_ce / total_loss
        kl_ratio = avg_kl / total_loss
        mse_ratio = avg_mse / total_loss

        # 保留给CE至少20%的权重，至多50%
        ce_weight = max(0.2, min(0.5, ce_ratio))
        available_weight = 1 - ce_weight

        # 根据KL和MSE的比例分配权重
        kl_mse_total = kl_ratio + mse_ratio
        if kl_mse_total < 1e-8:
            new_alpha = available_weight / 2
            new_beta = available_weight / 2
        else:
            new_alpha = available_weight * (kl_ratio / kl_mse_total)
            new_beta = available_weight * (mse_ratio / kl_mse_total)

        # 平滑更新（不要突变）：使用移动平均
        momentum = 0.7  # 保留70%的旧值
        self.alpha = momentum * self.alpha + (1 - momentum) * new_alpha
        self.beta = momentum * self.beta + (1 - momentum) * new_beta

        # 裁剪到合法范围
        self.alpha = max(self.min_weight, min(self.max_weight, self.alpha))
        self.beta = max(self.min_weight, min(self.max_weight, self.beta))

        # 确保 alpha + beta < 1
        if self.alpha + self.beta > 0.8:  # 最多占80%
            total = self.alpha + self.beta
            self.alpha = 0.8 * (self.alpha / total)
            self.beta = 0.8 * (self.beta / total)

        return self.alpha, self.beta

    def get_weights(self):
        """获取当前权重"""
        return self.alpha, self.beta

    def get_ce_weight(self):
        """获取CE的权重"""
        return 1 - self.alpha - self.beta

    def reset(self):
        """重置历史记录"""
        self.loss_history = {'ce': [], 'kl': [], 'mse': []}


# 简化版API：单次调整（无历史记录）
def adjust_weights_once(ce_loss, kl_loss, mse_loss, current_alpha, current_beta,
                       adjust_rate=0.05, min_weight=0.1, max_weight=0.6):
    """
    简化版：根据单次损失值调整权重（无状态）

    Args:
        ce_loss: 当前CE损失
        kl_loss: 当前KL损失
        mse_loss: 当前MSE损失
        current_alpha: 当前KL权重
        current_beta: 当前MSE权重
        adjust_rate: 调整幅度
        min_weight: 最小权重
        max_weight: 最大权重

    Returns:
        (new_alpha, new_beta): 调整后的权重
    """
    # 归一化
    total_loss = ce_loss + kl_loss + mse_loss
    if total_loss < 1e-8:
        return current_alpha, current_beta

    kl_ratio = kl_loss / total_loss
    mse_ratio = mse_loss / total_loss

    # 根据比例调整
    kl_mse_total = kl_ratio + mse_ratio
    if kl_mse_total < 1e-8:
        target_alpha = 0.3
        target_beta = 0.3
    else:
        available = 0.6
        target_alpha = available * (kl_ratio / kl_mse_total)
        target_beta = available * (mse_ratio / kl_mse_total)

    # 平滑调整
    new_alpha = 0.8 * current_alpha + 0.2 * target_alpha
    new_beta = 0.8 * current_beta + 0.2 * target_beta

    # 裁剪
    new_alpha = max(min_weight, min(max_weight, new_alpha))
    new_beta = max(min_weight, min(max_weight, new_beta))

    # 确保和不超过0.8
    if new_alpha + new_beta > 0.8:
        total = new_alpha + new_beta
        new_alpha = 0.8 * (new_alpha / total)
        new_beta = 0.8 * (new_beta / total)

    return new_alpha, new_beta


"""
使用示例：

# 方法1：使用状态化的调整器（推荐）
adjuster = DynamicWeightAdjuster(initial_alpha=0.4, initial_beta=0.2)

for batch in dataloader:
    # 前向传播
    ce_loss = ...
    kl_loss = ...
    mse_loss = ...

    # 更新权重
    alpha, beta = adjuster.update(ce_loss.item(), kl_loss.item(), mse_loss.item())

    # 使用新权重计算总损失
    total_loss = (1 - alpha - beta) * ce_loss + alpha * kl_loss + beta * mse_loss


# 方法2：使用无状态函数（简单场景）
alpha, beta = adjust_weights_once(
    ce_loss.item(), kl_loss.item(), mse_loss.item(),
    current_alpha=0.4, current_beta=0.2
)
"""
