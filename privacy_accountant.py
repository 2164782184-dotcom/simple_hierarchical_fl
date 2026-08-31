"""
差分隐私预算统计模块 (Privacy Accountant)
===========================================
功能：
1. 跟踪每轮训练的隐私预算消耗
2. 计算累积隐私预算（总epsilon和delta）
3. 使用高级组合定理（Advanced Composition）
4. 支持多层级隐私统计（客户端→边缘→云端）

理论基础：
- 基础组合：ε_total = Σ ε_i (太保守)
- 高级组合：ε_total ≈ √(2k·ln(1/δ')) · ε + k·ε·(e^ε - 1)
  其中 k 为训练轮数，δ' 为组合失败概率
"""

import math
import numpy as np


class PrivacyAccountant:
    """差分隐私预算统计器"""

    def __init__(self, target_epsilon=None, target_delta=None):
        """
        初始化隐私统计器

        Args:
            target_epsilon: 目标总隐私预算（可选，用于预警）
            target_delta: 目标总失败概率（可选）
        """
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

        # 记录每轮的隐私消耗
        self.rounds = []  # 每个元素: {'round': int, 'epsilon': float, 'delta': float, 'clients': int}

    def add_round(self, round_num, epsilon, delta, num_clients):
        """
        记录一轮训练的隐私消耗

        Args:
            round_num: 训练轮次
            epsilon: 本轮的epsilon（单个客户端）
            delta: 本轮的delta
            num_clients: 参与本轮训练的客户端数量
        """
        self.rounds.append({
            'round': round_num,
            'epsilon': epsilon,
            'delta': delta,
            'clients': num_clients
        })

    def get_total_epsilon_basic(self):
        """
        基础组合：ε_total = Σ ε_i
        最保守的估计，通常过于悲观

        Returns:
            total_epsilon: 累积epsilon
        """
        return sum(r['epsilon'] for r in self.rounds)

    def get_total_delta_basic(self):
        """
        基础组合：δ_total = Σ δ_i

        Returns:
            total_delta: 累积delta
        """
        return sum(r['delta'] for r in self.rounds)

    def get_total_epsilon_advanced(self, delta_prime=1e-6):
        """
        高级组合定理（Advanced Composition Theorem）
        更紧的界限，适用于多轮训练

        公式：
        ε_total ≤ √(2k·ln(1/δ')) · ε + k·ε·(e^ε - 1)

        其中：
        - k: 训练轮数
        - ε: 单轮epsilon
        - δ': 组合失败概率（通常设为 1e-6）

        Args:
            delta_prime: 高级组合的失败概率参数

        Returns:
            total_epsilon: 累积epsilon（高级组合）
        """
        if not self.rounds:
            return 0.0

        k = len(self.rounds)
        eps = self.rounds[0]['epsilon']  # 假设每轮epsilon相同

        # 检查是否所有轮次的epsilon相同
        if not all(r['epsilon'] == eps for r in self.rounds):
            # 如果不同，使用平均值
            eps = np.mean([r['epsilon'] for r in self.rounds])

        # 高级组合公式
        term1 = math.sqrt(2 * k * math.log(1 / delta_prime)) * eps
        term2 = k * eps * (math.exp(eps) - 1)

        return term1 + term2

    def get_total_delta_advanced(self):
        """
        高级组合的总delta
        δ_total = Σ δ_i + δ'

        Returns:
            total_delta: 累积delta
        """
        basic_delta = self.get_total_delta_basic()
        # 这里使用默认的delta_prime=1e-6
        return basic_delta + 1e-6

    def get_rdp_epsilon(self, alpha=10):
        """
        Rényi差分隐私（RDP）
        更精确的隐私分析，Google的DP-SGD使用此方法

        RDP(α) = (1/(α-1)) · log(E[(P/Q)^α])

        对于高斯机制：RDP(α) = (α·Δ²)/(2σ²)
        其中 σ = Δ·√(2·ln(1.25/δ))/ε

        Args:
            alpha: Rényi参数（通常取10或20）

        Returns:
            rdp_epsilon: RDP epsilon
        """
        if not self.rounds:
            return 0.0

        # 简化实现：累加每轮的RDP
        total_rdp = 0.0
        for r in self.rounds:
            eps = r['epsilon']
            delta = r['delta']

            # 计算噪声标准差 σ
            sensitivity = 1.0  # 假设敏感度为1（通过梯度裁剪保证）
            sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / eps

            # RDP公式
            rdp = (alpha * sensitivity ** 2) / (2 * sigma ** 2)
            total_rdp += rdp

        return total_rdp

    def check_privacy_budget(self):
        """
        检查是否超出目标隐私预算

        Returns:
            is_safe: 是否在安全范围内
            warning_msg: 警告信息
        """
        if self.target_epsilon is None:
            return True, "未设置目标隐私预算"

        current_epsilon = self.get_total_epsilon_advanced()

        if current_epsilon > self.target_epsilon:
            return False, f"⚠️  隐私预算超支！当前: {current_epsilon:.4f}, 目标: {self.target_epsilon:.4f}"
        elif current_epsilon > 0.8 * self.target_epsilon:
            return True, f"⚠️  隐私预算即将耗尽！当前: {current_epsilon:.4f}, 目标: {self.target_epsilon:.4f}"
        else:
            remaining = self.target_epsilon - current_epsilon
            return True, f"✓ 隐私预算充足，剩余: {remaining:.4f}"

    def get_privacy_report(self):
        """
        生成完整的隐私统计报告

        Returns:
            report: 隐私报告字符串
        """
        if not self.rounds:
            return "尚未开始训练，无隐私消耗记录"

        basic_eps = self.get_total_epsilon_basic()
        basic_delta = self.get_total_delta_basic()
        advanced_eps = self.get_total_epsilon_advanced()
        advanced_delta = self.get_total_delta_advanced()
        rdp_eps = self.get_rdp_epsilon()

        is_safe, warning = self.check_privacy_budget()

        report = "=" * 70 + "\n"
        report += "差分隐私预算统计报告\n"
        report += "=" * 70 + "\n"
        report += f"训练轮数: {len(self.rounds)}\n"
        report += f"总参与客户端次数: {sum(r['clients'] for r in self.rounds)}\n"
        report += f"单轮隐私参数: ε={self.rounds[0]['epsilon']}, δ={self.rounds[0]['delta']}\n"
        report += "\n"

        report += "【基础组合】（保守估计）\n"
        report += f"  累积 ε: {basic_eps:.6f}\n"
        report += f"  累积 δ: {basic_delta:.6e}\n"
        report += "\n"

        report += "【高级组合】（推荐使用）\n"
        report += f"  累积 ε: {advanced_eps:.6f}\n"
        report += f"  累积 δ: {advanced_delta:.6e}\n"
        report += "\n"

        report += "【Rényi DP】（精确分析）\n"
        report += f"  RDP ε(α=10): {rdp_eps:.6f}\n"
        report += "\n"

        report += "【隐私预算状态】\n"
        if self.target_epsilon:
            report += f"  目标 ε: {self.target_epsilon}\n"
            report += f"  当前 ε: {advanced_eps:.6f}\n"
            report += f"  使用率: {advanced_eps/self.target_epsilon*100:.2f}%\n"
            report += f"  状态: {warning}\n"
        else:
            report += f"  未设置目标隐私预算\n"

        report += "=" * 70

        return report

    def save_history(self, filepath):
        """
        保存隐私消耗历史到文件

        Args:
            filepath: 保存路径
        """
        import json

        history = {
            'target_epsilon': self.target_epsilon,
            'target_delta': self.target_delta,
            'rounds': self.rounds,
            'summary': {
                'basic_epsilon': self.get_total_epsilon_basic(),
                'basic_delta': self.get_total_delta_basic(),
                'advanced_epsilon': self.get_total_epsilon_advanced(),
                'advanced_delta': self.get_total_delta_advanced(),
                'rdp_epsilon': self.get_rdp_epsilon()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)


class HierarchicalPrivacyAccountant:
    """分层联邦学习的隐私统计器（客户端→边缘→云端）"""

    def __init__(self, target_epsilon=None):
        self.target_epsilon = target_epsilon

        # 三层隐私统计
        self.client_accountant = PrivacyAccountant(target_epsilon)
        self.edge_accountant = PrivacyAccountant()
        self.cloud_accountant = PrivacyAccountant()

    def add_client_round(self, round_num, epsilon, delta, num_clients):
        """记录客户端层的隐私消耗"""
        self.client_accountant.add_round(round_num, epsilon, delta, num_clients)

    def add_edge_round(self, round_num, epsilon, delta, num_edges):
        """记录边缘层的隐私消耗（如果边缘聚合也有DP）"""
        self.edge_accountant.add_round(round_num, epsilon, delta, num_edges)

    def add_cloud_round(self, round_num, epsilon, delta):
        """记录云端层的隐私消耗"""
        self.cloud_accountant.add_round(round_num, epsilon, delta, 1)

    def get_full_report(self):
        """生成完整的三层隐私报告"""
        report = "\n" + "=" * 70 + "\n"
        report += "分层联邦学习隐私统计报告\n"
        report += "=" * 70 + "\n\n"

        report += "【客户端层】\n"
        report += self.client_accountant.get_privacy_report()
        report += "\n\n"

        if self.edge_accountant.rounds:
            report += "【边缘层】\n"
            report += self.edge_accountant.get_privacy_report()
            report += "\n\n"

        if self.cloud_accountant.rounds:
            report += "【云端层】\n"
            report += self.cloud_accountant.get_privacy_report()
            report += "\n\n"

        return report


"""
使用示例：
=========

# 在 main.py 中初始化
privacy_accountant = PrivacyAccountant(target_epsilon=10.0, target_delta=1e-5)

# 每轮训练后记录
for round in range(NUM_ROUNDS):
    # ... 训练代码 ...

    privacy_accountant.add_round(
        round_num=round,
        epsilon=DP_EPSILON,
        delta=DP_DELTA,
        num_clients=len(selected_clients)
    )

    # 每10轮打印报告
    if round % 10 == 0:
        print(privacy_accountant.get_privacy_report())

    # 检查是否超预算
    is_safe, msg = privacy_accountant.check_privacy_budget()
    if not is_safe:
        print(msg)
        break

# 训练结束后保存
privacy_accountant.save_history('privacy_history.json')
"""
