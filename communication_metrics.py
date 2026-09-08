"""
通讯成本统计模块
用于衡量分层联邦学习中的通讯开销
"""

import torch
import numpy as np


class CommunicationMetrics:
    """通讯成本统计类"""

    def __init__(self):
        """初始化统计指标"""
        self.reset()

    def reset(self):
        """重置所有统计"""
        # 总体统计
        self.total_bits_uploaded = 0      # 上行总比特数
        self.total_bits_downloaded = 0    # 下行总比特数
        self.total_rounds = 0              # 总轮数

        # 分层统计
        self.client_to_edge_bits = 0      # 客户端→边缘
        self.edge_to_cloud_bits = 0       # 边缘→云
        self.cloud_to_edge_bits = 0       # 云→边缘
        self.edge_to_client_bits = 0      # 边缘→客户端

        # 每轮统计
        self.round_upload_bits = []       # 每轮上行
        self.round_download_bits = []     # 每轮下行

    def calculate_model_size(self, model_params):
        """
        计算模型参数的大小

        Args:
            model_params: 模型参数字典 (state_dict) 或 稀疏梯度元组

        Returns:
            size_bits: 模型大小（比特）
            size_mb: 模型大小（MB）
            num_params: 参数数量
        """
        if isinstance(model_params, dict):
            # 完整模型参数
            return self._calculate_full_model_size(model_params)
        elif isinstance(model_params, tuple):
            # 稀疏梯度 (gradient_vector, choices, shapes)
            return self._calculate_sparse_gradient_size(model_params)
        else:
            raise ValueError(f"Unknown model_params type: {type(model_params)}")

    def _calculate_full_model_size(self, state_dict):
        """
        计算完整模型参数大小

        Args:
            state_dict: 模型参数字典

        Returns:
            size_bits, size_mb, num_params
        """
        total_params = 0
        total_bits = 0

        for key, param in state_dict.items():
            num_elements = param.numel()
            total_params += num_elements

            # 假设使用 float32 (32 bits per parameter)
            total_bits += num_elements * 32

        size_mb = total_bits / (8 * 1024 * 1024)  # 转换为 MB

        return total_bits, size_mb, total_params

    def _calculate_sparse_gradient_size(self, sparse_tuple):
        """
        计算稀疏梯度大小

        Args:
            sparse_tuple: (gradient_vector, choices, shapes)

        Returns:
            size_bits, size_mb, num_params
        """
        gradient_vector, choices, shapes = sparse_tuple

        # 1. 非零梯度值的大小
        non_zero_count = len(choices)
        gradient_bits = non_zero_count * 32  # float32

        # 2. 索引的大小
        # 假设使用 int32 存储索引
        index_bits = non_zero_count * 32

        # 3. 形状信息的大小（可忽略不计）
        shape_bits = len(shapes) * 32  # 每个维度用一个 int32

        total_bits = gradient_bits + index_bits + shape_bits
        size_mb = total_bits / (8 * 1024 * 1024)

        return total_bits, size_mb, non_zero_count

    def record_client_upload(self, client_models):
        """
        记录客户端上传的通讯量

        Args:
            client_models: 字典，key=客户端ID，value=模型参数
        """
        total_bits = 0
        for client_id, params in client_models.items():
            bits, _, _ = self.calculate_model_size(params)
            total_bits += bits

        self.client_to_edge_bits += total_bits
        self.total_bits_uploaded += total_bits

        return total_bits

    def record_edge_upload(self, edge_models):
        """
        记录边缘服务器上传的通讯量

        Args:
            edge_models: 字典，key=边缘ID，value=模型参数
        """
        total_bits = 0
        for edge_id, params in edge_models.items():
            bits, _, _ = self.calculate_model_size(params)
            total_bits += bits

        self.edge_to_cloud_bits += total_bits
        self.total_bits_uploaded += total_bits

        return total_bits

    def record_cloud_broadcast(self, num_edges, model_params):
        """
        记录云服务器广播的通讯量

        Args:
            num_edges: 边缘服务器数量
            model_params: 全局模型参数
        """
        bits, _, _ = self.calculate_model_size(model_params)
        total_bits = bits * num_edges

        self.cloud_to_edge_bits += total_bits
        self.total_bits_downloaded += total_bits

        return total_bits

    def record_edge_broadcast(self, num_clients, model_params):
        """
        记录边缘服务器广播的通讯量

        Args:
            num_clients: 客户端数量
            model_params: 模型参数
        """
        bits, _, _ = self.calculate_model_size(model_params)
        total_bits = bits * num_clients

        self.edge_to_client_bits += total_bits
        self.total_bits_downloaded += total_bits

        return total_bits

    def record_round_end(self):
        """记录一轮结束"""
        self.total_rounds += 1

    def get_summary(self):
        """
        获取统计摘要

        Returns:
            summary: 字典，包含所有统计信息
        """
        total_bits = self.total_bits_uploaded + self.total_bits_downloaded
        total_mb = total_bits / (8 * 1024 * 1024)
        total_gb = total_mb / 1024

        summary = {
            # 总体统计
            'total_communication_bits': total_bits,
            'total_communication_mb': total_mb,
            'total_communication_gb': total_gb,
            'total_rounds': self.total_rounds,

            # 上行/下行
            'upload_bits': self.total_bits_uploaded,
            'download_bits': self.total_bits_downloaded,
            'upload_mb': self.total_bits_uploaded / (8 * 1024 * 1024),
            'download_mb': self.total_bits_downloaded / (8 * 1024 * 1024),

            # 分层统计
            'client_to_edge_mb': self.client_to_edge_bits / (8 * 1024 * 1024),
            'edge_to_cloud_mb': self.edge_to_cloud_bits / (8 * 1024 * 1024),
            'cloud_to_edge_mb': self.cloud_to_edge_bits / (8 * 1024 * 1024),
            'edge_to_client_mb': self.edge_to_client_bits / (8 * 1024 * 1024),

            # 平均每轮
            'avg_bits_per_round': total_bits / self.total_rounds if self.total_rounds > 0 else 0,
            'avg_mb_per_round': total_mb / self.total_rounds if self.total_rounds > 0 else 0,
        }

        return summary

    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_summary()

        print("\n" + "="*70)
        print("通讯成本统计报告")
        print("="*70)

        print(f"\n【总体统计】")
        print(f"  总通讯量: {summary['total_communication_mb']:.2f} MB ({summary['total_communication_gb']:.4f} GB)")
        print(f"  总轮数: {summary['total_rounds']}")
        print(f"  平均每轮: {summary['avg_mb_per_round']:.2f} MB")

        print(f"\n【上行/下行】")
        print(f"  上行通讯量: {summary['upload_mb']:.2f} MB")
        print(f"  下行通讯量: {summary['download_mb']:.2f} MB")

        print(f"\n【分层统计】")
        print(f"  客户端 → 边缘: {summary['client_to_edge_mb']:.2f} MB")
        print(f"  边缘 → 云: {summary['edge_to_cloud_mb']:.2f} MB")
        print(f"  云 → 边缘: {summary['cloud_to_edge_mb']:.2f} MB")
        print(f"  边缘 → 客户端: {summary['edge_to_client_mb']:.2f} MB")

        print("="*70)

        return summary

    def calculate_compression_ratio(self, baseline_metrics):
        """
        计算相对于基线的压缩比

        Args:
            baseline_metrics: 基线（无压缩）的 CommunicationMetrics 对象

        Returns:
            compression_ratio: 压缩比（baseline / current）
            reduction_percentage: 减少百分比
        """
        baseline_summary = baseline_metrics.get_summary()
        current_summary = self.get_summary()

        baseline_total = baseline_summary['total_communication_bits']
        current_total = current_summary['total_communication_bits']

        if current_total == 0:
            return float('inf'), 100.0

        compression_ratio = baseline_total / current_total
        reduction_percentage = (1 - current_total / baseline_total) * 100

        return compression_ratio, reduction_percentage


def calculate_communication_rounds(num_clients, num_edges, num_rounds):
    """
    计算理论通讯次数

    Args:
        num_clients: 客户端总数
        num_edges: 边缘服务器数量
        num_rounds: 全局训练轮数

    Returns:
        stats: 通讯次数统计
    """
    clients_per_edge = num_clients // num_edges

    stats = {
        # 每轮通讯次数
        'client_uploads_per_round': num_clients,          # 每个客户端上传一次
        'edge_uploads_per_round': num_edges,              # 每个边缘上传一次
        'cloud_broadcasts_per_round': num_edges,          # 云向每个边缘下发一次
        'edge_broadcasts_per_round': num_clients,         # 每个边缘向客户端下发

        # 总通讯次数
        'total_client_uploads': num_clients * num_rounds,
        'total_edge_uploads': num_edges * num_rounds,
        'total_cloud_broadcasts': num_edges * num_rounds,
        'total_edge_broadcasts': num_clients * num_rounds,

        # 总通讯次数（所有消息）
        'total_messages': (num_clients + num_edges) * 2 * num_rounds,
    }

    return stats
