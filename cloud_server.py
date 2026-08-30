"""
云服务器模块
========================================
功能：
1. 接收边缘服务器上传的模型
2. 聚合边缘服务器模型（FedAvg）
3. 下发全局模型到边缘服务器
"""

import torch
import copy
from models import LeNet


class CloudServer:
    """云服务器类"""

    def __init__(self):
        """初始化云服务器"""
        self.model = LeNet()
        self.edge_updates = {}  # 存储边缘服务器上传的模型

    def receive_from_edgeserver(self, edge_id, eshared_state_dict):
        """
        接收来自边缘服务器的模型

        Args:
            edge_id: 边缘服务器ID
            eshared_state_dict: 边缘服务器的模型参数
        """
        self.edge_updates[edge_id] = eshared_state_dict

    def aggregate(self):
        """聚合所有边缘服务器的模型（FedAvg）"""
        if len(self.edge_updates) == 0:
            return

        aggregated_params = {}
        num_edges = len(self.edge_updates)

        # 获取第一个边缘服务器的参数作为模板
        first_edge_params = list(self.edge_updates.values())[0]

        # 对每个参数进行平均
        for key in first_edge_params.keys():
            aggregated_params[key] = torch.zeros_like(first_edge_params[key])
            for edge_params in self.edge_updates.values():
                aggregated_params[key] += edge_params[key]
            aggregated_params[key] = aggregated_params[key] / num_edges

        # 更新云服务器的全局模型
        self.model.load_state_dict(aggregated_params)

        # 清空边缘服务器更新缓存
        self.edge_updates = {}

    def get_model(self):
        """获取全局模型参数"""
        return copy.deepcopy(self.model.state_dict())
