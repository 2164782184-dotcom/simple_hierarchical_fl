"""
边缘服务器模块
========================================
功能：
1. 接收客户端上传的模型
2. 聚合客户端模型（支持标准FedAvg和差分隐私）
3. 与云服务器通信
"""

import torch
import copy
from models import LeNet


class EdgeServer:
    """边缘服务器类"""

    def __init__(self, edge_id):
        """
        初始化边缘服务器

        Args:
            edge_id: 边缘服务器ID
        """
        self.edge_id = edge_id
        self.model = LeNet()
        self.client_updates = {}  # 存储客户端上传的模型/梯度
        self.cloud_buffer = None  # 存储来自云服务器的模型

    def receive_from_client(self, client_id, cshared_state_dict, use_dp=False):
        """
        接收来自客户端的模型更新

        Args:
            client_id: 客户端ID
            cshared_state_dict: 客户端的模型参数或梯度
            use_dp: 是否使用差分隐私
        """
        self.client_updates[client_id] = {
            'params': cshared_state_dict,
            'use_dp': use_dp
        }

    def aggregate(self, use_dp=False):
        """
        聚合所有客户端的模型

        Args:
            use_dp: 是否使用差分隐私模式
        """
        if len(self.client_updates) == 0:
            return

        if use_dp:
            self._aggregate_with_dp()
        else:
            self._aggregate_standard()

        # 清空客户端更新缓存
        self.client_updates = {}

    def _aggregate_standard(self):
        """标准FedAvg聚合"""
        aggregated_params = {}
        num_clients = len(self.client_updates)

        # 获取第一个客户端的参数作为模板
        first_client_params = list(self.client_updates.values())[0]['params']

        # 对每个参数进行平均
        for key in first_client_params.keys():
            aggregated_params[key] = torch.zeros_like(first_client_params[key])
            for client_data in self.client_updates.values():
                aggregated_params[key] += client_data['params'][key]
            aggregated_params[key] = aggregated_params[key] / num_clients

        # 更新边缘服务器的模型
        self.model.load_state_dict(aggregated_params)

    def _aggregate_with_dp(self):
        """差分隐私聚合（处理稀疏梯度）"""
        # 获取当前模型参数
        current_params = self.model.state_dict()
        aggregated_params = {}
        num_clients = len(self.client_updates)

        # 初始化聚合参数
        for key in current_params.keys():
            aggregated_params[key] = torch.zeros_like(current_params[key])

        # 累加所有客户端的梯度（差分）
        for client_data in self.client_updates.values():
            for key in client_data['params']:
                aggregated_params[key] += client_data['params'][key]

        # 取平均并更新参数：新参数 = 旧参数 + 平均梯度
        for key in current_params.keys():
            aggregated_params[key] = current_params[key] + aggregated_params[key] / num_clients

        # 更新边缘服务器的模型
        self.model.load_state_dict(aggregated_params)

    def send_to_cloudserver(self, cloud_server):
        """
        将模型上传到云服务器

        Args:
            cloud_server: 云服务器对象
        """
        cloud_server.receive_from_edgeserver(
            edge_id=self.edge_id,
            eshared_state_dict=copy.deepcopy(self.model.state_dict())
        )

    def receive_from_cloudserver(self, shared_state_dict):
        """
        接收来自云服务器的全局模型

        Args:
            shared_state_dict: 云服务器下发的模型参数
        """
        self.cloud_buffer = shared_state_dict

    def sync_with_cloudserver(self):
        """与云服务器同步模型"""
        if self.cloud_buffer is not None:
            self.model.load_state_dict(self.cloud_buffer)

    def get_model(self):
        """获取当前模型参数"""
        return copy.deepcopy(self.model.state_dict())
