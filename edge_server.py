import torch
import copy
from privacy import reshape_gradients


class EdgeServer:
    """边缘服务器类（支持差分隐私）"""

    def __init__(self, edge_id, client_ids, device='cpu'):
        """
        初始化边缘服务器

        Args:
            edge_id: 边缘服务器ID
            client_ids: 该边缘服务器管理的客户端ID列表
            device: 计算设备
        """
        self.edge_id = edge_id
        self.client_ids = client_ids
        self.device = device
        self.model = None
        self.clients = {}

    def set_model(self, global_model):
        """接收来自云服务器的全局模型"""
        self.model = copy.deepcopy(global_model)
        self.model.to(self.device)

    def register_client(self, client):
        """注册客户端"""
        self.clients[client.client_id] = client

    def distribute_model_to_clients(self):
        """将模型分发给所有客户端"""
        for client_id in self.client_ids:
            if client_id in self.clients:
                self.clients[client_id].set_model(self.model)

    def aggregate_client_models(self, client_models, use_dp=False):
        """
        聚合客户端模型参数（支持差分隐私）

        Args:
            client_models: 字典，key为客户端ID
                         - 不使用DP：value为模型参数字典
                         - 使用DP：value为 (梯度向量, top-k索引, 梯度形状) 元组
            use_dp: 是否使用差分隐私

        Returns:
            聚合后的模型参数
        """
        if len(client_models) == 0:
            return self.model.state_dict()

        if not use_dp:
            # 标准FedAvg聚合
            return self._aggregate_standard(client_models)
        else:
            # 差分隐私聚合
            return self._aggregate_with_dp(client_models)

    def _aggregate_standard(self, client_models):
        """
        标准FedAvg聚合（不使用差分隐私）

        Args:
            client_models: 字典，key为客户端ID，value为模型参数

        Returns:
            聚合后的模型参数
        """
        # 初始化聚合后的参数
        aggregated_params = {}

        # 获取第一个客户端的参数作为模板
        first_client_params = list(client_models.values())[0]

        # 对每个参数进行平均
        for key in first_client_params.keys():
            # 将所有客户端的该参数相加
            aggregated_params[key] = torch.zeros_like(first_client_params[key])
            for client_id, params in client_models.items():
                aggregated_params[key] += params[key]

            # 取平均
            aggregated_params[key] = aggregated_params[key] / len(client_models)

        # 更新边缘服务器的模型
        self.model.load_state_dict(aggregated_params)

        return aggregated_params

    def _aggregate_with_dp(self, client_models):
        """
        差分隐私聚合（处理稀疏梯度）

        Args:
            client_models: 字典，key为客户端ID，value为(梯度向量, top-k索引, 梯度形状)

        Returns:
            聚合后的模型参数
        """
        # 获取第一个客户端的梯度形状信息
        first_client_data = list(client_models.values())[0]
        gradient_vector, _, shapes = first_client_data

        # 初始化聚合后的梯度向量（全零）
        aggregated_gradient = torch.zeros_like(gradient_vector)

        # 累加所有客户端的梯度
        for client_id, (grad_vector, choices, _) in client_models.items():
            aggregated_gradient += grad_vector

        # 取平均
        aggregated_gradient = aggregated_gradient / len(client_models)

        # 将展平的梯度重塑回原始形状
        gradient_list = reshape_gradients(aggregated_gradient, shapes)

        # 获取当前模型参数
        current_params = self.model.state_dict()
        aggregated_params = {}

        # 更新参数：新参数 = 旧参数 + 聚合梯度
        param_idx = 0
        for key in current_params.keys():
            aggregated_params[key] = current_params[key] + gradient_list[param_idx]
            param_idx += 1

        # 更新边缘服务器的模型
        self.model.load_state_dict(aggregated_params)

        return aggregated_params

    def get_model_parameters(self):
        """返回模型参数"""
        return copy.deepcopy(self.model.state_dict())
