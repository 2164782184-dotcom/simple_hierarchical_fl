import torch
import copy


class EdgeServer:
    """边缘服务器类"""

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

    def aggregate_client_models(self, client_models):
        """
        聚合客户端模型参数（FedAvg算法）

        Args:
            client_models: 字典，key为客户端ID，value为模型参数

        Returns:
            聚合后的模型参数
        """
        if len(client_models) == 0:
            return self.model.state_dict()

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

    def get_model_parameters(self):
        """返回模型参数"""
        return copy.deepcopy(self.model.state_dict())
