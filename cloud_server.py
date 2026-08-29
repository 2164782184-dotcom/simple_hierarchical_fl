import torch
import copy


class CloudServer:
    """云服务器类"""

    def __init__(self, model, device='cpu'):
        """
        初始化云服务器

        Args:
            model: 全局模型
            device: 计算设备
        """
        self.model = model.to(device)
        self.device = device
        self.edge_servers = {}

    def register_edge_server(self, edge_server):
        """注册边缘服务器"""
        self.edge_servers[edge_server.edge_id] = edge_server

    def distribute_model_to_edges(self):
        """将全局模型分发给所有边缘服务器"""
        for edge_id, edge_server in self.edge_servers.items():
            edge_server.set_model(self.model)

    def aggregate_edge_models(self, edge_models):
        """
        聚合边缘服务器模型参数（FedAvg算法）

        Args:
            edge_models: 字典，key为边缘服务器ID，value为模型参数

        Returns:
            聚合后的模型参数
        """
        if len(edge_models) == 0:
            return self.model.state_dict()

        # 初始化聚合后的参数
        aggregated_params = {}

        # 获取第一个边缘服务器的参数作为模板
        first_edge_params = list(edge_models.values())[0]

        # 对每个参数进行平均
        for key in first_edge_params.keys():
            # 将所有边缘服务器的该参数相加
            aggregated_params[key] = torch.zeros_like(first_edge_params[key])
            for edge_id, params in edge_models.items():
                aggregated_params[key] += params[key]

            # 取平均
            aggregated_params[key] = aggregated_params[key] / len(edge_models)

        # 更新云服务器的全局模型
        self.model.load_state_dict(aggregated_params)

        return aggregated_params

    def evaluate(self, test_loader):
        """
        在测试集上评估全局模型

        Args:
            test_loader: 测试数据加载器

        Returns:
            准确率和损失
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        test_loss /= len(test_loader)
        accuracy = 100. * correct / total

        return accuracy, test_loss

    def get_model(self):
        """返回全局模型"""
        return self.model
