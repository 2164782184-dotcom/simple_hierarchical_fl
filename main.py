import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

from models import get_model
from data_utils import load_mnist, split_data_to_clients, create_data_loaders
from client import Client
from edge_server import EdgeServer
from cloud_server import CloudServer


def main():
    # ==================== 配置参数 ====================
    NUM_EDGES = 5                    # 边缘服务器数量
    NUM_CLIENTS_PER_EDGE = 10         # 每个边缘服务器的客户端数量
    NUM_CLIENTS = NUM_EDGES * NUM_CLIENTS_PER_EDGE  # 总客户端数量
    NUM_ROUNDS = 100                  # 全局训练轮数
    LOCAL_EPOCHS = 60                 # 客户端本地训练轮数
    BATCH_SIZE = 20                  # 批次大小
    LEARNING_RATE = 0.01             # 学习率
    LR_DECAY = 0.995                 # 学习率衰减（gamma）
    LR_DECAY_EPOCH = 1               # 每1个epoch执行一次学习率衰减
    MOMENTUM = 0                     # 0 表示不使用动量
    WEIGHT_DECAY = 0                 # 权值衰减
    CLIENT_IID = False                # 客户端数据是否IID分布
    EDGE_IID = True                   # 边缘服务器数据是否IID分布
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*50)
    print("简单分层联邦学习 (Simple Hierarchical FL)")
    print("="*50)
    print(f"设备: {DEVICE}")
    print(f"边缘服务器数量: {NUM_EDGES}")
    print(f"每个边缘服务器的客户端数量: {NUM_CLIENTS_PER_EDGE}")
    print(f"总客户端数量: {NUM_CLIENTS}")
    print(f"全局轮数: {NUM_ROUNDS}")
    print(f"本地训练轮数: {LOCAL_EPOCHS}")
    print(f"客户端数据分布: {'IID' if CLIENT_IID else 'Non-IID'}")
    print(f"边缘服务器数据分布: {'IID' if EDGE_IID else 'Non-IID'}")
    print("="*50)

    # ==================== 加载数据 ====================
    print("\n[1/6] 加载MNIST数据集...")
    train_dataset, test_dataset = load_mnist()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ==================== 分配数据给客户端 ====================
    print("[2/6] 将数据分配给客户端...")
    client_data_indices = split_data_to_clients(train_dataset, NUM_CLIENTS, NUM_EDGES,
                                                 client_iid=CLIENT_IID, edge_iid=EDGE_IID)
    client_loaders = create_data_loaders(train_dataset, client_data_indices, BATCH_SIZE)

    # ==================== 初始化模型和服务器 ====================
    print("[3/6] 初始化云服务器、边缘服务器和客户端...")
    global_model = get_model()
    cloud_server = CloudServer(global_model, device=DEVICE)

    # 创建边缘服务器
    edge_servers = []
    for edge_id in range(NUM_EDGES):
        client_ids = list(range(edge_id * NUM_CLIENTS_PER_EDGE,
                                (edge_id + 1) * NUM_CLIENTS_PER_EDGE))
        edge_server = EdgeServer(edge_id, client_ids, device=DEVICE)
        cloud_server.register_edge_server(edge_server)
        edge_servers.append(edge_server)

    # 创建客户端
    clients = []
    for client_id in range(NUM_CLIENTS):
        client = Client(client_id, client_loaders[client_id], device=DEVICE)
        clients.append(client)

        # 将客户端注册到对应的边缘服务器
        edge_id = client_id // NUM_CLIENTS_PER_EDGE
        edge_servers[edge_id].register_client(client)

    # ==================== 开始训练 ====================
    print("[4/6] 开始分层联邦学习训练...\n")

    accuracy_history = []
    loss_history = []

    for round_idx in range(NUM_ROUNDS):
        print(f"{'='*50}")
        print(f"轮次 {round_idx + 1}/{NUM_ROUNDS}")
        print(f"{'='*50}")

        # 步骤1: 云服务器分发模型给边缘服务器
        cloud_server.distribute_model_to_edges()

        edge_models = {}

        # 对每个边缘服务器
        for edge_server in edge_servers:
            print(f"\n边缘服务器 {edge_server.edge_id}:")

            # 步骤2: 边缘服务器分发模型给客户端
            edge_server.distribute_model_to_clients()

            # 步骤3: 客户端本地训练
            client_models = {}
            for client_id in edge_server.client_ids:
                client = clients[client_id]
                train_loss = client.train(LOCAL_EPOCHS, LEARNING_RATE, LR_DECAY_EPOCH, LR_DECAY, MOMENTUM, WEIGHT_DECAY)
                client_models[client_id] = client.get_model_parameters()
                print(f"  客户端 {client_id} 训练完成, 损失: {train_loss:.4f}")

            # 步骤4: 边缘服务器聚合客户端模型
            edge_server.aggregate_client_models(client_models)
            edge_models[edge_server.edge_id] = edge_server.get_model_parameters()
            print(f"  边缘服务器 {edge_server.edge_id} 聚合完成")

        # 步骤5: 云服务器聚合边缘服务器模型
        cloud_server.aggregate_edge_models(edge_models)
        print(f"\n云服务器聚合完成")

        # 步骤6: 评估全局模型
        accuracy, test_loss = cloud_server.evaluate(test_loader)
        accuracy_history.append(accuracy)
        loss_history.append(test_loss)

        print(f"\n轮次 {round_idx + 1} 结果:")
        print(f"  测试准确率: {accuracy:.2f}%")
        print(f"  测试损失: {test_loss:.4f}")

    # ==================== 训练完成 ====================
    print(f"\n{'='*50}")
    print("训练完成!")
    print(f"{'='*50}")
    print(f"最终测试准确率: {accuracy_history[-1]:.2f}%")
    print(f"最终测试损失: {loss_history[-1]:.4f}")

    # ==================== 绘制结果 ====================
    print("\n[5/6] 绘制训练曲线...")

    plt.figure(figsize=(12, 5))

    # 绘制准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_ROUNDS + 1), accuracy_history, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('轮次 (Round)', fontsize=12)
    plt.ylabel('准确率 (%)', fontsize=12)
    plt.title('测试准确率变化', fontsize=14)
    plt.grid(True, alpha=0.3)

    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_ROUNDS + 1), loss_history, 'r-o', linewidth=2, markersize=6)
    plt.xlabel('轮次 (Round)', fontsize=12)
    plt.ylabel('损失 (Loss)', fontsize=12)
    plt.title('测试损失变化', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    print("训练曲线已保存到 training_results.png")

    print("\n[6/6] 完成!")


if __name__ == "__main__":
    main()
