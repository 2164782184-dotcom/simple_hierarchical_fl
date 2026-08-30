"""
分层联邦学习主程序
========================================
功能：
1. MAML元学习 + KL蒸馏的混合训练
2. 差分隐私保护
3. TensorBoard实时可视化
4. 支持多种数据分布模式
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from data_utils import load_mnist, split_data_to_clients, create_data_loaders
from client import Client
from edge_server import EdgeServer
from cloud_server import CloudServer
import time
from datetime import datetime

# ==================== 超参数配置 ====================

# 网络结构
NUM_EDGES = 3                    # 边缘服务器数量
NUM_CLIENTS_PER_EDGE = 5         # 每个边缘服务器的客户端数量
NUM_CLIENTS = NUM_EDGES * NUM_CLIENTS_PER_EDGE

# 训练配置
NUM_ROUNDS = 50                  # 全局训练轮数
LOCAL_EPOCHS = 5                 # 每个客户端的本地训练轮数（MAML迭代次数）
BATCH_SIZE = 32                  # 批次大小
LEARNING_RATE_INNER = 0.01       # MAML内层学习率（support集）
LEARNING_RATE_OUTER = 0.001      # MAML外层学习率（query集，Adam）

# 数据配置
TRAIN_FRACTION = 1.0             # 使用的训练数据比例
SUPPORT_RATIO = 0.8              # support集占比（0.8 = 80% support, 20% query）
CLIENT_IID = False               # 客户端数据是否IID
EDGE_IID = True                  # 边缘服务器数据是否IID

# 知识蒸馏配置
USE_DISTILLATION = True          # 是否使用知识蒸馏
DISTILL_TEMPERATURE = 3.0        # KL散度的温度参数
DISTILL_ALPHA = 0.3              # 蒸馏损失权重（loss = (1-α)*CE + α*KL）

# 差分隐私配置
USE_DP = False                   # 是否使用差分隐私
DP_EPSILON = 1.0                 # 隐私预算 epsilon
DP_DELTA = 1e-5                  # 隐私预算 delta
DP_CLIP_C = 1.0                  # 梯度裁剪阈值
DP_RATE = 0.1                    # top-k 稀疏化比例

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TensorBoard配置
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f'runs/hierfl_maml_kd_{timestamp}'
writer = SummaryWriter(log_dir)

print("="*70)
print("分层联邦学习 - MAML + KL蒸馏")
print("="*70)
print(f"设备: {DEVICE}")
print(f"边缘服务器数量: {NUM_EDGES}")
print(f"客户端数量: {NUM_CLIENTS} ({NUM_CLIENTS_PER_EDGE} per edge)")
print(f"全局轮数: {NUM_ROUNDS}")
print(f"本地MAML迭代: {LOCAL_EPOCHS}")
print(f"数据分布: 客户端{'IID' if CLIENT_IID else 'Non-IID'}, 边缘{'IID' if EDGE_IID else 'Non-IID'}")
print(f"Support/Query比例: {SUPPORT_RATIO:.1%}/{1-SUPPORT_RATIO:.1%}")
print(f"知识蒸馏: {'启用' if USE_DISTILLATION else '禁用'}")
if USE_DISTILLATION:
    print(f"  - 温度: {DISTILL_TEMPERATURE}")
    print(f"  - 权重α: {DISTILL_ALPHA}")
print(f"差分隐私: {'启用' if USE_DP else '禁用'}")
if USE_DP:
    print(f"  - ε: {DP_EPSILON}, δ: {DP_DELTA}")
    print(f"  - 裁剪阈值: {DP_CLIP_C}, 稀疏化率: {DP_RATE}")
print(f"TensorBoard日志: {log_dir}")
print("="*70)

# ==================== 数据加载 ====================
print("\n加载数据集...")
train_dataset, test_dataset = load_mnist(train_fraction=TRAIN_FRACTION)

print("分配数据到客户端...")
client_data_indices = split_data_to_clients(
    train_dataset,
    NUM_CLIENTS,
    NUM_EDGES,
    client_iid=CLIENT_IID,
    edge_iid=EDGE_IID
)

print("创建数据加载器（Support/Query划分）...")
client_support_loaders, client_query_loaders = create_data_loaders(
    train_dataset,
    client_data_indices,
    BATCH_SIZE,
    support_ratio=SUPPORT_RATIO
)

# 创建测试集加载器
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==================== 初始化服务器和客户端 ====================
print("\n初始化云服务器、边缘服务器和客户端...")

# 云服务器
cloud_server = CloudServer()

# 边缘服务器
edge_servers = [EdgeServer(edge_id=i) for i in range(NUM_EDGES)]

# 客户端
clients = []
for client_id in range(NUM_CLIENTS):
    client = Client(
        client_id=client_id,
        support_loader=client_support_loaders[client_id],
        query_loader=client_query_loaders[client_id],
        test_loader=test_loader,
        device=DEVICE
    )
    clients.append(client)

print(f"初始化完成：1个云服务器，{NUM_EDGES}个边缘服务器，{NUM_CLIENTS}个客户端")

# ==================== 训练循环 ====================
print("\n开始训练...")
print("="*70)

start_time = time.time()

for round_idx in range(NUM_ROUNDS):
    round_start_time = time.time()

    print(f"\n{'='*70}")
    print(f"全局轮次 {round_idx + 1}/{NUM_ROUNDS}")
    print(f"{'='*70}")

    # === 第1步：客户端从边缘服务器同步模型 ===
    for edge_id, edge_server in enumerate(edge_servers):
        edge_client_start = edge_id * NUM_CLIENTS_PER_EDGE
        edge_client_end = edge_client_start + NUM_CLIENTS_PER_EDGE

        for client_id in range(edge_client_start, edge_client_end):
            client = clients[client_id]
            # 从边缘服务器获取模型
            if round_idx == 0:
                # 第一轮：使用初始模型
                client.receive_from_edgeserver(edge_server.get_model())
            else:
                # 后续轮次：从边缘服务器获取聚合后的模型
                client.receive_from_edgeserver(edge_server.get_model())
            client.sync_with_edgeserver()

            # === 知识蒸馏：设置教师模型 ===
            if USE_DISTILLATION and round_idx > 0:
                # 使用上一轮的全局模型作为教师
                client.set_teacher_model(cloud_server.get_model())

    # === 第2步：客户端本地训练（MAML + KL蒸馏）===
    print("\n客户端本地训练...")
    client_losses = []
    client_samples = []

    for client_id, client in enumerate(clients):
        # MAML元学习 + KL蒸馏
        loss, num_samples = client.train(
            num_iter=LOCAL_EPOCHS,
            use_distillation=USE_DISTILLATION,
            temperature=DISTILL_TEMPERATURE,
            alpha=DISTILL_ALPHA,
            use_dp=USE_DP,
            dp_epsilon=DP_EPSILON,
            dp_delta=DP_DELTA,
            dp_clip_c=DP_CLIP_C,
            dp_rate=DP_RATE
        )
        client_losses.append(loss)
        client_samples.append(num_samples)

        if (client_id + 1) % 5 == 0:
            print(f"  客户端 {client_id}: loss={loss:.4f}, samples={num_samples}")

    avg_client_loss = sum(client_losses) / len(client_losses)
    print(f"\n平均客户端损失: {avg_client_loss:.4f}")

    # === 第3步：客户端上传模型到边缘服务器 ===
    print("\n客户端上传模型到边缘服务器...")
    for edge_id, edge_server in enumerate(edge_servers):
        edge_client_start = edge_id * NUM_CLIENTS_PER_EDGE
        edge_client_end = edge_client_start + NUM_CLIENTS_PER_EDGE

        for client_id in range(edge_client_start, edge_client_end):
            client = clients[client_id]
            client.send_to_edgeserver(
                edge_server,
                use_dp=USE_DP,
                dp_epsilon=DP_EPSILON,
                dp_delta=DP_DELTA,
                dp_clip_c=DP_CLIP_C,
                dp_rate=DP_RATE
            )

    # === 第4步：边缘服务器聚合 ===
    print("边缘服务器聚合模型...")
    for edge_server in edge_servers:
        edge_server.aggregate(use_dp=USE_DP)

    # === 第5步：边缘服务器上传到云服务器 ===
    print("边缘服务器上传到云服务器...")
    for edge_server in edge_servers:
        edge_server.send_to_cloudserver(cloud_server)

    # === 第6步：云服务器聚合 ===
    print("云服务器聚合全局模型...")
    cloud_server.aggregate()

    # === 第7步：云服务器下发到边缘服务器 ===
    print("云服务器下发到边缘服务器...")
    global_model = cloud_server.get_model()
    for edge_server in edge_servers:
        edge_server.receive_from_cloudserver(global_model)
        edge_server.sync_with_cloudserver()

    # === 第8步：测试全局模型 ===
    print("\n测试全局模型...")
    clients[0].set_model_params(global_model)
    test_accuracy = clients[0].test()

    round_time = time.time() - round_start_time

    print(f"\n{'='*70}")
    print(f"轮次 {round_idx + 1} 完成")
    print(f"  测试准确率: {test_accuracy:.4f}")
    print(f"  平均损失: {avg_client_loss:.4f}")
    print(f"  耗时: {round_time:.2f}秒")
    print(f"{'='*70}")

    # === TensorBoard记录 ===
    writer.add_scalar('Accuracy/test', test_accuracy, round_idx)
    writer.add_scalar('Loss/average_client', avg_client_loss, round_idx)
    writer.add_scalar('Time/round', round_time, round_idx)

    # 记录每个客户端的损失分布
    for client_id, loss in enumerate(client_losses):
        writer.add_scalar(f'Loss/client_{client_id}', loss, round_idx)

# ==================== 训练结束 ====================
total_time = time.time() - start_time

print("\n" + "="*70)
print("训练完成！")
print("="*70)
print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
print(f"平均每轮: {total_time/NUM_ROUNDS:.2f}秒")
print(f"最终测试准确率: {test_accuracy:.4f}")
print(f"TensorBoard日志保存在: {log_dir}")
print("="*70)

# 关闭 TensorBoard
writer.close()

print("\n查看训练结果：")
print(f"  tensorboard --logdir={log_dir}")
