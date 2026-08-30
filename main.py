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

# ==================== 配置参数 ====================
NUM_EDGES = 5                         # 边缘服务器数量
NUM_CLIENTS_PER_EDGE = 10             # 每个边缘服务器的客户端数量
NUM_CLIENTS = NUM_EDGES * NUM_CLIENTS_PER_EDGE  # 总客户端数量
NUM_ROUNDS = 100                      # 全局训练轮数
LOCAL_EPOCHS = 60                     # 客户端本地训练轮数（MAML迭代次数）
BATCH_SIZE = 20                       # 批次大小
LEARNING_RATE = 0.01                  # 初始学习率（MAML内层学习率）
LR_DECAY = 0.995                      # 学习率衰减系数（gamma）
LR_DECAY_EPOCH = 1                    # 每1个epoch执行一次学习率衰减
MOMENTUM = 0                          # SGD动量（0表示不使用动量）
WEIGHT_DECAY = 0                      # 权重衰减/L2正则化（0表示不使用）
TRAIN_FRACTION = 0.01                 # 训练集的使用比例（0.01=使用1%数据）
CLIENT_IID = False                    # 客户端数据是否IID分布
EDGE_IID = True                       # 边缘服务器数据是否IID分布
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================== MAML元学习配置 ====================
SUPPORT_RATIO = 0.8                   # support集占比（0.8 = 80% support, 20% query）
LEARNING_RATE_OUTER = 0.001           # MAML外层学习率（query集，Adam）
TRAIN_INNER_STEP = 5                  # support集采样batch数（0=遍历全部，>0=采样指定数量batch）
TEST_INNER_STEP = 5                   # query集采样batch数（0=遍历全部，>0=采样指定数量batch）

# ==================== 差分隐私配置 ====================
USE_DP = True                         # 是否使用差分隐私
DP_EPSILON = 0.5                      # 隐私预算（越小隐私保护越强，典型值：0.1-10）
DP_DELTA = 0.001                      # 失败概率（典型值：1e-5 到 1e-7）
DP_CLIP_C = 0.01                      # 梯度裁剪阈值
DP_RATE = 50                          # 稀疏化率（rate=50 表示保留 2% 的梯度）
DP_MECHANISM = 'laplace'              # 噪声机制（'laplace' 或 'gaussian'）

# ==================== 知识蒸馏配置 ====================
USE_DISTILLATION = True               # 是否使用教师-学生互蒸馏
DISTILL_TEMPERATURE = 3.0             # 蒸馏温度（T），越大软标签越平滑（典型值：1-10）
DISTILL_ALPHA = 0.4                   # KL散度损失权重（典型值：0.3-0.7）
DISTILL_BETA = 0.3                    # MSE特征损失权重（典型值：0.1-0.4）
# 注意：当前版本实现的是 MAML + KL蒸馏，不包含特征层MSE（可扩展）
# 实际损失 = (1-alpha)*CE + alpha*KL

# ==================== TensorBoard配置 ====================
USE_TENSORBOARD = True                # 是否启用TensorBoard实时可视化

# ==================== 初始化 ====================
if USE_TENSORBOARD:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/hierfl_maml_kd_{timestamp}'
    writer = SummaryWriter(log_dir)
else:
    writer = None
    log_dir = None

print("="*70)
print("分层联邦学习 - MAML + KL蒸馏")
print("="*70)
print(f"设备: {DEVICE}")
print(f"边缘服务器数量: {NUM_EDGES}")
print(f"客户端数量: {NUM_CLIENTS} ({NUM_CLIENTS_PER_EDGE} per edge)")
print(f"全局轮数: {NUM_ROUNDS}")
print(f"本地MAML迭代: {LOCAL_EPOCHS}")
print(f"批次大小: {BATCH_SIZE}")
print(f"学习率: 内层={LEARNING_RATE}, 外层={LEARNING_RATE_OUTER}")
print(f"学习率衰减: γ={LR_DECAY}, 每{LR_DECAY_EPOCH}轮")
print(f"动量: {MOMENTUM}, 权重衰减: {WEIGHT_DECAY}")
print(f"训练数据比例: {TRAIN_FRACTION:.1%}")
print(f"数据分布: 客户端{'IID' if CLIENT_IID else 'Non-IID'}, 边缘{'IID' if EDGE_IID else 'Non-IID'}")
print(f"Support/Query比例: {SUPPORT_RATIO:.1%}/{1-SUPPORT_RATIO:.1%}")
print(f"知识蒸馏: {'启用' if USE_DISTILLATION else '禁用'}")
if USE_DISTILLATION:
    print(f"  - 温度: {DISTILL_TEMPERATURE}")
    print(f"  - KL权重α: {DISTILL_ALPHA}")
    print(f"  - 特征权重β: {DISTILL_BETA} (当前版本未实现)")
print(f"差分隐私: {'启用' if USE_DP else '禁用'}")
if USE_DP:
    print(f"  - ε: {DP_EPSILON}, δ: {DP_DELTA}")
    print(f"  - 裁剪阈值: {DP_CLIP_C}, 稀疏化率: {DP_RATE}")
    print(f"  - 噪声机制: {DP_MECHANISM}")
if USE_TENSORBOARD:
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
        device=DEVICE,
        train_inner_step=TRAIN_INNER_STEP,
        test_inner_step=TEST_INNER_STEP
    )
    # 设置MAML学习率
    client.inner_lr = LEARNING_RATE
    client.outer_lr = LEARNING_RATE_OUTER
    clients.append(client)

print(f"初始化完成：1个云服务器，{NUM_EDGES}个边缘服务器，{NUM_CLIENTS}个客户端")

# ==================== 训练循环 ====================
print("\n开始训练...")
print("="*70)

start_time = time.time()
current_lr_inner = LEARNING_RATE  # 跟踪当前内层学习率

for round_idx in range(NUM_ROUNDS):
    round_start_time = time.time()

    print(f"\n{'='*70}")
    print(f"全局轮次 {round_idx + 1}/{NUM_ROUNDS}")
    print(f"当前学习率: 内层={current_lr_inner:.6f}, 外层={LEARNING_RATE_OUTER:.6f}")
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
        # 更新客户端的学习率
        client.inner_lr = current_lr_inner

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

        if (client_id + 1) % 10 == 0:
            print(f"  客户端 {client_id}: loss={loss:.4f}, samples={num_samples}")

    avg_client_loss = sum(client_losses) / len(client_losses)
    print(f"\n平均客户端损失: {avg_client_loss:.4f}")

    # === 学习率衰减 ===
    if (round_idx + 1) % LR_DECAY_EPOCH == 0:
        current_lr_inner *= LR_DECAY
        print(f"学习率衰减: {current_lr_inner:.6f}")

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
    if USE_TENSORBOARD and writer is not None:
        writer.add_scalar('Accuracy/test', test_accuracy, round_idx)
        writer.add_scalar('Loss/average_client', avg_client_loss, round_idx)
        writer.add_scalar('Time/round', round_time, round_idx)
        writer.add_scalar('LearningRate/inner', current_lr_inner, round_idx)
        writer.add_scalar('LearningRate/outer', LEARNING_RATE_OUTER, round_idx)

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
if USE_TENSORBOARD:
    print(f"TensorBoard日志保存在: {log_dir}")
    print(f"\n查看训练结果：")
    print(f"  tensorboard --logdir={log_dir}")
print("="*70)

# 关闭 TensorBoard
if USE_TENSORBOARD and writer is not None:
    writer.close()
