import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti SC', 'Hiragino Sans GB', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

from models import get_model
from data_utils import load_mnist, split_data_to_clients, create_data_loaders
from client import Client, DPConfig
from edge_server import EdgeServer
from cloud_server import CloudServer


def main():
    # ==================== 配置参数 ====================
    NUM_EDGES = 5                         # 边缘服务器数量
    NUM_CLIENTS_PER_EDGE = 10             # 每个边缘服务器的客户端数量
    NUM_CLIENTS = NUM_EDGES * NUM_CLIENTS_PER_EDGE  # 总客户端数量
    NUM_ROUNDS = 100                      # 全局训练轮数
    NUM_EDGE_AGGREGATION = 1              # 边缘聚合次数（每次云聚合前，边缘进行多少轮本地聚合）
    LOCAL_EPOCHS = 60                     # 客户端本地训练轮数
    BATCH_SIZE = 20                       # 批次大小
    LEARNING_RATE = 0.01                  # 初始学习率
    LR_DECAY = 0.995                      # 学习率衰减系数（gamma）
    LR_DECAY_EPOCH = 1                    # 每1个epoch执行一次学习率衰减
    MOMENTUM = 0                          # SGD动量（0表示不使用动量）
    WEIGHT_DECAY = 0                      # 权重衰减/L2正则化（0表示不使用）
    TRAIN_FRACTION = 0.01                 # 训练集的使用比例（0.01=使用1%数据）
    CLIENT_IID = False                    # 客户端数据是否IID分布
    EDGE_IID = True                       # 边缘服务器数据是否IID分布
    FRAC = 1.0                            # 每轮参与训练的客户端比例（1.0=全部参与）
    SEED = 42                             # 随机种子（用于实验可复现性）
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ==================== 差分隐私配置 ====================
    USE_DP = True                         # 是否使用差分隐私
    DP_EPSILON = 0.5                      # 隐私预算（越小隐私保护越强，典型值：0.1-10）
    DP_DELTA = 0.001                      # 失败概率（典型值：1e-5 到 1e-7）
    DP_CLIP_C = 0.01                      # 梯度裁剪阈值
    DP_RATE = 50                          # 稀疏化率（rate=50 表示保留 2% 的梯度）
    DP_MECHANISM = 'laplace'              # 噪声机制（'laplace' 或 'gaussian'）

    # ==================== MAML元学习配置 ====================
    USE_MAML = True                       # 是否使用MAML元学习
    SUPPORT_RATIO = 0.5                   # support集占比（0.8 = 80% support, 20% query）
    BETA = 0.001                          # MAML外层学习率（Adam优化器）
    TRAIN_INNER_STEP = 5                  # support集采样batch数（0=遍历全部，>0=采样指定数量batch）
    TEST_INNER_STEP = 5                   # query集采样batch数（0=遍历全部，>0=采样指定数量batch）

    # ==================== 知识蒸馏配置 ====================
    USE_DISTILLATION = True               # 是否使用KL蒸馏
    DISTILL_TEMPERATURE = 3.0             # 蒸馏温度参数
    DISTILL_ALPHA = 0.4                   # KL损失权重（软标签）
    DISTILL_BETA = 0.3                    # MSE损失权重（特征层）
    USE_DP_DISTILLATION = True            # 是否对蒸馏过程使用差分隐私（保护教师模型输出）
    # 总损失 = (1-α-β)*CE + α*KL + β*MSE

    # ==================== TensorBoard配置 ====================
    USE_TENSORBOARD = True                # 是否启用TensorBoard实时可视化

    # ==================== 设置随机种子 ====================
    import random
    import numpy as np
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("="*70)
    print("分层联邦学习 + 差分隐私 + MAML + KL蒸馏")
    print("="*70)
    print(f"设备: {DEVICE}")
    print(f"边缘服务器数量: {NUM_EDGES}")
    print(f"每个边缘服务器的客户端数量: {NUM_CLIENTS_PER_EDGE}")
    print(f"总客户端数量: {NUM_CLIENTS}")
    print(f"全局轮数: {NUM_ROUNDS}")
    print(f"本地训练轮数: {LOCAL_EPOCHS}")
    print(f"客户端数据分布: {'IID' if CLIENT_IID else 'Non-IID'}")
    print(f"边缘服务器数据分布: {'IID' if EDGE_IID else 'Non-IID'}")
    print(f"\n{'='*70}")
    print(f"MAML元学习配置:")
    print(f"  启用状态: {'是' if USE_MAML else '否'}")
    if USE_MAML:
        print(f"  Support/Query比例: {SUPPORT_RATIO:.0%}/{1-SUPPORT_RATIO:.0%}")
        print(f"  内层学习率: {LEARNING_RATE}")
        print(f"  外层学习率: {BETA}")
        print(f"  Support采样batch数: {TRAIN_INNER_STEP if TRAIN_INNER_STEP > 0 else '全部'}")
        print(f"  Query采样batch数: {TEST_INNER_STEP if TEST_INNER_STEP > 0 else '全部'}")
    print(f"\n{'='*70}")
    print(f"知识蒸馏配置:")
    print(f"  启用状态: {'是' if USE_DISTILLATION else '否'}")
    if USE_DISTILLATION:
        print(f"  温度参数: {DISTILL_TEMPERATURE}")
        print(f"  KL损失权重α: {DISTILL_ALPHA}")
    print(f"\n{'='*70}")
    print(f"差分隐私配置:")
    print(f"  启用状态: {'是' if USE_DP else '否'}")
    if USE_DP:
        print(f"  隐私预算 ε: {DP_EPSILON}")
        print(f"  失败概率 δ: {DP_DELTA}")
        print(f"  梯度裁剪阈值: {DP_CLIP_C}")
        print(f"  稀疏化率: {DP_RATE} (保留 {100/DP_RATE:.1f}% 的梯度)")
        print(f"  噪声机制: {DP_MECHANISM}")
    print("="*70)

    # ==================== 初始化TensorBoard ====================
    writer = None
    if USE_TENSORBOARD:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f'runs/HierFL_{"DP" if USE_DP else "noDP"}_{timestamp}'
        writer = SummaryWriter(log_dir)
        print(f"\n{'='*70}")
        print(f"TensorBoard 已启用")
        print(f"  日志目录: {log_dir}")
        print(f"  启动命令: tensorboard --logdir=runs")
        print(f"  访问地址: http://localhost:6006")
        print(f"{'='*70}")

    # ==================== 加载数据 ====================
    print(f"\n[1/6] 加载MNIST数据集...(训练集使用比例: {TRAIN_FRACTION * 100}%)")
    train_dataset, test_dataset = load_mnist(train_fraction=TRAIN_FRACTION)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ==================== 分配数据给客户端 ====================
    print("[2/6] 将数据分配给客户端...")
    client_data_indices = split_data_to_clients(train_dataset, NUM_CLIENTS, NUM_EDGES,
                                                 client_iid=CLIENT_IID, edge_iid=EDGE_IID)

    # 根据是否使用MAML决定如何创建数据加载器
    if USE_MAML:
        print(f"      使用MAML模式，划分support/query集 ({SUPPORT_RATIO:.0%}/{1-SUPPORT_RATIO:.0%})")
        client_support_loaders, client_query_loaders = create_data_loaders(
            train_dataset, client_data_indices, BATCH_SIZE, support_ratio=SUPPORT_RATIO
        )
        client_loaders = None  # 标准模式不使用
    else:
        print("      使用标准训练模式")
        client_loaders = create_data_loaders(train_dataset, client_data_indices, BATCH_SIZE)
        client_support_loaders = None
        client_query_loaders = None

    # ==================== 初始化差分隐私配置 ====================
    dp_config = None
    if USE_DP:
        dp_config = DPConfig(
            epsilon=DP_EPSILON,
            delta=DP_DELTA,
            clip_C=DP_CLIP_C,
            rate=DP_RATE,
            mechanism=DP_MECHANISM
        )

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
        if USE_MAML:
            # MAML模式：传入support和query加载器
            client = Client(
                client_id,
                data_loader=None,  # 标准模式不使用
                device=DEVICE,
                support_loader=client_support_loaders[client_id],
                query_loader=client_query_loaders[client_id],
                train_inner_step=TRAIN_INNER_STEP,
                test_inner_step=TEST_INNER_STEP
            )
        else:
            # 标准模式
            client = Client(client_id, client_loaders[client_id], device=DEVICE)
        clients.append(client)

        # 将客户端注册到对应的边缘服务器
        edge_id = client_id // NUM_CLIENTS_PER_EDGE
        edge_servers[edge_id].register_client(client)

    # ==================== 开始训练 ====================
    print("[4/6] 开始分层联邦学习训练...\n")

    accuracy_history = []
    loss_history = []

    # 云轮次计数器
    cloud_round = 0

    for round_idx in range(NUM_ROUNDS):
        print(f"{'='*70}")
        print(f"轮次 {round_idx + 1}/{NUM_ROUNDS}")
        print(f"{'='*70}")

        # 步骤1: 云服务器分发模型给边缘服务器（每 NUM_EDGE_AGGREGATION 轮执行一次）
        if round_idx % NUM_EDGE_AGGREGATION == 0:
            cloud_server.distribute_model_to_edges()
            if NUM_EDGE_AGGREGATION > 1:
                print(f"云服务器轮次 {cloud_round + 1}, 开始边缘聚合阶段 ({NUM_EDGE_AGGREGATION}轮)")

        edge_models = {}
        round_client_losses = []  # 记录本轮所有客户端的损失

        # 对每个边缘服务器
        for edge_server in edge_servers:
            print(f"\n边缘服务器 {edge_server.edge_id}:")

            # 步骤2: 边缘服务器分发模型给客户端
            edge_server.distribute_model_to_clients()

            # 客户端采样：根据 FRAC 比例随机选择参与训练的客户端
            all_client_ids = edge_server.client_ids
            num_selected = max(1, int(len(all_client_ids) * FRAC))
            selected_client_ids = np.random.choice(
                all_client_ids,
                num_selected,
                replace=False
            ).tolist()

            if FRAC < 1.0:
                print(f"  客户端采样: {num_selected}/{len(all_client_ids)} 个客户端参与训练")

            # 步骤3: 客户端本地训练
            client_models = {}
            for client_id in selected_client_ids:  # 只训练被选中的客户端
                client = clients[client_id]

                # 设置教师模型（用于KL蒸馏，从第2轮开始）
                if USE_DISTILLATION and round_idx > 0:
                    client.set_teacher_model(cloud_server.model.state_dict())

                # 训练
                train_loss = client.train(
                    LOCAL_EPOCHS, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY,
                    LR_DECAY, LR_DECAY_EPOCH,
                    use_dp=USE_DP, dp_config=dp_config,
                    use_maml=USE_MAML, beta=BETA,
                    use_distillation=USE_DISTILLATION,
                    temperature=DISTILL_TEMPERATURE,
                    alpha=DISTILL_ALPHA,
                    beta_feat=DISTILL_BETA,
                    use_dp_distillation=USE_DP_DISTILLATION
                )

                # 获取模型参数（如果使用DP，返回处理后的梯度）
                client_models[client_id] = client.get_model_parameters(
                    use_dp=USE_DP, dp_config=dp_config
                )

                print(f"  客户端 {client_id} 训练完成, 损失: {train_loss:.4f}")
                round_client_losses.append(train_loss)

                # TensorBoard: 记录每个客户端的训练损失
                if writer:
                    writer.add_scalar(f'Client/Loss_Client_{client_id}', train_loss, round_idx)

            # 步骤4: 边缘服务器聚合客户端模型
            edge_server.aggregate_client_models(client_models, use_dp=USE_DP)
            edge_models[edge_server.edge_id] = edge_server.get_model_parameters()
            print(f"  边缘服务器 {edge_server.edge_id} 聚合完成")

        # 步骤5: 每 NUM_EDGE_AGGREGATION 轮执行一次云聚合
        if (round_idx + 1) % NUM_EDGE_AGGREGATION == 0:
            # 云服务器聚合边缘服务器模型
            cloud_server.aggregate_edge_models(edge_models)
            print(f"\n云服务器聚合完成 (边缘聚合轮次: {(round_idx % NUM_EDGE_AGGREGATION) + 1}/{NUM_EDGE_AGGREGATION})")
            cloud_round += 1

        # 步骤5: 云服务器聚合边缘服务器模型
        cloud_server.aggregate_edge_models(edge_models)
        print(f"\n云服务器聚合完成")

        # 计算本轮平均客户端损失
        avg_client_loss = sum(round_client_losses) / len(round_client_losses)

        # 步骤6: 评估全局模型
        accuracy, test_loss = cloud_server.evaluate(test_loader)
        accuracy_history.append(accuracy)
        loss_history.append(test_loss)

        # TensorBoard: 记录全局指标
        if writer:
            writer.add_scalar('Global/Test_Accuracy', accuracy, round_idx)
            writer.add_scalar('Global/Test_Loss', test_loss, round_idx)
            writer.add_scalar('Global/Avg_Client_Train_Loss', avg_client_loss, round_idx)
            writer.add_scalar('Hyperparameters/Learning_Rate', LEARNING_RATE * (LR_DECAY ** round_idx), round_idx)

        print(f"\n轮次 {round_idx + 1} 结果:")
        print(f"  测试准确率: {accuracy:.2f}%")
        print(f"  测试损失: {test_loss:.4f}")
        print(f"  平均客户端训练损失: {avg_client_loss:.4f}")

    # ==================== 训练完成 ====================
    print(f"\n{'='*70}")
    print("训练完成!")
    print(f"{'='*70}")
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
    title_suffix = ' (with DP)' if USE_DP else ' (without DP)'
    plt.title('测试准确率变化' + title_suffix, fontsize=14)
    plt.grid(True, alpha=0.3)

    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_ROUNDS + 1), loss_history, 'r-o', linewidth=2, markersize=6)
    plt.xlabel('轮次 (Round)', fontsize=12)
    plt.ylabel('损失 (Loss)', fontsize=12)
    plt.title('测试损失变化' + title_suffix, fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'training_results_{"with_dp" if USE_DP else "without_dp"}.png'
    plt.savefig(filename, dpi=150)
    print(f"训练曲线已保存到 {filename}")

    print("\n[6/6] 完成!")


if __name__ == "__main__":
    main()
