import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime
import copy

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti SC', 'Hiragino Sans GB', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

from models import get_model
from data_utils import load_mnist, split_data_to_clients, create_data_loaders
from client import Client, DPConfig
from edge_server import EdgeServer
from cloud_server import CloudServer
from privacy_accountant import PrivacyAccountant
from dynamic_weights import DynamicWeightAdjuster


def main():
    # ==================== 配置参数 ====================
    NUM_EDGES = 5                         # 边缘服务器数量
    NUM_CLIENTS_PER_EDGE = 10             # 每个边缘服务器的客户端数量
    NUM_CLIENTS = NUM_EDGES * NUM_CLIENTS_PER_EDGE  # 总客户端数量
    NUM_ROUNDS = 100                      # 全局训练轮数
    NUM_EDGE_AGGREGATION = 1              # 边缘聚合次数（每次云聚合前，边缘进行多少轮本地聚合）
    LOCAL_EPOCHS = 60                     # 客户端本地训练轮数
    BATCH_SIZE = 64                       # 批次大小（GPU优化：64适合中等数据集）
    LEARNING_RATE = 0.001                 # 初始学习率
    LR_DECAY = 0.995                      # 学习率衰减系数（gamma）
    LR_DECAY_EPOCH = 1                    # 每1个epoch执行一次学习率衰减
    MOMENTUM = 0                          # SGD动量（0表示不使用动量）
    WEIGHT_DECAY = 0                      # 权重衰减/L2正则化（0表示不使用）
    TRAIN_FRACTION = 1                    # 训练集的使用比例（1.0=使用100%数据）
    CLIENT_IID = False                    # 客户端数据是否IID分布
    EDGE_IID = True                       # 边缘服务器数据是否IID分布
    FRAC = 1.0                            # 每轮参与训练的客户端比例（1.0=全部参与）
    SEED = 42                             # 随机种子（用于实验可复现性）
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ==================== 差分隐私配置 ====================
    USE_DP = True                         # 是否使用差分隐私
    DP_EPSILON = 5                        # 隐私预算（越小隐私保护越强，典型值：0.1-10）
    DP_DELTA = 0.001                      # 失败概率（典型值：1e-5 到 1e-7）
    DP_CLIP_C = 0.01                      # 梯度裁剪阈值
    DP_RATE = 4/3                         # 稀疏化率（rate=50 表示保留 2% 的梯度）
    DP_MECHANISM = 'laplace'              # 噪声机制（'laplace' 或 'gaussian'）

    # ==================== 知识蒸馏配置 ====================
    USE_DISTILLATION = True               # 是否使用知识蒸馏
    DISTILL_TEMPERATURE = 3.0             # 蒸馏温度参数
    DISTILL_ALPHA = 0.4                   # KL损失权重（软标签）初始值
    DISTILL_BETA = 0.2                    # MSE损失权重（特征层）初始值
    USE_DP_DISTILLATION = True            # 是否对蒸馏过程使用差分隐私（保护教师模型输出）
    DISTILL_START_THRESHOLD = 0.7         # 互蒸馏启动阈值（学生达到教师70%性能时才开始，0=立即开始）
    USE_DYNAMIC_WEIGHTS = True            # 是否动态调整KL和MSE权重（根据CE/KL/MSE损失大小自适应）
    # 总损失 = (1-α-β)*CE + α*KL + β*MSE

    # ==================== 隐私预算统计配置 ====================
    USE_PRIVACY_ACCOUNTANT = False        # 是否启用隐私预算统计（关闭以避免超标警告）

    # ==================== TensorBoard配置 ====================
    USE_TENSORBOARD = False                # 是否启用TensorBoard实时可视化

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
    print("分层联邦学习 + 差分隐私 + 教师-学生互蒸馏")
    print("="*70)
    print(f"设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU设备名称: {torch.cuda.get_device_name(0)}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"边缘服务器数量: {NUM_EDGES}")
    print(f"每个边缘服务器的客户端数量: {NUM_CLIENTS_PER_EDGE}")
    print(f"总客户端数量: {NUM_CLIENTS}")
    print(f"全局轮数: {NUM_ROUNDS}")
    print(f"本地训练轮数: {LOCAL_EPOCHS}")
    print(f"客户端数据分布: {'IID' if CLIENT_IID else 'Non-IID'}")
    print(f"边缘服务器数据分布: {'IID' if EDGE_IID else 'Non-IID'}")
    print(f"\n{'='*70}")
    print(f"\n{'='*70}")
    print(f"知识蒸馏配置:")
    print(f"  启用状态: {'是' if USE_DISTILLATION else '否'}")
    if USE_DISTILLATION:
        print(f"  温度参数: {DISTILL_TEMPERATURE}")
        print(f"  KL损失权重α: {DISTILL_ALPHA}")
        print(f"  MSE损失权重β: {DISTILL_BETA}")
        if DISTILL_START_THRESHOLD > 0:
            print(f"  互蒸馏启动: 学生达到教师{DISTILL_START_THRESHOLD:.0%}性能后启动")
        else:
            print(f"  互蒸馏启动: 关闭（第2轮立即开始）")
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

    # ==================== 初始化隐私统计器 ====================
    privacy_accountant = None
    if USE_DP and USE_PRIVACY_ACCOUNTANT:
        # 设置目标隐私预算为单轮的100倍（可根据需要调整）
        target_epsilon = DP_EPSILON * 100
        privacy_accountant = PrivacyAccountant(target_epsilon=target_epsilon, target_delta=DP_DELTA * NUM_ROUNDS)
        print(f"\n{'='*70}")
        print(f"隐私预算统计已启用")
        print(f"  目标总隐私预算 ε: {target_epsilon}")
        print(f"  单轮隐私预算 ε: {DP_EPSILON}")
        print(f"  预计可训练轮数: ~{target_epsilon / DP_EPSILON:.0f} 轮")
        print(f"{'='*70}")

    # ==================== 加载数据 ====================
    print(f"\n[1/6] 加载MNIST数据集...(训练集使用比例: {TRAIN_FRACTION * 100}%)")
    train_dataset, test_dataset = load_mnist(train_fraction=TRAIN_FRACTION)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ==================== 分配数据给客户端 ====================
    print("[2/6] 将数据分配给客户端...")
    client_data_indices = split_data_to_clients(train_dataset, NUM_CLIENTS, NUM_EDGES,
                                                 client_iid=CLIENT_IID, edge_iid=EDGE_IID)

    print("      使用标准训练模式")
    client_loaders = create_data_loaders(train_dataset, client_data_indices, BATCH_SIZE)

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
        client = Client(client_id, client_loaders[client_id], device=DEVICE)
        clients.append(client)

        # 将客户端注册到对应的边缘服务器
        edge_id = client_id // NUM_CLIENTS_PER_EDGE
        edge_servers[edge_id].register_client(client)

    # ==================== 开始训练 ====================
    print("[4/6] 开始分层联邦学习训练...\n")

    accuracy_history = []
    loss_history = []

    # 自适应蒸馏相关变量
    distillation_enabled = False      # 学生是否达标（可以让教师开始蒸馏学生）
    teacher_accuracy = 0.0            # 教师模型的准确率
    student_accuracy = 0.0            # 学生模型的准确率

    # 动态权重调整器（每个客户端一个）
    weight_adjusters = {}
    if USE_DYNAMIC_WEIGHTS and USE_DISTILLATION:
        for client_id in range(NUM_CLIENTS):
            weight_adjusters[client_id] = DynamicWeightAdjuster(
                initial_alpha=DISTILL_ALPHA,
                initial_beta=DISTILL_BETA
            )
        print(f"互蒸馏机制的动态权重调整已启用")

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
            if FRAC == 1.0:
                selected_client_ids = all_client_ids
            else:
                selected_client_ids = np.random.choice(
                    all_client_ids,
                    num_selected,
                    replace=False
                ).tolist()
                print(f"  客户端采样: {num_selected}/{len(all_client_ids)} 个客户端参与训练")

            # 步骤3: 客户端本地训练（双向互蒸馏）
            client_models = {}
            teacher_losses = []
            student_losses = []

            for client_id in selected_client_ids:
                client = clients[client_id]

                # ========== 阶段1: 教师训练 ==========
                # 教师总是只用CE训练（不蒸馏学生）
                teacher_train_loss = client.train(
                    LOCAL_EPOCHS, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY,
                    LR_DECAY, LR_DECAY_EPOCH,
                    use_dp=USE_DP, dp_config=dp_config,
                    use_distillation=False,  # 教师第一阶段只用CE
                    temperature=DISTILL_TEMPERATURE,
                    alpha=DISTILL_ALPHA,
                    beta_feat=DISTILL_BETA,
                    use_dp_distillation=USE_DP_DISTILLATION,
                    weight_adjuster=None
                )
                teacher_losses.append(teacher_train_loss)

                # 保存教师模型参数
                # 1. 带DP的版本用于传递给学生（隐私保护）
                teacher_params_with_dp = client.get_model_parameters(use_dp=USE_DP, dp_config=dp_config)

                # 2. 不带DP的state_dict用于本地蒸馏目标（仅当不使用DP时）
                if not USE_DP:
                    teacher_state_dict = teacher_params_with_dp  # 不使用DP时，两者相同
                else:
                    # 使用DP时，保存当前模型的state_dict用于后续互蒸馏
                    teacher_state_dict = copy.deepcopy(client.model.state_dict())

                # ========== 阶段2: 学生蒸馏教师（使用CE+KL+MSE） ==========
                # 将教师参数加载为学生的蒸馏目标
                # 注意：这里需要先恢复教师的带DP参数到模型，再设为蒸馏目标
                if USE_DP:
                    # 使用DP时，需要从带DP的梯度重建参数
                    # 但set_teacher_model需要state_dict，所以我们直接用teacher_state_dict
                    client.set_teacher_model(teacher_state_dict)
                else:
                    client.set_teacher_model(teacher_params_with_dp)

                # 学生训练（蒸馏教师）
                student_train_loss = client.train(
                    LOCAL_EPOCHS, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY,
                    LR_DECAY, LR_DECAY_EPOCH,
                    use_dp=USE_DP, dp_config=dp_config,
                    use_distillation=USE_DISTILLATION,  # 学生总是蒸馏教师
                    temperature=DISTILL_TEMPERATURE,
                    alpha=DISTILL_ALPHA,
                    beta_feat=DISTILL_BETA,
                    use_dp_distillation=USE_DP_DISTILLATION,
                    weight_adjuster=weight_adjusters.get(client_id) if USE_DYNAMIC_WEIGHTS else None
                )
                student_losses.append(student_train_loss)

                # 获取学生模型参数（用于上传，可能带DP）
                student_params_for_upload = client.get_model_parameters(use_dp=USE_DP, dp_config=dp_config)

                # 保存学生模型的state_dict（不加DP，用于蒸馏目标）
                student_state_dict = copy.deepcopy(client.model.state_dict())

                # ========== 阶段3: 如果互蒸馏已启动，教师蒸馏学生 ==========
                if distillation_enabled:
                    # 互蒸馏已启动：教师再次训练，这次蒸馏学生
                    # 将学生参数设为教师的蒸馏目标
                    client.model.load_state_dict(teacher_state_dict)  # 恢复教师参数
                    client.set_teacher_model(student_state_dict)  # 学生作为教师的蒸馏目标

                    mutual_teacher_loss = client.train(
                        LOCAL_EPOCHS, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY,
                        LR_DECAY, LR_DECAY_EPOCH,
                        use_dp=USE_DP, dp_config=dp_config,
                        use_distillation=True,  # 教师蒸馏学生
                        temperature=DISTILL_TEMPERATURE,
                        alpha=DISTILL_ALPHA,
                        beta_feat=DISTILL_BETA,
                        use_dp_distillation=USE_DP_DISTILLATION,
                        weight_adjuster=weight_adjusters.get(client_id) if USE_DYNAMIC_WEIGHTS else None
                    )

                    # 注意：无论教师是否蒸馏学生，上传的始终是学生参数
                    client_models[client_id] = student_params_for_upload
                    print(f"  客户端 {client_id} 训练完成 [互蒸馏] - 教师损失: {teacher_train_loss:.4f}, 学生损失: {student_train_loss:.4f}, 互蒸馏教师损失: {mutual_teacher_loss:.4f}")
                else:
                    # 未达标，只上传学生参数
                    client_models[client_id] = student_params_for_upload
                    print(f"  客户端 {client_id} 训练完成 [单向S→T] - 教师损失: {teacher_train_loss:.4f}, 学生损失: {student_train_loss:.4f}")

                round_client_losses.append(student_train_loss)  # 记录学生损失

                # TensorBoard: 记录每个客户端的训练损失
                if writer:
                    writer.add_scalar(f'Client/Loss_Client_{client_id}', student_train_loss, round_idx)

            # 步骤4: 边缘服务器聚合客户端模型
            edge_server.aggregate_client_models(client_models, use_dp=USE_DP)
            edge_models[edge_server.edge_id] = edge_server.get_model_parameters()
            print(f"  边缘服务器 {edge_server.edge_id} 聚合完成")

        # 步骤5: 云服务器聚合边缘服务器模型
        cloud_server.aggregate_edge_models(edge_models)
        print(f"\n云服务器聚合完成")

        # 计算本轮平均客户端损失
        avg_client_loss = sum(round_client_losses) / len(round_client_losses)

        # 步骤6: 评估全局模型
        accuracy, test_loss = cloud_server.evaluate(test_loader)
        accuracy_history.append(accuracy)
        loss_history.append(test_loss)

        # 评估教师模型和学生模型的性能
        # 注意：这里的accuracy是学生模型的准确率（最终上传的模型）
        student_accuracy = accuracy

        # 教师模型的准确率（第一轮后记录）
        if round_idx == 0:
            teacher_accuracy = accuracy  # 第一轮作为基准
        else:
            # 检查学生是否达到教师的70%
            if not distillation_enabled and DISTILL_START_THRESHOLD > 0:
                if student_accuracy >= teacher_accuracy * DISTILL_START_THRESHOLD:
                    distillation_enabled = True
                    print(f"\n🎓 互蒸馏已启动！学生准确率 {student_accuracy:.2f}% ≥ {DISTILL_START_THRESHOLD:.0%} × {teacher_accuracy:.2f}% (教师准确率)")

            # 更新教师准确率为当前学生准确率（学生成为下一轮的教师）
            teacher_accuracy = student_accuracy

        # TensorBoard: 记录全局指标
        if writer:
            writer.add_scalar('Global/Test_Accuracy', accuracy, round_idx)
            writer.add_scalar('Global/Test_Loss', test_loss, round_idx)
            writer.add_scalar('Global/Avg_Client_Train_Loss', avg_client_loss, round_idx)
            writer.add_scalar('Hyperparameters/Learning_Rate', LEARNING_RATE * (LR_DECAY ** round_idx), round_idx)
            writer.add_scalar('Distillation/Enabled', int(distillation_enabled), round_idx)

        print(f"\n轮次 {round_idx + 1} 结果:")
        print(f"  测试准确率: {accuracy:.2f}%")
        print(f"  测试损失: {test_loss:.4f}")
        print(f"  平均客户端训练损失: {avg_client_loss:.4f}")
        if USE_DISTILLATION and DISTILL_START_THRESHOLD > 0:
            if distillation_enabled:
                print(f"  互蒸馏状态: ✓ 已启动（教师和学生互相蒸馏）")
            else:
                target = teacher_accuracy * DISTILL_START_THRESHOLD
                print(f"  互蒸馏状态: 等待中 (当前: {student_accuracy:.2f}%, 需达到: {target:.2f}%)")

        # 记录隐私消耗
        if privacy_accountant is not None:
            privacy_accountant.add_round(
                round_num=round_idx + 1,
                epsilon=DP_EPSILON,
                delta=DP_DELTA,
                num_clients=int(NUM_CLIENTS * FRAC)
            )

            # 每10轮或最后一轮打印隐私报告
            if (round_idx + 1) % 10 == 0 or round_idx == NUM_ROUNDS - 1:
                print(f"\n{privacy_accountant.get_privacy_report()}")

            # 检查隐私预算
            is_safe, warning = privacy_accountant.check_privacy_budget()
            if not is_safe:
                print(f"\n{warning}")
                print("⚠️  隐私预算耗尽，提前终止训练！")
                break

    # ==================== 训练完成 ====================
    print(f"\n{'='*70}")
    print("训练完成!")
    print(f"{'='*70}")
    print(f"最终测试准确率: {accuracy_history[-1]:.2f}%")
    print(f"最终测试损失: {loss_history[-1]:.4f}")
    if USE_DISTILLATION and DISTILL_START_THRESHOLD > 0:
        print(f"蒸馏最终状态: {'已启动' if distillation_enabled else '未达到启动条件'}")

    # 保存隐私统计历史
    if privacy_accountant is not None:
        print(f"\n{privacy_accountant.get_privacy_report()}")
        privacy_accountant.save_history('privacy_history.json')
        print(f"\n隐私统计历史已保存到: privacy_history.json")

    # ==================== 绘制结果 ====================
    print("\n[5/6] 绘制训练曲线...")

    # 使用实际训练的轮数
    actual_rounds = len(accuracy_history)

    plt.figure(figsize=(12, 5))

    # 绘制准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, actual_rounds + 1), accuracy_history, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('轮次 (Round)', fontsize=12)
    plt.ylabel('准确率 (%)', fontsize=12)
    title_suffix = ' (with DP)' if USE_DP else ' (without DP)'
    plt.title('测试准确率变化' + title_suffix, fontsize=14)
    plt.grid(True, alpha=0.3)

    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, actual_rounds + 1), loss_history, 'r-o', linewidth=2, markersize=6)
    plt.xlabel('轮次 (Round)', fontsize=12)
    plt.ylabel('损失 (Loss)', fontsize=12)
    plt.title('测试损失变化' + title_suffix, fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'training_results_adaptive_distillation.png'
    plt.savefig(filename, dpi=150)
    print(f"训练曲线已保存到 {filename} (实际训练 {actual_rounds}/{NUM_ROUNDS} 轮)")

    print("\n[6/6] 完成!")


if __name__ == "__main__":
    main()
