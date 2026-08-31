import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Subset

class SubsetWithClasses(Subset):
    """带有classes属性的Subset，保持与原始数据集的兼容性"""

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        # 继承原始数据集的classes属性
        self.classes = dataset.classes if hasattr(dataset, 'classes') else list(range(10))


def load_mnist(data_dir='./data', train_fraction=1.0):
    def stratified_split(dataset, fraction, seed=42):
        """分层采样，保持各类别比例不变"""
        # 按标签分组
        label_to_indices = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)

        # 每个类别按比例采样
        selected_indices = []
        generator = torch.Generator().manual_seed(seed)

        for label, indices in label_to_indices.items():
            num_samples = int(len(indices) * fraction)
            # 随机打乱后取前 num_samples 个
            perm = torch.randperm(len(indices), generator=generator)
            selected_indices.extend([indices[i] for i in perm[:num_samples]])

        return SubsetWithClasses(dataset, selected_indices)

    """加载MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    if train_fraction < 1.0:
        train_dataset = stratified_split(train_dataset, train_fraction)

    return train_dataset, test_dataset


def split_data_to_clients(train_dataset, num_clients, num_edges, client_iid=True, edge_iid=True):
    """
    将训练数据分配给客户端，支持分别控制客户端和边缘服务器的数据分布
    确保最大化使用数据集且无重复分配

    Args:
        train_dataset: 完整的训练数据集
        num_clients: 客户端数量
        num_edges: 边缘服务器数量
        client_iid: 客户端数据是否IID分布
        edge_iid: 边缘服务器数据是否IID分布

    Returns:
        client_data_indices: 字典，key为客户端ID，value为该客户端的数据索引列表
    """
    num_samples = len(train_dataset)
    num_classes = len(train_dataset.classes) # MNIST有10个类别
    clients_per_edge = num_clients // num_edges

    # 获取所有样本的标签
    labels = np.array([train_dataset[i][1] for i in range(num_samples)])

    # 按类别分组所有数据索引
    label_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}

    client_data_indices = {}

    # 情况1: 客户端IID + 边缘IID
    if client_iid and edge_iid:
        # 所有数据随机打乱后均匀分配
        indices = np.random.permutation(num_samples)
        samples_per_client = num_samples // num_clients

        for client_id in range(num_clients):
            start = client_id * samples_per_client
            end = start + samples_per_client
            client_data_indices[client_id] = indices[start:end].tolist()

    # 情况2: 客户端Non-IID + 边缘IID
    elif not client_iid and edge_iid:
        labels_per_client = 2  # 每个客户端拥有的类别数

        for edge_id in range(num_edges):
            edge_client_start = edge_id * clients_per_edge

            # 为该边缘服务器准备每个类别的数据池（打乱后顺序分配，确保无重复）
            edge_label_pools = {}
            for class_id in range(num_classes):
                label_idx = label_indices[class_id]
                # 计算该边缘服务器分配到的该类别样本数
                samples_per_label_per_edge = len(label_idx) // num_edges
                edge_start = edge_id * samples_per_label_per_edge
                edge_end = edge_start + samples_per_label_per_edge
                edge_class_indices = label_idx[edge_start:edge_end].copy()
                # 打乱顺序
                np.random.shuffle(edge_class_indices)
                edge_label_pools[class_id] = edge_class_indices

            # 记录每个类别已经分配到的位置
            label_offset = {class_id: 0 for class_id in range(num_classes)}

            for local_client_idx in range(clients_per_edge):
                client_id = edge_client_start + local_client_idx

                # 为该客户端选择2个类别（循环分配，确保所有类别都被覆盖）
                primary_labels = [(local_client_idx * labels_per_client + i) % num_classes
                                 for i in range(labels_per_client)]

                client_indices = []
                for label in primary_labels:
                    edge_class_pool = edge_label_pools[label]
                    # 计算该客户端从这个类别分配多少样本
                    samples_per_client_per_label = len(edge_class_pool) // labels_per_client

                    # 切片分配（确保无重复）
                    start_idx = label_offset[label]
                    end_idx = min(start_idx + samples_per_client_per_label, len(edge_class_pool))
                    selected = edge_class_pool[start_idx:end_idx]

                    # 更新偏移量
                    label_offset[label] = end_idx

                    client_indices.extend(selected.tolist())

                client_data_indices[client_id] = client_indices


    # 情况3: 客户端IID + 边缘Non-IID
    elif client_iid and not edge_iid:
        def niid_split_to_edges(label_indices, num_edges, alpha=0.5):
            """
            第一层：用 Dirichlet 分布将数据 Non-IID 地分配给各边缘节点
            alpha 越小，边缘之间的数据分布差异越大
            """
            edge_indices = [[] for _ in range(num_edges)]

            for label in range(num_classes):
                indices = label_indices[label].copy()
                np.random.shuffle(indices)

                # Dirichlet 分布生成各边缘的分配比例
                proportions = np.random.dirichlet([alpha] * num_edges)
                split_points = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
                splits = np.split(indices, split_points)

                for edge_id in range(num_edges):
                    edge_indices[edge_id].extend(splits[edge_id])

            return edge_indices

        def iid_split_within_edge(edge_data_indices, clients_per_edge):
            """
            第二层：在单个边缘节点内部，IID 地均匀分配给各客户端
            """
            indices = np.array(edge_data_indices)
            np.random.shuffle(indices)

            # 使用 array_split 自动处理余数
            splits = np.array_split(indices, clients_per_edge)
            return [split.tolist() for split in splits]

        # 第一步：Non-IID 划分给边缘
        edge_indices = niid_split_to_edges(label_indices, num_edges)

        # 第二步：每个边缘内部 IID 划分给客户端
        for edge_id in range(num_edges):
            client_indices = iid_split_within_edge(edge_indices[edge_id], clients_per_edge)
            for i in range(clients_per_edge):
                client_data_indices[edge_id * clients_per_edge + i] = client_indices[i]

    # 情况4: 客户端Non-IID + 边缘Non-IID
    else:  # not client_iid and not edge_iid
        labels_per_client = 2  # 每个客户端拥有的类别数
        classes_per_edge = max(1, num_classes // num_edges)  # 每个边缘服务器的类别数

        for edge_id in range(num_edges):
            edge_client_start = edge_id * clients_per_edge

            # 为这个边缘服务器分配特定类别（Non-IID at edge level）
            edge_class_start = (edge_id * classes_per_edge) % num_classes
            edge_classes = [(edge_class_start + i) % num_classes for i in range(classes_per_edge)]

            # 收集该边缘服务器的所有类别数据
            edge_label_pools = {}
            for class_id in edge_classes:
                label_idx = label_indices[class_id].copy()
                np.random.shuffle(label_idx)
                edge_label_pools[class_id] = label_idx

            # 为客户端分配数据（Non-IID at client level）
            label_offset = {class_id: 0 for class_id in edge_classes}

            for local_client_idx in range(clients_per_edge):
                client_id = edge_client_start + local_client_idx

                # 从边缘服务器的类别中选择2个给客户端
                available_classes = list(edge_label_pools.keys())
                client_classes = [available_classes[(local_client_idx * labels_per_client + i) % len(available_classes)]
                                 for i in range(labels_per_client)]

                client_indices = []
                for class_id in client_classes:
                    class_pool = edge_label_pools[class_id]
                    samples_per_client_per_label = len(class_pool) // labels_per_client

                    start_idx = label_offset[class_id]
                    end_idx = min(start_idx + samples_per_client_per_label, len(class_pool))
                    selected = class_pool[start_idx:end_idx]
                    label_offset[class_id] = end_idx

                    client_indices.extend(selected.tolist())

                client_data_indices[client_id] = client_indices

    return client_data_indices


def create_data_loaders(train_dataset, client_data_indices, batch_size, support_ratio=1.0):
    """
    为每个客户端创建DataLoader（支持support/query划分用于MAML元学习）

    Args:
        train_dataset: 完整的训练数据集
        client_data_indices: 客户端数据索引字典
        batch_size: 批次大小
        support_ratio: support集占比（1.0表示不划分，使用完整数据；<1.0时划分为support和query）

    Returns:
        如果support_ratio=1.0: 返回client_loaders字典
        如果support_ratio<1.0: 返回(client_support_loaders, client_query_loaders)元组
    """
    if support_ratio >= 1.0:
        # 不划分，返回完整数据加载器（保持原有行为）
        client_loaders = {}
        for client_id, indices in client_data_indices.items():
            subset = torch.utils.data.Subset(train_dataset, indices)
            loader = torch.utils.data.DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )
            client_loaders[client_id] = loader
        return client_loaders
    else:
        # 划分为support和query集（用于MAML）
        client_support_loaders = {}
        client_query_loaders = {}

        for client_id, indices in client_data_indices.items():
            # 随机打乱索引
            indices = np.array(indices)
            np.random.shuffle(indices)

            # 划分support和query
            split_point = int(len(indices) * support_ratio)
            support_indices = indices[:split_point].tolist()
            query_indices = indices[split_point:].tolist()

            # 创建support loader
            support_subset = torch.utils.data.Subset(train_dataset, support_indices)
            support_loader = torch.utils.data.DataLoader(
                support_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )
            client_support_loaders[client_id] = support_loader

            # 创建query loader
            query_subset = torch.utils.data.Subset(train_dataset, query_indices)
            query_loader = torch.utils.data.DataLoader(
                query_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )
            client_query_loaders[client_id] = query_loader

        return client_support_loaders, client_query_loaders


def analyze_data_distribution(train_dataset, client_data_indices, num_edges, num_clients,description):
    """
    分析并可视化数据在边缘服务器和客户端之间的分布情况

    Args:
        train_dataset: 训练数据集
        client_data_indices: 客户端数据索引字典
        num_edges: 边缘服务器数量
        num_clients: 客户端总数
    """
    import matplotlib.pyplot as plt

    num_classes = len(train_dataset.classes)
    clients_per_edge = num_clients // num_edges

    # 获取所有样本的标签
    labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])

    # 统计每个客户端的类别分布
    client_class_counts = {}
    for client_id, indices in client_data_indices.items():
        client_labels = labels[indices]
        class_counts = np.bincount(client_labels, minlength=num_classes)
        client_class_counts[client_id] = class_counts

    # 统计每个边缘服务器的类别分布
    edge_class_counts = {}
    for edge_id in range(num_edges):
        edge_counts = np.zeros(num_classes, dtype=int)
        for local_client_idx in range(clients_per_edge):
            client_id = edge_id * clients_per_edge + local_client_idx
            edge_counts += client_class_counts[client_id]
        edge_class_counts[edge_id] = edge_counts

    # 计算统计信息
    total_samples = sum(len(indices) for indices in client_data_indices.values())
    samples_per_client = [len(indices) for indices in client_data_indices.values()]

    print("="*70)
    print("数据分布分析报告")
    print("="*70)
    print(f"总样本数: {len(train_dataset)}")
    print(f"已分配样本数: {total_samples}")
    print(f"数据利用率: {total_samples/len(train_dataset)*100:.2f}%")
    print(f"边缘服务器数量: {num_edges}")
    print(f"客户端数量: {num_clients}")
    print(f"每个边缘服务器的客户端数: {clients_per_edge}")
    print(f"每个客户端平均样本数: {np.mean(samples_per_client):.1f}")
    print(f"每个客户端样本数范围: [{min(samples_per_client)}, {max(samples_per_client)}]")
    print()

    # 打印每个边缘服务器的详细信息
    for edge_id in range(num_edges):
        print(f"{'='*70}")
        print(f"边缘服务器 {edge_id}")
        print(f"{'='*70}")
        edge_total = edge_class_counts[edge_id].sum()
        print(f"总样本数: {edge_total}")
        print(f"类别分布: {edge_class_counts[edge_id]}")
        print(f"类别比例: ", end="")
        for class_id in range(num_classes):
            ratio = edge_class_counts[edge_id][class_id] / edge_total * 100 if edge_total > 0 else 0
            print(f"{class_id}:{ratio:.1f}% ", end="")
        print("\n")

        # 打印该边缘服务器下每个客户端的信息
        for local_client_idx in range(clients_per_edge):
            client_id = edge_id * clients_per_edge + local_client_idx
            client_total = client_class_counts[client_id].sum()
            print(f"  客户端 {client_id}: {client_total}个样本")
            print(f"    类别分布: {client_class_counts[client_id]}")
            print(f"    主要类别: ", end="")
            # 找出样本数最多的类别
            top_classes = np.argsort(client_class_counts[client_id])[::-1][:3]
            for class_id in top_classes:
                if client_class_counts[client_id][class_id] > 0:
                    ratio = client_class_counts[client_id][class_id] / client_total * 100
                    print(f"{class_id}({ratio:.1f}%) ", end="")
            print()
        print()

    # 可视化
    fig = plt.figure(figsize=(20, 12))

    # 1. 边缘服务器的类别分布（堆叠柱状图）
    ax1 = plt.subplot(2, 3, 1)
    edge_data = np.array([edge_class_counts[i] for i in range(num_edges)])
    x = np.arange(num_edges)
    bottom = np.zeros(num_edges)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for class_id in range(num_classes):
        ax1.bar(x, edge_data[:, class_id], bottom=bottom, label=f'类别 {class_id}', color=colors[class_id])
        bottom += edge_data[:, class_id]

    ax1.set_xlabel('边缘服务器 ID', fontsize=12)
    ax1.set_ylabel('样本数量', fontsize=12)
    ax1.set_title('边缘服务器的类别分布（堆叠图）', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xticks(x)
    ax1.grid(axis='y', alpha=0.3)

    # 2. 边缘服务器的类别分布（热力图）
    ax2 = plt.subplot(2, 3, 2)
    im = ax2.imshow(edge_data.T, aspect='auto', cmap='YlOrRd')
    ax2.set_xlabel('边缘服务器 ID', fontsize=12)
    ax2.set_ylabel('类别', fontsize=12)
    ax2.set_title('边缘服务器类别分布热力图', fontsize=14, fontweight='bold')
    ax2.set_xticks(np.arange(num_edges))
    ax2.set_yticks(np.arange(num_classes))
    plt.colorbar(im, ax=ax2, label='样本数量')

    # 在热力图上添加数值
    for i in range(num_edges):
        for j in range(num_classes):
            text = ax2.text(i, j, int(edge_data[i, j]),
                           ha="center", va="center", color="black", fontsize=8)

    # 3. 所有客户端的样本数量
    ax3 = plt.subplot(2, 3, 3)
    client_ids = list(client_data_indices.keys())
    client_samples = [len(client_data_indices[cid]) for cid in client_ids]
    colors_clients = [plt.cm.Set3(i % num_edges / num_edges) for i in client_ids]
    ax3.bar(client_ids, client_samples, color=colors_clients)
    ax3.set_xlabel('客户端 ID', fontsize=12)
    ax3.set_ylabel('样本数量', fontsize=12)
    ax3.set_title('每个客户端的样本数量', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # 4. 客户端类别分布热力图
    ax4 = plt.subplot(2, 3, 4)
    client_data = np.array([client_class_counts[i] for i in range(num_clients)])
    im2 = ax4.imshow(client_data.T, aspect='auto', cmap='viridis')
    ax4.set_xlabel('客户端 ID', fontsize=12)
    ax4.set_ylabel('类别', fontsize=12)
    ax4.set_title('客户端类别分布热力图', fontsize=14, fontweight='bold')
    ax4.set_yticks(np.arange(num_classes))
    plt.colorbar(im2, ax=ax4, label='样本数量')

    # 添加边缘服务器分隔线
    for edge_id in range(1, num_edges):
        ax4.axvline(x=edge_id * clients_per_edge - 0.5, color='red', linewidth=2, linestyle='--')

    # 5. 累积直方图 - 边缘服务器
    ax5 = plt.subplot(2, 3, 5)
    for edge_id in range(num_edges):
        cumsum = np.cumsum(edge_class_counts[edge_id])
        ax5.plot(range(num_classes), cumsum, marker='o', label=f'边缘 {edge_id}', linewidth=2)
    ax5.set_xlabel('类别', fontsize=12)
    ax5.set_ylabel('累积样本数量', fontsize=12)
    ax5.set_title('边缘服务器累积类别分布', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks(range(num_classes))

    # 6. 类别分布的统计图（箱线图）
    ax6 = plt.subplot(2, 3, 6)
    class_distributions = []
    for class_id in range(num_classes):
        class_dist = [client_class_counts[cid][class_id] for cid in range(num_clients)]
        class_distributions.append(class_dist)

    bp = ax6.boxplot(class_distributions, labels=range(num_classes), patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax6.set_xlabel('类别', fontsize=12)
    ax6.set_ylabel('样本数量', fontsize=12)
    ax6.set_title('各类别在客户端中的分布（箱线图）', fontsize=14, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    # plt.savefig(f'data_distribution_analysis_{description}.png', dpi=150, bbox_inches='tight')
    # print(f"{'='*70}")
    # print(f"可视化图表已保存到: data_distribution_analysis_{description}.png")
    # print(f"{'='*70}")
    plt.show()


if __name__ == "__main__":
    print("数据分布测试程序\n")

    # 配置参数
    NUM_EDGES = 5
    NUM_CLIENTS_PER_EDGE = 10
    NUM_CLIENTS = NUM_EDGES * NUM_CLIENTS_PER_EDGE

    # 加载数据
    print("加载MNIST数据集...")
    train_dataset, _ = load_mnist()

    # 测试四种情况
    scenarios = [
        (True, True, "客户端IID + 边缘IID"),
        (False, True, "客户端Non-IID + 边缘IID"),
        (True, False, "客户端IID + 边缘Non-IID"),
        (False, False, "客户端Non-IID + 边缘Non-IID")
    ]

    for idx, (client_iid, edge_iid, description) in enumerate(scenarios):
        print(f"\n\n{'#'*70}")
        print(f"场景 {idx + 1}: {description}")
        print(f"{'#'*70}\n")

        # 分配数据
        client_data_indices = split_data_to_clients(
            train_dataset,
            NUM_CLIENTS,
            NUM_EDGES,
            client_iid=client_iid,
            edge_iid=edge_iid
        )

        # 分析和可视化
        analyze_data_distribution(train_dataset, client_data_indices, NUM_EDGES, NUM_CLIENTS,description)

