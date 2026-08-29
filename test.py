import torchvision
from collections import Counter

# 加载 MNIST 训练集
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=None
)

# 统计各类别样本数量
labels = [train_dataset[i][1] for i in range(len(train_dataset))]
counter = Counter(labels)

print("MNIST 训练集各类别样本数量：")
for class_id in range(10):
    print(f"  类别 {class_id}: {counter[class_id]} 个样本")
print(f"  总计: {len(train_dataset)} 个样本")