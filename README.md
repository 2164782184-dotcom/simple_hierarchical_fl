# 简单分层联邦学习 (Simple Hierarchical Federated Learning)

## 项目结构
```
simple_hierarchical_fl/
├── README.md           # 项目说明
├── requirements.txt    # 依赖包
├── main.py            # 主程序入口
├── models.py          # 神经网络模型定义
├── client.py          # 客户端实现
├── edge_server.py     # 边缘服务器实现
├── cloud_server.py    # 云服务器实现
└── data_utils.py      # 数据处理工具
```

## 架构说明
```
云服务器 (Cloud Server)
    ├── 边缘服务器1 (Edge Server 1)
    │   ├── 客户端1 (Client 1)
    │   ├── 客户端2 (Client 2)
    │   └── 客户端3 (Client 3)
    └── 边缘服务器2 (Edge Server 2)
        ├── 客户端4 (Client 4)
        ├── 客户端5 (Client 5)
        └── 客户端6 (Client 6)
```

## 训练流程
1. 云服务器将全局模型分发给各边缘服务器
2. 边缘服务器将模型分发给所属客户端
3. 客户端在本地数据上训练模型
4. 客户端将更新上传到边缘服务器
5. 边缘服务器聚合客户端更新
6. 边缘服务器将聚合结果上传到云服务器
7. 云服务器聚合所有边缘服务器的更新，更新全局模型
8. 重复步骤1-7

## 运行
```bash
pip install -r requirements.txt
python main.py
```

## 参数说明
- `num_edges`: 边缘服务器数量
- `num_clients_per_edge`: 每个边缘服务器下的客户端数量
- `num_rounds`: 全局训练轮数
- `local_epochs`: 客户端本地训练轮数
- `batch_size`: 批次大小
- `learning_rate`: 学习率
