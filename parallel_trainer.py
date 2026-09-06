"""
多GPU并行训练模块
==================
功能：
1. 自动检测GPU数量
2. 方案1：多进程并行（每GPU串行训练多个客户端）
3. 方案2：每GPU内部也并行（充分利用GPU并行能力）
"""

import torch
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import copy
from typing import List, Dict, Any, Tuple
import time


def get_gpu_count():
    """
    自动检测可用GPU数量

    Returns:
        int: GPU数量，如果没有GPU返回0
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def allocate_clients_to_gpus(client_ids: List[int], num_gpus: int) -> List[List[int]]:
    """
    将客户端均匀分配到各个GPU

    Args:
        client_ids: 客户端ID列表
        num_gpus: GPU数量

    Returns:
        List[List[int]]: 每个GPU负责的客户端列表
    """
    clients_per_gpu = [[] for _ in range(num_gpus)]

    for idx, client_id in enumerate(client_ids):
        gpu_id = idx % num_gpus
        clients_per_gpu[gpu_id].append(client_id)

    return clients_per_gpu


def train_single_client(client_id: int, client, global_model_state: Dict,
                       device: torch.device, dp_config, distillation_enabled: bool,
                       training_params: Dict) -> Dict[str, Any]:
    """
    训练单个客户端（教师-学生互蒸馏三阶段）

    Args:
        client_id: 客户端ID
        client: 客户端对象
        global_model_state: 全局模型参数
        device: 设备
        dp_config: 差分隐私配置
        distillation_enabled: 是否启用互蒸馏
        training_params: 训练参数字典

    Returns:
        Dict: 包含训练结果的字典
    """
    try:
        # 设置客户端设备
        client.device = device

        # 从models模块获取模型
        from models import get_model
        client.model = get_model()
        client.model.load_state_dict(global_model_state)
        client.model.to(device)

        # 保存初始参数
        client.initial_params = copy.deepcopy(client.model.state_dict())

        # ========== 阶段1: 教师训练 ==========
        teacher_loss = client.train(
            epochs=training_params['LOCAL_EPOCHS'],
            learning_rate=training_params['LEARNING_RATE'],
            momentum=training_params['MOMENTUM'],
            weight_decay=training_params['WEIGHT_DECAY'],
            lr_decay=training_params['LR_DECAY'],
            lr_decay_epoch=training_params['LR_DECAY_EPOCH'],
            use_dp=training_params['USE_DP'],
            dp_config=dp_config,
            use_distillation=False,
            temperature=training_params['DISTILL_TEMPERATURE'],
            alpha=training_params['DISTILL_ALPHA'],
            beta_feat=training_params['DISTILL_BETA'],
            use_dp_distillation=training_params['USE_DP_DISTILLATION'],
            weight_adjuster=None
        )

        # 保存教师参数
        teacher_state_dict = copy.deepcopy(client.model.state_dict())

        # ========== 阶段2: 学生训练 ==========
        client.set_teacher_model(teacher_state_dict)

        student_loss = client.train(
            epochs=training_params['LOCAL_EPOCHS'],
            learning_rate=training_params['LEARNING_RATE'],
            momentum=training_params['MOMENTUM'],
            weight_decay=training_params['WEIGHT_DECAY'],
            lr_decay=training_params['LR_DECAY'],
            lr_decay_epoch=training_params['LR_DECAY_EPOCH'],
            use_dp=training_params['USE_DP'],
            dp_config=dp_config,
            use_distillation=training_params['USE_DISTILLATION'],
            temperature=training_params['DISTILL_TEMPERATURE'],
            alpha=training_params['DISTILL_ALPHA'],
            beta_feat=training_params['DISTILL_BETA'],
            use_dp_distillation=training_params['USE_DP_DISTILLATION'],
            weight_adjuster=training_params.get('weight_adjuster', None)
        )

        # 保存学生参数
        student_state_dict = copy.deepcopy(client.model.state_dict())

        # 获取上传参数
        student_params_for_upload = client.get_model_parameters(
            use_dp=training_params['USE_DP'],
            dp_config=dp_config
        )

        mutual_teacher_loss = None

        # ========== 阶段3: 互蒸馏 ==========
        if distillation_enabled:
            client.model.load_state_dict(teacher_state_dict)
            client.set_teacher_model(student_state_dict)

            mutual_teacher_loss = client.train(
                epochs=training_params['LOCAL_EPOCHS'],
                learning_rate=training_params['LEARNING_RATE'],
                momentum=training_params['MOMENTUM'],
                weight_decay=training_params['WEIGHT_DECAY'],
                lr_decay=training_params['LR_DECAY'],
                lr_decay_epoch=training_params['LR_DECAY_EPOCH'],
                use_dp=training_params['USE_DP'],
                dp_config=dp_config,
                use_distillation=True,
                temperature=training_params['DISTILL_TEMPERATURE'],
                alpha=training_params['DISTILL_ALPHA'],
                beta_feat=training_params['DISTILL_BETA'],
                use_dp_distillation=training_params['USE_DP_DISTILLATION'],
                weight_adjuster=training_params.get('weight_adjuster', None)
            )

        # 将参数移回CPU以节省GPU内存
        student_params_cpu = {k: v.cpu() for k, v in student_params_for_upload.items()}

        return {
            'client_id': client_id,
            'params': student_params_cpu,
            'teacher_loss': teacher_loss,
            'student_loss': student_loss,
            'mutual_teacher_loss': mutual_teacher_loss,
            'success': True
        }

    except Exception as e:
        print(f"客户端 {client_id} 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'client_id': client_id,
            'success': False,
            'error': str(e)
        }
    finally:
        # 清理GPU内存
        if hasattr(client, 'model') and client.model is not None:
            del client.model
            client.model = None
        if hasattr(client, 'teacher_model') and client.teacher_model is not None:
            del client.teacher_model
            client.teacher_model = None
        torch.cuda.empty_cache()


def train_clients_on_gpu_sequential(gpu_id: int, client_ids: List[int],
                                   clients: List, global_model_state: Dict,
                                   dp_config, distillation_enabled: bool,
                                   training_params: Dict,
                                   return_dict: Dict) -> None:
    """
    方案1：在单个GPU上串行训练多个客户端

    Args:
        gpu_id: GPU编号
        client_ids: 该GPU负责的客户端ID列表
        clients: 所有客户端对象列表
        global_model_state: 全局模型参数
        dp_config: 差分隐私配置
        distillation_enabled: 是否启用互蒸馏
        training_params: 训练参数
        return_dict: 用于返回结果的共享字典
    """
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    print(f"[GPU {gpu_id}] 开始串行训练 {len(client_ids)} 个客户端...")

    results = []

    for client_id in client_ids:
        client = clients[client_id]
        result = train_single_client(
            client_id, client, global_model_state, device,
            dp_config, distillation_enabled, training_params
        )
        results.append(result)

    return_dict[gpu_id] = results
    print(f"[GPU {gpu_id}] 完成训练")


def train_clients_on_gpu_parallel(gpu_id: int, client_ids: List[int],
                                  clients: List, global_model_state: Dict,
                                  dp_config, distillation_enabled: bool,
                                  training_params: Dict,
                                  return_dict: Dict) -> None:
    """
    方案2：在单个GPU上并行训练多个客户端（GPU内部并行）

    Args:
        gpu_id: GPU编号
        client_ids: 该GPU负责的客户端ID列表
        clients: 所有客户端对象列表
        global_model_state: 全局模型参数
        dp_config: 差分隐私配置
        distillation_enabled: 是否启用互蒸馏
        training_params: 训练参数
        return_dict: 用于返回结果的共享字典
    """
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    print(f"[GPU {gpu_id}] 开始并行训练 {len(client_ids)} 个客户端...")

    # 使用线程池在GPU上并行训练
    with ThreadPoolExecutor(max_workers=len(client_ids)) as executor:
        futures = []
        for client_id in client_ids:
            client = clients[client_id]
            future = executor.submit(
                train_single_client,
                client_id, client, global_model_state, device,
                dp_config, distillation_enabled, training_params
            )
            futures.append(future)

        # 等待所有任务完成
        results = [future.result() for future in futures]

    return_dict[gpu_id] = results
    print(f"[GPU {gpu_id}] 完成训练")


class ParallelTrainer:
    """多GPU并行训练管理器"""

    def __init__(self, mode='auto'):
        """
        初始化并行训练器

        Args:
            mode: 'sequential' (方案1), 'parallel' (方案2), 'auto' (自动选择)
        """
        self.num_gpus = get_gpu_count()
        self.mode = mode

        if self.num_gpus == 0:
            print("警告: 未检测到GPU，将使用CPU训练")
        else:
            print(f"检测到 {self.num_gpus} 个GPU")

            # 自动选择模式
            if mode == 'auto':
                # 根据GPU显存自动选择
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
                if gpu_memory >= 16:  # 显存>=16GB使用方案2
                    self.mode = 'parallel'
                    print(f"GPU显存充足 ({gpu_memory:.1f}GB)，自动选择方案2（GPU内部并行）")
                else:
                    self.mode = 'sequential'
                    print(f"GPU显存适中 ({gpu_memory:.1f}GB)，自动选择方案1（串行训练）")

    def train_clients(self, selected_client_ids: List[int], clients: List,
                     global_model_state: Dict, dp_config,
                     distillation_enabled: bool, training_params: Dict) -> Dict[int, Dict]:
        """
        并行训练选中的客户端

        Args:
            selected_client_ids: 选中的客户端ID列表
            clients: 所有客户端对象
            global_model_state: 全局模型参数
            dp_config: 差分隐私配置
            distillation_enabled: 是否启用互蒸馏
            training_params: 训练参数字典

        Returns:
            Dict[int, Dict]: 客户端ID到训练结果的映射
        """
        if self.num_gpus == 0:
            # CPU训练（回退到串行）
            return self._train_on_cpu(selected_client_ids, clients, global_model_state,
                                     dp_config, distillation_enabled, training_params)

        # 分配客户端到GPU
        clients_per_gpu = allocate_clients_to_gpus(selected_client_ids, self.num_gpus)

        print(f"\n{'='*70}")
        print("GPU分配方案:")
        for gpu_id in range(self.num_gpus):
            if len(clients_per_gpu[gpu_id]) > 0:
                print(f"  GPU {gpu_id}: 客户端 {clients_per_gpu[gpu_id][:3]}{'...' if len(clients_per_gpu[gpu_id]) > 3 else ''} "
                      f"({len(clients_per_gpu[gpu_id])}个)")
        print(f"{'='*70}\n")

        # 选择训练函数
        if self.mode == 'parallel':
            train_func = train_clients_on_gpu_parallel
            print("使用方案2: GPU内部并行训练\n")
        else:
            train_func = train_clients_on_gpu_sequential
            print("使用方案1: 每GPU串行训练\n")

        # 创建进程
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []

        start_time = time.time()

        for gpu_id in range(self.num_gpus):
            if len(clients_per_gpu[gpu_id]) == 0:
                continue

            p = mp.Process(
                target=train_func,
                args=(gpu_id, clients_per_gpu[gpu_id], clients, global_model_state,
                      dp_config, distillation_enabled, training_params, return_dict)
            )
            p.start()
            processes.append(p)

        # 等待所有进程完成
        for p in processes:
            p.join()

        elapsed_time = time.time() - start_time
        print(f"\n所有GPU训练完成，耗时: {elapsed_time:.2f}秒\n")

        # 收集结果
        all_results = {}
        for gpu_id in range(self.num_gpus):
            if gpu_id in return_dict:
                for result in return_dict[gpu_id]:
                    if result['success']:
                        all_results[result['client_id']] = result
                    else:
                        print(f"警告: 客户端 {result['client_id']} 训练失败: {result['error']}")

        return all_results

    def _train_on_cpu(self, selected_client_ids: List[int], clients: List,
                     global_model_state: Dict, dp_config,
                     distillation_enabled: bool, training_params: Dict) -> Dict[int, Dict]:
        """CPU回退方案：串行训练"""
        print("使用CPU串行训练...")
        device = torch.device('cpu')

        all_results = {}
        for client_id in selected_client_ids:
            client = clients[client_id]
            result = train_single_client(
                client_id, client, global_model_state, device,
                dp_config, distillation_enabled, training_params
            )
            if result['success']:
                all_results[client_id] = result

        return all_results


def print_gpu_info():
    """打印GPU信息"""
    num_gpus = get_gpu_count()

    if num_gpus == 0:
        print("未检测到可用GPU")
        return

    print(f"\n{'='*70}")
    print(f"检测到 {num_gpus} 个GPU:")
    print(f"{'='*70}")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1e9
        print(f"GPU {i}: {props.name}")
        print(f"  显存: {memory_gb:.1f} GB")
        print(f"  计算能力: {props.major}.{props.minor}")
        print(f"  多处理器数量: {props.multi_processor_count}")

    print(f"{'='*70}\n")
