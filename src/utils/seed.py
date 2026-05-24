"""
随机种子固定模块

固定 Python random / numpy / torch / CUDA 的随机种子，确保实验可复现。
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    固定所有随机种子，确保实验可复现。

    Args:
        seed: 随机种子值，默认 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 场景

    # 确保 CUDA 卷积运算的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置环境变量（部分库会读取）
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[Seed] 随机种子已固定为 {seed}")
