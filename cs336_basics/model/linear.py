"""
Linear Module
"""

import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        初始化自定义的线性变换模块（无偏置）。
        
        参数:
            in_features (int): 输入特征的维度
            out_features (int): 输出特征的维度
            device (torch.device | None): 参数存储的设备
            dtype (torch.dtype | None): 参数的数据类型
        """
        super().__init__()
        # TODO: 构造权重并存储为 self.W，确保包裹在 nn.Parameter 中
        # 注意 1: 根据要求，为了内存顺序的原因，构造 W 而不是 W 的转置
        # 注意 2: 使用 torch.nn.init.trunc_normal_ 来初始化权重 W
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量应用线性变换。
        
        参数:
            x (torch.Tensor): 输入张量
            
        返回:
            torch.Tensor: 变换后的张量
        """
        # TODO: 在此实现输入 x 与权重 W 的矩阵乘法
        # 注意: 严格禁止使用 nn.Linear 或 nn.functional.linear
        pass