"""
Linear Module
"""

from einops import einsum
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
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量应用线性变换。

        参数:
            x (torch.Tensor): 输入张量

        返回:
            torch.Tensor: 变换后的张量
        """
        return einsum(x, self.weight, "... i, o i -> ... o")
