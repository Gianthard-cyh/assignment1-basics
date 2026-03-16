import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        初始化 RMSNorm 模块。

        参数:
            d_model (int): 模型的隐藏层维度
            eps (float): 用于数值稳定的常数
            device (torch.device | None): 参数存储的设备
            dtype (torch.dtype | None): 参数的数据类型
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        torch.nn.init.zeros_(self.weight)
        self.d_model = d_model
        self.eps = eps
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对形状为 (batch_size, sequence_length, d_model) 的输入张量执行均方根归一化。

        参数:
            x (torch.Tensor): 输入张量

        返回:
            torch.Tensor: 归一化后的张量
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        sq_sum = x.pow(2).mean(-1, keepdim=True)
        norm_x = x / torch.sqrt(sq_sum + self.eps) * self.weight
        return norm_x.to(in_dtype)
