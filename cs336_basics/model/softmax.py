import torch
from torch import nn


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        在给定维度上应用Softmax函数。

        参数：
          x (torch.Tensor): 输入张量
          dim (int): 执行softmax的维度

        返回：
          torch.Tensor: 输出张量
        """
        x_max = torch.max(x, dim=dim, keepdim=True)[0]
        e_x = torch.e ** (x - x_max)
        sum = torch.sum(e_x, dim=dim, keepdim=True)
        return e_x / sum
