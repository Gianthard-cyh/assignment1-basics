import torch
from torch import nn


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量执行SiLU激活。

        参数:
            x (torch.Tensor): 输入张量

        返回:
            torch.Tensor: 激活后的张量
        """
        return x * self.sigmoid(x)
