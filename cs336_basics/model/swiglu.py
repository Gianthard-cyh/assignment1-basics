import torch
from torch import nn
from .silu import SiLU
from .linear import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device)
        self.w2 = Linear(d_ff, d_model, device)
        self.w3 = Linear(d_model, d_ff, device)

        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU层。

        参数:
            x (torch.Tensor): 输入张量

        返回:
            torch.Tensor: 输出张量
        """
        return self.w2(self.silu(self.w1(x)) * (self.w3(x)))
