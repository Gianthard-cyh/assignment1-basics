from sympy import Float
import torch
from torch import nn, Tensor
from .silu import SiLU
from jaxtyping import Float


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model))

        torch.nn.init.trunc_normal_(self.w1)
        torch.nn.init.trunc_normal_(self.w2)
        torch.nn.init.trunc_normal_(self.w3)

        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU层。

        参数:
            x (torch.Tensor): 输入张量

        返回:
            torch.Tensor: 输出张量
        """
        return (self.silu(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T
