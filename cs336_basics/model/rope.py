from einops import einsum
import torch
import torch.nn as nn


class RoPE(nn.Module):
    r: torch.Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        初始化 RoPE 模块。

        Args:
            theta: 控制旋转频率的基数 (Theta)
            d_k: 查询 (Query) 和键 (Key) 向量的维度
            max_seq_len: 支持的最大序列长度
            device: 存储缓存张量的设备
        """
        super().__init__()
        assert d_k % 2 == 0
        self.freq = 1 / (torch.tensor([theta]) ** (torch.arange(0, d_k, 2).float() / d_k))

        r_list = []
        for i in range(max_seq_len):
            pos = []
            for f in self.freq:
                cur_theta = i * f
                pos.append(
                    [[torch.cos(cur_theta), -torch.sin(cur_theta)], [torch.sin(cur_theta), torch.cos(cur_theta)]]
                )
            r_list.append(pos)
        self.register_buffer("r", torch.tensor(r_list), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        对输入张量应用旋转位置编码。

        Args:
            x: 输入张量，形状为 (..., seq_len, d_k)
            token_positions: 标记位置张量，形状为 (..., seq_len)

        Returns:
            应用 RoPE 后的张量，形状与 x 相同
        """
        x_pairs = x.view(*x.shape[:-1], -1, 2)
        r = self.r[token_positions]
        sum = einsum(x_pairs, r, "... s d v, ... s d v2 v -> ... s d v2")
        return sum.reshape(*x.shape)
