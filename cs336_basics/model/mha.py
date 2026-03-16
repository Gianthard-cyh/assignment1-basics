from einops import einsum, rearrange
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

from cs336_basics.model.attention import Attention
from cs336_basics.model.rope import RoPE
from cs336_basics.model.linear import Linear


class MHA(nn.Module):
    """
    Multi Head Self-Attention Module
    """

    def __init__(self, d_model: int, num_heads: int, rope: RoPE | None = None, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, device)
        self.k_proj = Linear(d_model, d_model, device)
        self.v_proj = Linear(d_model, d_model, device)
        self.output_proj = Linear(d_model, d_model, device)

        self.attn = Attention()
        self.rope = rope
        self.device = device

    def forward(
        self,
        x: Float[Tensor, "... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        s_len = x.size(-2)
        mask = torch.tril(torch.ones(s_len, s_len,device=self.device))

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = rearrange(Q, "... s (h k) -> ... h s k", h=self.num_heads)
        K = rearrange(K, "... s (h k) -> ... h s k", h=self.num_heads)
        V = rearrange(V, "... s (h k) -> ... h s k", h=self.num_heads)

        if self.rope:
            assert token_positions is not None
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        O_raw = self.attn(Q, K, V, mask)
        O_raw = rearrange(O_raw, "... h s k -> ... s (h k)")

        return self.output_proj(O_raw)
