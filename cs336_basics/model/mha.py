from einops import einsum, rearrange
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

from cs336_basics.model.attention import Attention
from cs336_basics.model.rope import RoPE


class MHA(nn.Module):
    """
    Multi Head Self-Attention Module
    """

    def __init__(self, d_model: int, num_heads: int, rope: RoPE | None = None):
        super().__init__()
        d_k = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.k_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.o_proj = nn.Parameter(torch.empty(d_model, d_model))

        torch.nn.init.trunc_normal_(self.q_proj)
        torch.nn.init.trunc_normal_(self.k_proj)
        torch.nn.init.trunc_normal_(self.v_proj)
        torch.nn.init.trunc_normal_(self.o_proj)

        self.attn = Attention()
        self.rope = rope

    def forward(
        self,
        x: Float[Tensor, "... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        s_len = x.size(-2)
        self.mask = torch.tril(torch.ones(s_len, s_len))
        W_q = rearrange(self.q_proj, "(h k) d -> h k d", h=self.num_heads)
        W_k = rearrange(self.k_proj, "(h k) d -> h k d", h=self.num_heads)
        W_v = rearrange(self.v_proj, "(h k) d -> h k d", h=self.num_heads)
        W_o = rearrange(self.o_proj, "d (h k) -> h d k", h=self.num_heads)

        Q = einsum(x, W_q, "... s d, h k d -> ... h s k")
        K = einsum(x, W_k, "... s d, h k d -> ... h s k")
        V = einsum(x, W_v, "... s d, h k d -> ... h s k")

        if self.rope:
            assert token_positions is not None
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        O_raw = self.attn(Q, K, V, self.mask)
        return einsum(O_raw, W_o, "... h s k, h d k -> ... s d")
