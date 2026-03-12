from einops import einsum
import torch
import torch.nn as nn
from torch import Tensor, inf
from jaxtyping import Float, Bool

from .softmax import Softmax


class Attention(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.softmax = Softmax()

    def forward(
        self,
        Q: Float[Tensor, "... queries d_k"],
        K: Float[Tensor, "... keys d_k"],
        V: Float[Tensor, "... values d_v"],
        mask: Bool[Tensor, "... queries keys"] | None = None,
    ) -> Float[Tensor, "... queries d_v"]:
        score = einsum(Q, K, "... q d, ... k d -> q k") / (Q.size(-1) ** 0.5)
        if mask is not None:
            score = score.masked_fill(mask == False, -inf)
        attn = self.softmax(score, -1)
        # attn: (quries, keys)
        return einsum(attn, V, "... q k ,... k d -> ... q d")
