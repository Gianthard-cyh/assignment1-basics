import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

from cs336_basics.model.mha import MHA
from cs336_basics.model.rmsnorm import RMSNorm
from cs336_basics.model.rope import RoPE
from cs336_basics.model.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)
        self.rope = RoPE(theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len)
        self.attn = MHA(d_model=d_model, num_heads=num_heads, rope=self.rope)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: Float[Tensor, "batch sequence_length d_model"]):
        token_pos = torch.arange(1, x.size(-2) + 1)
        y = x + self.attn(self.ln1(x), token_pos)
        y = y + self.ffn(self.ln2(y))
        return y
