import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

from cs336_basics.model.embedding import Embedding
from cs336_basics.model.linear import Linear
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


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: Int[Tensor, "batch_size sequence_length"]):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
