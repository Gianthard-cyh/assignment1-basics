import torch
from torch import Tensor
from jaxtyping import Int, Float


def log_softmax(inputs: Float[Tensor, "..."], dim: int):
    m, _ = torch.max(inputs, dim=dim, keepdim=True)
    lse = m + torch.log(torch.exp(inputs - m).sum(dim, keepdim=True))
    return inputs - lse


def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]):
    logits_flat = inputs.view(-1, inputs.size(-1))
    targets_flat = targets.view(-1).long()
    
    log_probs = log_softmax(logits_flat, -1)
    batch_idx = torch.arange(targets_flat.shape[0], device=inputs.device)
    target_log_probs = log_probs[batch_idx, targets_flat]
    
    return -target_log_probs.mean()