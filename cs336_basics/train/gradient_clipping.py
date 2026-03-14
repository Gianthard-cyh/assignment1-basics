import torch
from collections.abc import Iterable


def clip_gradient(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    parameters = [p for p in parameters if p.grad is not None]
    grads = [p.grad.detach().view(-1) for p in parameters if p.grad is not None]
    all_grads = torch.cat(grads)
    total_norm = torch.norm(all_grads, 2)

    clip_coef = max_l2_norm / (total_norm + 1e-6)

    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)
