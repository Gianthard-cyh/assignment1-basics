import torch
from collections.abc import Iterable

@torch.no_grad()
def clip_gradient(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return 0.0

    norms = torch._foreach_norm(parameters, 2)
    total_norm = torch.linalg.vector_norm(torch.stack(norms), 2)
    clip_coef = max_l2_norm / (total_norm + 1e-6)

    if clip_coef < 1:
        torch._foreach_mul_( [p.grad for p in parameters], clip_coef.item() if torch.is_tensor(clip_coef) else clip_coef)
            
    return total_norm