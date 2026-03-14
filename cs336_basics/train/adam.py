import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay: float,
        betas: tuple[float, float],
        eps,
    ) -> None:
        defaults = {"lr": lr, "weight_decay": weight_decay, "beta1": betas[0], "beta2": betas[1]}
        self.eps = eps
        super().__init__(params, defaults)

    def step(self) -> None:
        eps = self.eps
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad**2)

                adjusted_lr = lr * ((1 - beta2**t) ** 0.5 / (1 - beta1**t))
                p.data -= adjusted_lr * m / (v**0.5 + eps)
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
