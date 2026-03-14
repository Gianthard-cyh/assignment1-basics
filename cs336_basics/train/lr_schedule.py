import math


def cosine_lr_schedule(t: int, lr_max: float, lr_min: float, t_w: float, t_c: float):
    if t < t_w:
        return t / t_w * lr_max
    elif t_w <= t <= t_c:
        return lr_min + 1 / 2 * (1 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min
