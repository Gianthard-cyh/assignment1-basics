import torch
import numpy as np

def get_batch(x, batch_size, context_length, device):
    n = len(x)

    max_val = n - context_length
    ix = torch.randint(0, max_val, (batch_size,))

    inputs = torch.stack([torch.from_numpy(x[i : i + context_length].astype(np.int64)) for i in ix])
    targets = torch.stack([torch.from_numpy(x[i + 1 : i + 1 + context_length].astype(np.int64)) for i in ix])

    return inputs.to(device), targets.to(device)
