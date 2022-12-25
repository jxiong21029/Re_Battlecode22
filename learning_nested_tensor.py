import random
import timeit

import torch
import torch.nn as nn

observations = [
    [torch.rand(11) for _ in range(random.choice([2, 4, 6, 8]))] for _ in range(32)
]
linear = nn.Linear(11, 4)
optim = torch.optim.Adam(linear.parameters())


def nested_trial():
    stacked = [torch.stack(entry, dim=0) for entry in observations]
    nstd = torch.nested.nested_tensor(stacked)
    preds = linear(nstd)
    loss = torch.tensor(0.0)
    for tensor in preds.unbind():
        loss += tensor.pow(2).sum()

    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()


def vanilla_trial():
    loss = torch.tensor(0.0)
    for obs_list in observations:
        loss += linear(torch.stack(obs_list, dim=0)).pow(2).sum()

    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()


print(timeit.Timer("nested_trial()", globals=globals()).timeit(1000))
print(timeit.Timer("vanilla_trial()", globals=globals()).timeit(1000))
