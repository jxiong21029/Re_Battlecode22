from typing import Callable

import numpy as np


class SMTP:
    def __init__(
        self,
        objective: Callable,
        dims: int,
        lr: float,
        momentum=0.5,
        mode="min",
        seed=None,
    ):
        assert mode in ("min", "max")
        assert 0 <= momentum < 1

        self.rng = np.random.default_rng(seed=seed)

        self.x = self.rng.normal(size=dims)
        self.z = self.x
        self.v = 0

        self.objective = objective
        self.dims = dims
        self.lr = lr
        self.momentum = momentum
        self.mode = mode

    def learn(self):
        beta = self.momentum
        s = self.rng.normal(size=self.dims)
        s /= np.linalg.norm(s)
        v_pos = beta * self.v + s
        v_neg = beta * self.v - s
        x_pos = self.x - self.lr * v_pos
        x_neg = self.x - self.lr * v_neg
        z_pos = x_pos - self.lr * beta / (1 - beta) * v_pos
        z_neg = x_neg - self.lr * beta / (1 - beta) * v_neg

        # no point in caching previous scores for noisy objectives, as it leads to bias
        scores = [self.objective(z) for z in (self.z, z_pos, z_neg)]
        best_idx = np.argmin(scores) if self.mode == "min" else np.argmax(scores)
        self.z = (self.z, z_pos, z_neg)[best_idx]
        self.x = (self.x, x_pos, x_neg)[best_idx]
        self.v = (self.v, v_pos, v_neg)[best_idx]
