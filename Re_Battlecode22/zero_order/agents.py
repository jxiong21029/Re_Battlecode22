import math

import numpy as np

from Re_Battlecode22.env import (
    Archon,
    BattlecodeEnv,
    Builder,
    Miner,
    Sage,
    Soldier,
    Unit,
)


class LinearAgents:
    def __init__(self, init_seed=None):
        rng = np.random.default_rng(init_seed)
        self.global_rng = None
        self.unit_rngs = None

        classes = (Miner, Builder, Soldier, Sage, Archon)
        self.weight_self = {
            cls: rng.normal(size=(cls.num_actions, 11 if cls == Archon else 5))
            for cls in classes
        }
        self.weight_tiles = {}
        for cls in classes:
            ksize = 1 + 2 * math.floor(math.sqrt(cls.vis_rad))
            mask = np.ones((cls.num_actions, 7, ksize, ksize))
            mid = ksize // 2
            for row in range(ksize):
                for col in range(ksize):
                    if (mid - row) ** 2 + (mid - col) ** 2 > cls.vis_rad:
                        mask[:, :, row, col] = 0

            weight = mask * rng.normal(size=(cls.num_actions, 7, ksize, ksize))
            self.weight_tiles[cls] = weight

        self.bias = {cls: rng.normal(size=cls.num_actions) for cls in classes}

    @property
    def parameters(self):
        classes = (Miner, Builder, Soldier, Sage, Archon)
        return np.concatenate(
            [self.weight_self[cls].flatten() for cls in classes]
            + [self.weight_tiles[cls].flatten() for cls in classes]
            + [self.bias[cls] for cls in classes]
        )

    @parameters.setter
    def parameters(self, value):
        classes = (Miner, Builder, Soldier, Sage, Archon)
        i = 0
        for cls in classes:
            x = self.weight_self[cls].ravel()
            x[:] = value[i : i + x.shape[0]]
            i += x.shape[0]

    def reset_seeds(self, seed: int):
        self.global_rng = np.random.default_rng(seed=seed)
        self.unit_rngs = {}

    def predict(self, unit: Unit, self_obs, tile_obs, action_mask):
        cls = unit.__class__
        logits = (self.weight_self[cls] * self_obs).sum(axis=1) + self.bias[cls]
        # self.weight_tiles[unit.__class__] * tile_obs

        pad = math.floor(math.sqrt(cls.vis_rad))
        ksize = 2 * pad + 1
        logits += np.tensordot(
            self.weight_tiles[unit.__class__],
            np.pad(tile_obs, ((0, 0), (pad, pad), (pad, pad)))[
                :, unit.y : unit.y + ksize, unit.x : unit.x + ksize
            ],
            axes=3,
        )

        if unit not in self.unit_rngs:
            self.unit_rngs[unit] = np.random.default_rng(
                self.global_rng.integers(2**32)
            )

        # gumbel-max trick: used to encourage "similar" results for similar logits
        # under the same seed, in order to reduce variance
        noisy = logits + self.unit_rngs[unit].gumbel(size=logits.shape)
        noisy[~action_mask] = -1e8
        selected = np.argmax(noisy)
        return selected
