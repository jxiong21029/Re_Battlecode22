from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import BattlecodeEnv
from .env.entities import *


@dataclass
class Transition:
    state: BattlecodeEnv
    actions: list[int]
    reward: int


class Model(nn.Module):
    def __init__(
        self, augment_inputs=True, augment_targets=True, target_augmentations=4
    ):
        super().__init__()
        self.augment_inputs = augment_inputs
        self.augment_targets = augment_targets
        self.target_augmentations = target_augmentations

        self.utility_nets = nn.ModuleDict(
            {
                cls.__name__: nn.Linear(
                    cls.observation_space.shape[0], cls.action_space.n
                )
                for cls in (Miner, Builder, Soldier, Sage, Archon)
            }
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(30, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
        )
        self.squeeze_and_excite = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(
        self,
        transition,
    ):
        if actions is None:
            utilities = [
                self.utility_nets[entity.__class__.__name__](obs).max()
                for entity, obs in zip(entities, observations)
            ]
        else:
            utilities = [
                self.utility_nets[entity.__class__.__name__](obs)[action]
                for entity, obs, action in zip(entities, observations, actions)
            ]

        x = self.conv1(state)
        x = self.squeeze_and_excite(x)[:, :, None, None] * x
        x = self.head(x)

        return sum(
            x[entity.y, entity.x] * x
            for entity, utility in zip(entities, utilities)
        )


class Trainer:
    def __init__(
        self,
        env: BattlecodeEnv,
        gamma=0.99,
        buffer_size=1_000_000,
        pre_learning_steps=50_000,
        steps_per_update=4,
        minibatch_size=32,
        lr=1e-4,
        target_update_period=10000,
        target_update_polyak=1,
    ):
        self.env = env
        self.gamma = gamma

        self.buffer_size = buffer_size

    def learn(self):
        pass
