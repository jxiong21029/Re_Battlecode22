import random

import torch

from Re_Battlecode22.env import BattlecodeEnv
from Re_Battlecode22.qmix import Trainer

random.seed(0)
torch.manual_seed(random.randrange(0, 2**32))

trainer = Trainer(
    BattlecodeEnv(max_episode_length=200),
    pre_learning_steps=1_000,
    epsilon_decrease_steps=0,
    verbose=True,
)
trainer.learn(1_000)
trainer.learn(9_000)
