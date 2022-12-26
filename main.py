import random

import torch

from Re_Battlecode22.env import BattlecodeEnv
from Re_Battlecode22.qmix import Trainer

random.seed(0)
torch.manual_seed(random.randrange(2**32))

trainer = Trainer(
    BattlecodeEnv(max_episode_length=200),
    epsilon_decrease_steps=0,
    verbose=True,
    seed=random.randrange(2**32),
)
trainer.evaluate(num_episodes=10)
trainer.gather_random_experience(32000)
while True:
    trainer.learn(1000)
    trainer.evaluate(num_episodes=10)
