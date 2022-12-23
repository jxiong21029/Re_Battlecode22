import torch

from Re_Battlecode22.env import BattlecodeEnv
from Re_Battlecode22.trainer import Trainer
from Re_Battlecode22.utils import Logger

le_seed = 0xBEEF

torch.manual_seed(le_seed)
trainer = Trainer(
    BattlecodeEnv(seed=le_seed), verbose=True, pre_learning_steps=1000, seed=le_seed
)
trainer.learn(num_steps=100_000)
