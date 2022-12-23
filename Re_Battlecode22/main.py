import torch

from Re_Battlecode22.env import BattlecodeEnv
from Re_Battlecode22.trainer import Trainer
from Re_Battlecode22.utils import Logger

le_seed = 0xBEEF + 5

torch.manual_seed(le_seed)
trainer = Trainer(
    BattlecodeEnv(max_episode_length=200, seed=le_seed),
    verbose=True,
    buffer_size=128_000,
    pre_learning_steps=32_000,
    seed=le_seed,
)
trainer.eval_with_render(eps=1)
# trainer.learn(num_steps=100_000)

# TODO: team 2 rewards - optimized wrong - should minimize - fix

# TODO: replay viewer (pygame?)
# TODO: more testing -- environment, trainer objects/refs, reward / value propagation
#   verify action masks
# TODO: more logging -- q value distribution (local + global), TD errors / losses
#   try: tune LR, modify initialization

# TODO: environment simplification:
#   shorter epsiode horizon
#   one map only
