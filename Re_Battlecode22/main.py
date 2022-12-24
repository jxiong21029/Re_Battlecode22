import random

import torch

from Re_Battlecode22.env import BattlecodeEnv
from Re_Battlecode22.qmix import Trainer


def profileit(func):
    import cProfile

    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile"  # Name the data file sensibly
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(datafn)
        return retval

    return wrapper


@profileit
def train2(trainer):
    trainer.learn(num_steps=1000)


def main():
    le_seed = 0xBEEF + 5

    random.seed(le_seed)
    torch.manual_seed(le_seed)

    trainer = Trainer(
        BattlecodeEnv(max_episode_length=200),
        verbose=True,
        buffer_size=128_000,
        pre_learning_steps=1_000,
    )
    # trainer.eval_with_render(eps=1)
    trainer.learn(num_steps=1_000)
    train2(trainer)


main()


# TODO: reward normalization
# TODO: episode-length metrics
# TODO: more testing -- environment, trainer objects/refs, reward / value propagation
#   verify action masks
#   try: tune LR, modify initialization

# TODO: environment simplification:
#   one map only
