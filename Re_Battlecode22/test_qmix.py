import pytest

from .env import BattlecodeEnv
from .qmix import Trainer


@pytest.mark.skip
def test_random_exploration():
    trainer = Trainer(BattlecodeEnv(max_episode_length=200))
    trainer.gather_random_experience(1000)
