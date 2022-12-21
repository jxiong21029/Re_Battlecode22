import numpy as np
import pytest

from .battlecode_env import BattlecodeEnv

SEED = 69420


def make_sample_env(num_rounds, seed):
    rng = np.random.default_rng(seed)
    env = BattlecodeEnv(seed=rng.integers(2**31))
    env.reset()
    for _ in range(num_rounds):
        for bot, obs, act_mask in env.agent_inputs():
            selected_action = rng.choice(np.arange(bot.action_space.n)[act_mask])
            env.step(bot, selected_action)
    return env


@pytest.fixture
def sample_envs():
    rng = np.random.default_rng(SEED)

    ret = [BattlecodeEnv(seed=rng.integers(2**32))]
    ret[0].reset()

    for num_rounds in (1, 5, 5, 5, 100):
        ret.append(make_sample_env(num_rounds, seed=rng.integers(2**32)))
    return ret


def test_pos_map(sample_envs: list[BattlecodeEnv]):
    for env in sample_envs:
        for unit in env.units:
            assert env.pos_map[(unit.y, unit.x)] is unit, breakpoint()
        for pos, unit in env.pos_map.items():
            assert pos == (unit.y, unit.x)
