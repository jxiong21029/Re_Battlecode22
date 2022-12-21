import numpy as np
import pytest

from Re_Battlecode22.utils import pos_symmetry_trsfm

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

    for num_rounds in (1, 5, 5, 5, 100, 100):
        ret.append(make_sample_env(num_rounds, seed=rng.integers(2**32)))
    return ret


def test_symmetries():
    starty, startx = 2, 3
    expected_results = {
        (2, 3),
        (3, 2),
        (1, 3),
        (3, 1),
        (2, 0),
        (0, 2),
        (1, 0),
        (0, 1),
    }
    actual_results = set()
    for symmetry in range(8):
        result = pos_symmetry_trsfm(starty, startx, symmetry, 4, 4)
        actual_results.add(result)
        assert result == pos_symmetry_trsfm(starty, startx, symmetry + 8, 4, 4)
    assert expected_results == actual_results


def test_pos_map(sample_envs: list[BattlecodeEnv]):
    for env in sample_envs:
        for unit in env.units:
            assert env.pos_map[(unit.y, unit.x)] is unit, breakpoint()
        for pos, unit in env.pos_map.items():
            assert pos == (unit.y, unit.x)
