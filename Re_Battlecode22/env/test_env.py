import copy
import random

import numpy as np
import pytest

from . import Archon, BattlecodeEnv, Builder, Laboratory, Miner, Sage, Soldier


def make_sample_env(num_rounds):
    env = BattlecodeEnv()
    env.reset()
    for _ in range(num_rounds):
        for unit, action_mask in env.iter_agents():
            selected_action = random.choice(np.arange(unit.num_actions)[action_mask])
            env.step(unit, selected_action)
    return env


def make_dense_env():
    env = BattlecodeEnv()
    env.reset()
    for cls in (Miner, Builder, Soldier, Sage, Laboratory):
        for _ in range(10):
            y = random.randrange(env.height)
            x = random.randrange(env.width)
            if (y, x) not in env.pos_map:
                new_unit = cls(y=y, x=x, team=random.randint(0, 1))
                env.units.append(new_unit)
                env.pos_map[y, x] = new_unit
    env.lead_banks = [random.randint(0, 500), random.randint(0, 500)]
    env.gold_banks = [random.randint(0, 100), random.randint(0, 100)]
    return env


@pytest.fixture
def test_envs():
    random.seed(42)

    ret = [BattlecodeEnv()]
    ret[0].reset()

    for num_rounds in (1, 5, 100, 100):
        ret.append(make_sample_env(num_rounds))
    for _ in range(5):
        ret.append(make_dense_env())
    return ret


def test_observations(test_envs: list[BattlecodeEnv]):
    for env in test_envs:
        for unit in env.units:
            self_obs = env.self_observation(unit)
            assert np.isclose(self_obs[0], (unit.y - env.height / 2 + 1 / 2) / 30)


def test_generation(test_envs: list[BattlecodeEnv]):
    for env in test_envs:
        assert (0 <= env.rubble).all() and (env.rubble <= 100).all()
        assert env.lead.min() >= 0 and env.gold.min() >= 0


def test_archon_counts(test_envs: list[BattlecodeEnv]):
    for env in test_envs:
        for team in (0, 1):
            assert (
                sum(
                    1
                    for unit in env.units
                    if unit.team == team and isinstance(unit, Archon)
                )
                == env.unit_counts[Archon][team]
            )


def test_pos_map(test_envs: list[BattlecodeEnv]):
    for env in test_envs:
        for unit in env.units:
            assert (unit.y, unit.x) in env.pos_map
            assert env.pos_map[(unit.y, unit.x)] is unit
        for pos, unit in env.pos_map.items():
            assert pos == (unit.y, unit.x)


def test_action_mask(test_envs: list[BattlecodeEnv]):
    for env in test_envs:
        stored = copy.deepcopy(env)

        idx = [random.randrange(len(env.units)) for _ in range(50)]
        for i in idx:
            env = copy.deepcopy(stored)
            for j, (unit, action_mask) in enumerate(env.iter_agents()):
                if i == j and not np.all(action_mask):
                    illegal_action = random.choice(
                        np.arange(unit.num_actions)[~action_mask]
                    )
                    with pytest.raises(AssertionError):
                        env.step(unit, illegal_action)
                        print(">>>>>", unit, illegal_action)
                else:
                    legal_action = random.choice(
                        np.arange(unit.num_actions)[action_mask]
                    )
                    env.step(unit, legal_action)


def test_tile_cache_correctness(test_envs: list[BattlecodeEnv]):
    for env in test_envs:
        for unit, action_mask in env.iter_agents():
            recomputed = env.recomputed_tiles()
            for i in range(recomputed.shape[0]):
                assert np.array_equal(recomputed[i], env.cached_tiles[i]), str(i)
            selected_action = random.choice(np.arange(unit.num_actions)[action_mask])
            env.step(unit, selected_action)


def test_state(test_envs: list[BattlecodeEnv]):
    for env in test_envs:
        assert np.abs(env.state()).max() < 10
