import copy
import random

import numpy as np
import pytest

from . import Archon, BattlecodeEnv, Builder, Laboratory, Miner, Sage, Soldier


def make_sample_env(num_rounds):
    env = BattlecodeEnv()
    env.reset()
    for _ in range(num_rounds):
        for bot, act_mask in env.iter_agents():
            selected_action = random.choice(np.arange(bot.action_space.n)[act_mask])
            env.step(bot, selected_action)
    return env


def make_dense_env():
    env = BattlecodeEnv()
    env.reset()
    env.lead_banks = [9999, 9999]
    env.gold_banks = [9999, 9999]
    for cls in (Miner, Builder, Soldier, Sage, Laboratory):
        for _ in range(10):
            y = random.randrange(env.height)
            x = random.randrange(env.width)
            if (y, x) not in env.pos_map:
                env.spawn(cls, (y, x), random.randint(0, 1))
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
    cnt = 0
    for env in test_envs:
        obs = env.observations()
        for unit in env.units:
            if unit.team == 0:
                assert np.isclose(obs[1, unit.y, unit.x], np.log1p(unit.curr_hp) / 2)
                assert obs[2, unit.y, unit.x] == 0

                if isinstance(unit, (Archon, Laboratory)):
                    cnt += 1
                    assert obs[3, unit.y, unit.x] == 1
                else:
                    assert obs[3, unit.y, unit.x] == 0
                assert obs[4, unit.y, unit.x] == 0
            else:
                assert obs[1, unit.y, unit.x] == 0
                assert np.isclose(obs[2, unit.y, unit.x], np.log1p(unit.curr_hp) / 2)

                if isinstance(unit, (Archon, Laboratory)):
                    assert obs[4, unit.y, unit.x] == 1
                else:
                    assert obs[4, unit.y, unit.x] == 0
                assert obs[3, unit.y, unit.x] == 0
    assert cnt


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
                == env.archon_counts[team]
            )


def test_pos_map(test_envs: list[BattlecodeEnv]):
    for env in test_envs:
        for unit in env.units:
            assert env.pos_map[(unit.y, unit.x)] is unit, breakpoint()
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
                        np.arange(unit.action_space.n)[~action_mask]
                    )
                    with pytest.raises(AssertionError):
                        env.step(unit, illegal_action)
                        print(">>>>>", unit, illegal_action)
                else:
                    legal_action = random.choice(
                        np.arange(unit.action_space.n)[action_mask]
                    )
                    env.step(unit, legal_action)


def test_state(test_envs: list[BattlecodeEnv]):
    for env in test_envs:
        assert np.abs(env.state()).max() < 10
