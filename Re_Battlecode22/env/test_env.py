import copy
import random

import numpy as np
import pytest

from ..utils import symmetry_transform
from .battlecode_env import BattlecodeEnv
from .entities import Archon

SEED = 69420


def make_sample_env(num_rounds):
    env = BattlecodeEnv()
    env.reset()
    for _ in range(num_rounds):
        for bot, act_mask in env.iter_agents():
            selected_action = random.choice(np.arange(bot.action_space.n)[act_mask])
            env.step(bot, selected_action)
    return env


@pytest.fixture
def sample_envs():
    random.seed(0xBEEF)

    ret = [BattlecodeEnv()]
    ret[0].reset()

    for num_rounds in (1, 5, 5, 5, 100, 100, 100, 100):
        ret.append(make_sample_env(num_rounds))
    return ret


def test_observations():
    for _ in range(10):
        env = make_sample_env(100)
        env.observations()


# def test_symmetry_transform():
#     starty, startx = 2, 3
#     expected_results = {
#         (2, 3),
#         (3, 2),
#         (1, 3),
#         (3, 1),
#         (2, 0),
#         (0, 2),
#         (1, 0),
#         (0, 1),
#     }
#     actual_results = set()
#     for symmetry in range(8):
#         result = symmetry_transform(starty, startx, symmetry, 4, 4)
#         actual_results.add(result)
#         assert result == symmetry_transform(starty, startx, symmetry + 8, 4, 4)
#     assert expected_results == actual_results
#
#
# def test_terrain(sample_envs):
#     for env in sample_envs:
#         assert (0 <= env.rubble).all() and (env.rubble <= 100).all()
#
#
# def test_archon_counts(sample_envs):
#     for env in sample_envs:
#         for team in (0, 1):
#             assert (
#                 sum(
#                     1
#                     for unit in env.units
#                     if unit.team == team and isinstance(unit, Archon)
#                 )
#                 == env.archon_counts[team]
#             )
#
#
# def test_global_symmetry_yflip(sample_envs):
#     for env in sample_envs:
#         env: BattlecodeEnv
#         yflip = copy.deepcopy(env)
#
#         yflip.rubble = yflip.rubble[::-1]
#         yflip.lead = yflip.lead[::-1]
#         yflip.gold = yflip.gold[::-1]
#         for unit in yflip.units:
#             unit.y = yflip.rubble.shape[0] - unit.y - 1
#         yflip.pos_map = {(unit.y, unit.x): unit for unit in yflip.units}
#
#         orig = env.global_observation(symmetry=1)
#         copied = yflip.global_observation(symmetry=0)
#         for i in range(orig.shape[0]):
#             if i == 7:
#                 continue
#             assert np.isclose(orig[i], copied[i]).all(), str(i)
#
#
# def test_global_symmetry_tranpose(sample_envs):
#     for env in sample_envs:
#         env: BattlecodeEnv
#         tranposed = copy.deepcopy(env)
#
#         tranposed.rubble = tranposed.rubble.transpose()
#         tranposed.lead = tranposed.lead.transpose()
#         tranposed.gold = tranposed.gold.transpose()
#         for unit in tranposed.units:
#             unit.x, unit.y = unit.y, unit.x
#         tranposed.pos_map = {(unit.y, unit.x): unit for unit in tranposed.units}
#
#         orig = env.global_observation(symmetry=4)
#         copied = tranposed.global_observation(symmetry=0)
#         for i in range(orig.shape[0]):
#             assert np.isclose(orig[i], copied[i]).all(), str(i)
#
#
# def test_global_symmetry_teamflip(sample_envs):
#     for env in sample_envs:
#         env: BattlecodeEnv
#         teamflip = copy.deepcopy(env)
#
#         for unit in teamflip.units:
#             unit.team = 1 - unit.team
#         teamflip.lead_banks = teamflip.lead_banks[::-1]
#         teamflip.gold_banks = teamflip.gold_banks[::-1]
#
#         orig = env.global_observation(symmetry=8)
#         copied = teamflip.global_observation(symmetry=0)
#         for i in range(orig.shape[0]):
#             assert np.isclose(orig[i], copied[i]).all(), str(i)
#
#
# def test_pos_map(sample_envs: list[BattlecodeEnv]):
#     for env in sample_envs:
#         for unit in env.units:
#             assert env.pos_map[(unit.y, unit.x)] is unit, breakpoint()
#         for pos, unit in env.pos_map.items():
#             assert pos == (unit.y, unit.x)
#
#
# def test_unsigned_problems(sample_envs: list[BattlecodeEnv]):
#     for env in sample_envs:
#         assert (env.lead <= 200).all()
#         assert (env.gold <= 200).all()
