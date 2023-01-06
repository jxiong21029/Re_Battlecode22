import random

import numpy as np

from ..env import Archon, BattlecodeEnv
from .agents import LinearAgents
from .elo import League


def test_agents_smoke():
    for seed in range(3):
        rng = random.Random(seed)

        env = BattlecodeEnv(max_episode_length=20)
        agents_a = LinearAgents(rng.randrange(2**32))
        agents_b = LinearAgents(rng.randrange(2**32))

        agents_a.reset_seeds(rng.randrange(2**32))
        agents_b.reset_seeds(rng.randrange(2**32))
        env.reset(rng.randrange(2**32))
        while not env.done:
            for unit, action_mask in env.iter_agents():
                self_obs = env.self_observation(unit)
                tile_obs = env.tile_observations(unit.team)

                if unit.team == 0:
                    action = agents_a.predict(unit, self_obs, tile_obs, action_mask)
                else:
                    action = agents_b.predict(unit, self_obs, tile_obs, action_mask)
                env.step(unit, action)


def test_parameters_getting_setting():
    rng = random.Random(42)
    agents_0 = LinearAgents(rng.randrange(2**32))
    agents_1 = LinearAgents(rng.randrange(2**32))

    assert len(agents_0.parameters.shape) == 1
    assert not np.array_equal(
        agents_0.weight_self[Archon], agents_1.weight_self[Archon]
    )
    agents_0.parameters = agents_1.parameters
    assert agents_0.parameters is not agents_1.parameters
    assert agents_0.weight_self[Archon] is not agents_1.weight_self[Archon]
    assert np.array_equal(agents_0.weight_self[Archon], agents_1.weight_self[Archon])


def test_league_evaluation():
    rng = random.Random(4242)
    agents = [LinearAgents(rng.randrange(2**32)) for _ in range(3)]

    league = League()
    for agent in agents:
        league.insert_agent(agent, placement_matches=1)
    league.run_matches(2)
    assert len(league.normalized_ratings()) == len(agents)
