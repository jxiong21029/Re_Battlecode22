import random

import numpy as np
import trueskill

from ..env import BattlecodeEnv
from .agents import LinearAgents


def play(seed, p0: LinearAgents, p1: LinearAgents | None = None):
    rng = random.Random(seed)
    env = BattlecodeEnv(max_episode_length=200)

    env.reset(seed=rng.randrange(2**32))
    p0.reset_seeds(seed=rng.randrange(2**32))
    if p1 is not None:
        p1.reset_seeds(seed=rng.randrange(2**32))
    while not env.done:
        for unit, action_mask in env.iter_agents():
            self_obs = env.self_observation(unit)
            tile_obs = env.tile_observations(unit.team)

            if unit.team == 0:
                action = p0.predict(unit, self_obs, tile_obs, action_mask)
            elif p1 is None:
                action = rng.choice(np.arange(action_mask.shape[0])[action_mask])
            else:
                action = p1.predict(unit, self_obs, tile_obs, action_mask)
            env.step(unit, action)

    return env.winner()


trueskill.setup(draw_probability=0.0)


class League:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.agents = []
        self.ratings = [trueskill.Rating()]

    def play_and_adjust(self, i, j):
        game_seed = self.rng.randrange(2**32)
        if j == 0:
            winner = play(game_seed, self.agents[i], None)
        else:
            winner = play(game_seed, self.agents[i], self.agents[j - 1])
        if winner == 0:
            self.ratings[i], self.ratings[j] = trueskill.rate_1vs1(
                self.ratings[i], self.ratings[j]
            )
        else:
            self.ratings[j], self.ratings[i] = trueskill.rate_1vs1(
                self.ratings[j], self.ratings[i]
            )

    def insert_agent(self, agent: LinearAgents, placement_matches=12):
        self.agents.append(agent)
        self.ratings.append(trueskill.Rating())

        for _ in range(placement_matches):
            j = self.rng.randrange(len(self.agents) + 1)
            self.play_and_adjust(-1, j)

    def run_matches(self, n):
        for _ in range(n):
            i = self.rng.randrange(len(self.agents))
            j = self.rng.randrange(len(self.agents) + 1)
            self.play_and_adjust(i, j)

    def normalized_ratings(self):
        exposures = [trueskill.global_env().expose(rating) for rating in self.ratings]
        return [x - exposures[0] for x in exposures[1:]]
