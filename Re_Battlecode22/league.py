import numpy as np
import torch
from env import BattlecodeEnv
from qmix import QMixAgents, Trainer


def expected_score(own_rating, other_rating):
    return 1 / (1 + 10 ** ((other_rating - own_rating) / 400))


def adjusted_ratings(rating_a, rating_b, score, k=32):
    expected_a = expected_score(rating_a, rating_b)
    new_a = rating_a + k * (score - expected_a)
    new_b = rating_a - new_a + rating_b
    return new_a, new_b


def play(p0: QMixAgents, p1: QMixAgents | None = None):
    # None: random agent
    env = BattlecodeEnv(max_episode_length=200)
    env.reset()
    while not env.done:
        for bot, obs, action_mask in env.iter_agents():
            if bot.team == 0:
                action_qs = p0.utility_nets[bot.__class__.__name__](torch.tensor(obs))
                action = torch.argmax(action_qs).item()
            else:
                assert bot.team == 1
                if p1 is None:
                    action = np.random.choice(
                        np.arange(action_mask.shape[0])[action_mask]
                    )
                else:
                    action_qs = p1.utility_nets[bot.__class__.__name__](
                        torch.tensor(obs)
                    )
                    action = torch.argmin(action_qs).item()

            env.step(bot, action)

    return env.winner()


trainer = Trainer()
