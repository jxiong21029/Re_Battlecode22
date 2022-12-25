import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from .env import Archon, BattlecodeEnv, Builder, Miner, Sage, Soldier
from .utils import Logger


@dataclass
class Transition:
    env_state: BattlecodeEnv
    actions: list[int]
    reward: float


class QMixAgents(nn.Module):
    def __init__(
        self,
        obs_refresh_prob=0.03,
        augment_inputs=True,
        augment_targets=True,
        target_augmentations=4,
        logger=None,
    ):
        super().__init__()
        self.obs_refresh_prob = obs_refresh_prob
        self.augment_inputs = augment_inputs  # TODO augmentations
        self.augment_targets = augment_targets
        self.target_augmentations = target_augmentations
        self.logger = logger

        self.utility_nets = nn.ModuleDict(
            {
                cls.__name__: nn.Linear(
                    cls.observation_space.shape[0], cls.action_space.n
                )
                for cls in (Miner, Builder, Soldier, Sage, Archon)
            }
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(30, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
        )
        self.squeeze_and_excite = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=0),  # no batching
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def get_q_preds(self, env):
        obses: dict[type, np.ndarray] = env.observations()
        q_preds = {}
        for cls in obses.keys():
            cls_preds = self.utility_nets[cls.__name__](torch.tensor(obses[cls]))
            i = 0
            for unit in env.units:
                if isinstance(unit, cls):
                    q_preds[unit] = cls_preds[i]
                    i += 1
        return q_preds

    def explore_step(self, env: BattlecodeEnv, epsilon=0.01):
        """Runs epsilon-greedy exploration: steps env, returning actions and reward"""

        preds = {}
        actions = []
        for unit, action_mask in env.iter_agents():
            if random.random() < epsilon:
                action = random.choice(np.arange(unit.action_space.n)[action_mask])
            else:
                if unit not in preds or random.random() < self.obs_refresh_prob:
                    preds = self.get_q_preds(env)
                if unit.team == 0:
                    preds[unit][~action_mask] = -1e8
                    action = torch.argmax(preds[unit]).item()
                else:
                    preds[unit][~action_mask] = 1e8
                    action = torch.argmin(preds[unit]).item()
            actions.append(action)
            env.step(unit, action)
        reward = env.get_team_reward()
        return actions, reward

    def forward(
        self,
        env: BattlecodeEnv,
        actions: list[int],
    ) -> tuple[torch.Tensor, BattlecodeEnv]:
        """Computes Q-values given actions"""
        x = self.conv1(torch.tensor(env.global_observation()))
        x = self.squeeze_and_excite(x)[:, None, None] * x
        weights = F.softplus(self.head(x).squeeze(0))  # (1, H, W), then (H, W)

        global_q = torch.tensor(0.0)
        preds = {}
        for i, (unit, action_mask) in enumerate(env.iter_agents()):
            if unit not in preds:
                preds = self.get_q_preds(env)
            local_q = preds[unit][actions[i]]
            global_q += weights[unit.y, unit.x] * local_q  # linear QMIX
            env.step(unit, actions[i])

        return global_q, env

    # def forward_precomputed(self, env_state, observations, actions):
    #     x = self.conv1(torch.tensor(env_state.global_observation()))
    #     x = self.squeeze_and_excite(x)[:, None, None] * x
    #     weights = F.softplus(self.head(x).squeeze(0))  # (1, H, W), then (H, W)
    #
    #     global_q = torch.tensor(0.0)
    #     for i, (bot, obs, action) in enumerate(
    #         zip(env_state.units, observations, actions)
    #     ):
    #         action_qs = self.utility_nets[bot.__class__.__name__](obs)
    #         global_q += weights[bot.y, bot.x] * action_qs[action]
    #     return global_q


class Trainer:
    def __init__(
        self,
        env: BattlecodeEnv,
        discount_factor=0.99,
        buffer_size=128_000,
        steps_per_update=4,
        minibatch_size=32,
        lr=1e-3,
        target_update_period=8000,
        target_update_polyak=1,
        pre_learning_steps=8_000,
        epsilon_decrease_steps=64_000,
        epsilon_max=1,
        epsilon_min=0.01,
        logging_interval=1000,
        verbose=False,
    ):
        assert pre_learning_steps + epsilon_decrease_steps <= buffer_size

        self.logging_interval = logging_interval
        self.logger = Logger()
        self.verbose = verbose

        self.env = env
        self.env.reset()
        self.discount_factor = discount_factor

        self.buffer_size = buffer_size
        self.steps_per_update = steps_per_update
        self.minibatch_size = minibatch_size

        self.model = QMixAgents(logger=self.logger)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.target_model = copy.deepcopy(self.model)
        self.target_update_period = target_update_period
        self.tau = target_update_polyak

        self.pre_learning_steps = pre_learning_steps
        self.epsilon_decrease_steps = epsilon_decrease_steps
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min

        self.replay_buffer = []
        self.t = 0

    def curr_epsilon(self):
        if self.t < self.pre_learning_steps:
            return 1
        return max(
            self.epsilon_min,
            self.epsilon_min
            + (self.epsilon_max - self.epsilon_min)
            * (1 - (self.t - self.pre_learning_steps) / self.epsilon_decrease_steps),
        )

    def store(self, state, actions, reward):
        new_transition = Transition(state, actions, reward)
        if len(self.replay_buffer) < self.buffer_size:
            self.replay_buffer.append(new_transition)
        else:
            self.replay_buffer[self.t % self.buffer_size] = new_transition

    def learn(self, num_steps):
        iterator = tqdm.trange(num_steps) if self.verbose else range(num_steps)
        for _ in iterator:
            curr_state = copy.deepcopy(self.env)
            actions, reward = self.model.explore_step(self.env, self.curr_epsilon())
            self.store(curr_state, actions, reward)
            if self.env.done:
                self.env.reset()

            if (
                self.t >= self.pre_learning_steps
                and (self.t - self.pre_learning_steps + 1) % self.steps_per_update == 0
            ):
                loss = torch.tensor(0.0)
                for _ in range(self.minibatch_size):
                    transition = random.choice(self.replay_buffer)
                    y, next_state = self.model(
                        copy.deepcopy(transition.env_state), transition.actions
                    )

                    if next_state.done:
                        target = transition.reward
                    else:
                        next_actions, _ = self.model.explore_step(
                            copy.deepcopy(next_state), epsilon=0
                        )
                        next_q, _ = self.target_model(next_state, actions=next_actions)
                        target = transition.reward + self.discount_factor * next_q

                    loss += ((y - target) ** 2) / self.minibatch_size
                    self.logger.push(td_error=(y - target).item(), loss=loss.item())

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                self.optim.step()

            if (
                self.t >= self.pre_learning_steps
                and (self.t - self.pre_learning_steps + 1) % self.target_update_period
                == 0
            ):
                with torch.no_grad():
                    for p1, p2 in zip(
                        self.model.parameters(), self.target_model.parameters()
                    ):
                        p2[:] = self.tau * p1 + (1 - self.tau) * p2

            if (self.t + 1) % self.logging_interval == 0:
                self.logger.step()
                self.logger.generate_plots()

            self.t += 1

    def eval_with_render(self, eps=0):
        from .env.rendering import Renderer

        renderer = Renderer()
        eval_env = copy.deepcopy(self.env)
        eval_env.reset()

        with tqdm.tqdm(total=2000) as pbar:
            while not eval_env.done:
                renderer.render(eval_env)
                self.model.explore_step(eval_env, eps)
                pbar.update(1)
