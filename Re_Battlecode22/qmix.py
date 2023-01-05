import copy
import itertools
import math
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


class MaskedConv(nn.Conv2d):
    def __init__(self, unit_cls, device=None):
        ksize = 1 + 2 * math.floor(math.sqrt(unit_cls.vis_rad))
        super().__init__(
            in_channels=7,
            out_channels=unit_cls.action_space.n,
            kernel_size=ksize,
            padding="same",
            device=device,
        )
        mask = torch.ones((unit_cls.action_space.n, 7, ksize, ksize))
        mid = ksize // 2
        for row in range(ksize):
            for col in range(ksize):
                if (mid - row) ** 2 + (mid - col) ** 2 > unit_cls.vis_rad:
                    mask[:, :, row, col] = 0
        self.register_buffer("mask", mask)

    def forward(self, x):
        return self._conv_forward(x, self.mask * self.weight, self.bias)


class QMixAgents(nn.Module):
    def __init__(
        self,
        # augment_inputs=True,
        augment_targets=True,
        target_augmentations=4,
        logger=None,
        seed=None,
    ):
        super().__init__()
        self.rng = random.Random(seed)
        # self.augment_inputs = augment_inputs  # TODO augmentations
        self.augment_targets = augment_targets
        self.target_augmentations = target_augmentations
        self.logger = logger

        self.self_nets = nn.ModuleDict(
            {
                cls.__name__: nn.Linear(11 if cls == Archon else 5, cls.action_space.n)
                for cls in (Miner, Builder, Soldier, Sage, Archon)
            }
        )
        self.tile_nets = nn.ModuleDict(
            {
                cls.__name__: MaskedConv(cls)
                for cls in (Miner, Builder, Soldier, Sage, Archon)
            }
        )

        self.state_conv = nn.Sequential(
            nn.Conv2d(30, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
        )
        self.squeeze_and_excite = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=0),  # no batching
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
        self.weight_head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self.bias_head = nn.Sequential(nn.ReLU(), nn.Linear(16, 1))

    def explore_step(self, envs: list[BattlecodeEnv], epsilon: float):
        """Runs epsilon-greedy exploration: steps env, returning actions and reward"""
        cls_map = {
            Archon: 0,
            Miner: 1,
            Builder: 2,
            Soldier: 3,
            Sage: 4,
        }

        all_tiles = []
        max_height = max(env.height for env in envs)
        max_width = max(env.width for env in envs)
        for env in envs:
            tiles0 = env.tile_observations(team=0)
            height_pad = max_height - tiles0.shape[0]
            width_pad = max_width - tiles0.shape[1]

            all_tiles.append(
                F.pad(torch.tensor(tiles0), pad=(0, width_pad, 0, height_pad))
            )
            all_tiles.append(
                F.pad(
                    torch.tensor(env.tile_observations(team=1)),
                    pad=(0, width_pad, 0, height_pad),
                )
            )

        if epsilon == 1:
            tile_preds = {
                cls: self.tile_nets[cls.__name__](torch.stack(all_tiles))
                for cls in cls_map.keys()
            }
        else:
            tile_preds = {}

        for i, env in enumerate(envs):
            actions = []
            for unit, action_mask in env.iter_agents():
                if self.rng.random() < epsilon:
                    action = self.rng.choice(np.arange(unit.action_space.n)[action_mask])
                else:
                    self_preds = self.self_nets[unit.__class__.__name__](torch.tensor(env.self_observation(unit)))
                    preds = tile_preds[unit][i, unit.y, unit.x] + self_preds

                    if unit.team == 0:
                        preds[~action_mask] = -1e8
                        action = torch.argmax(preds).item()
                    else:
                        preds[~action_mask] = 1e8
                        action = torch.argmin(preds).item()

                actions.append(action)
                env.step(unit, action)

            reward = env.get_team_reward()
            yield actions, reward

    def forward(
        self,
        env: BattlecodeEnv,
        actions: list[int],
    ) -> tuple[torch.Tensor, BattlecodeEnv]:
        """Computes Q-values given actions"""
        x = self.state_conv(torch.tensor(env.state()))
        z = self.squeeze_and_excite(x)
        global_q = self.bias_head(z).squeeze()
        weights = F.softplus(
            self.weight_head(torch.sigmoid(z)[:, None, None] * x).squeeze(0)
        )

        obs = torch.tensor(env.observations())
        preds = {
            cls: self.local_qnets[cls.__name__](obs)
            for cls in (Miner, Builder, Soldier, Sage, Archon)
        }
        # TODO fix ally/opponent observations
        for i, (unit, action_mask) in enumerate(env.iter_agents()):
            local_q = preds[unit.__class__][actions[i], unit.y, unit.x]
            global_q += weights[unit.y, unit.x] * local_q  # linear QMIX
            env.step(unit, actions[i])

        return global_q, env


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
        epsilon_decrease_steps=64_000,
        epsilon_max=1,
        epsilon_min=0.01,
        logging_interval=1000,
        seed=None,
        verbose=False,
    ):
        assert epsilon_decrease_steps <= buffer_size

        self.rng = random.Random(seed)
        self.logging_interval = logging_interval
        self.logger = Logger()
        self.verbose = verbose

        self.env = env
        self.env.reset(seed=self.rng.randrange(2**32))
        self.discount_factor = discount_factor

        self.buffer_size = buffer_size
        self.steps_per_update = steps_per_update
        self.minibatch_size = minibatch_size

        self.model = QMixAgents(logger=self.logger, seed=self.rng.randrange(2**32))
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.target_model = copy.deepcopy(self.model)
        self.target_update_period = target_update_period
        self.tau = target_update_polyak

        self.epsilon_decrease_steps = epsilon_decrease_steps
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min

        self.replay_buffer = []
        self.t = 0

    def curr_epsilon(self):
        if self.epsilon_decrease_steps == 0:
            return self.epsilon_min
        return max(
            self.epsilon_min,
            self.epsilon_min
            + (self.epsilon_max - self.epsilon_min)
            * (1 - self.t / self.epsilon_decrease_steps),
        )

    def store(self, state, actions, reward):
        new_transition = Transition(state, actions, reward)
        if len(self.replay_buffer) < self.buffer_size:
            self.replay_buffer.append(new_transition)
        else:
            self.replay_buffer[self.t % self.buffer_size] = new_transition

    def gather_random_experience(self, num_steps):
        iterator = (
            tqdm.trange(num_steps, desc="random experience")
            if self.verbose
            else range(num_steps)
        )
        for _ in iterator:
            curr_state = copy.deepcopy(self.env)
            action, reward = self.model.explore_step([self.env], epsilon=1)
            self.store(curr_state, action, reward)
            if self.env.done:
                self.env.reset(seed=self.rng.randrange(2**32))

    def learn(self, num_steps):
        iterator = (
            tqdm.trange(num_steps, desc="training")
            if self.verbose
            else range(num_steps)
        )
        for _ in iterator:
            curr_state = copy.deepcopy(self.env)
            actions, reward = self.model.explore_step(self.env, self.curr_epsilon())
            self.store(curr_state, actions, reward)
            if self.env.done:
                # self.logger.push(self.env.episode_metrics)
                self.env.reset(seed=self.rng.randrange(2**32))

            if (self.t + 1) % self.steps_per_update == 0:
                loss = torch.tensor(0.0)
                for _ in range(self.minibatch_size):
                    transition = self.rng.choice(self.replay_buffer)
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

            if (self.t + 1) % self.target_update_period == 0:
                with torch.no_grad():
                    for p1, p2 in zip(
                        self.model.parameters(), self.target_model.parameters()
                    ):
                        p2[:] = self.tau * p1 + (1 - self.tau) * p2

            if (self.t + 1) % self.logging_interval == 0:
                self.logger.step()
                self.logger.generate_plots()

            self.t += 1

    def evaluate(self, num_episodes=10, epsilon=0):
        eval_env = copy.deepcopy(self.env)
        iterator = (
            tqdm.trange(num_episodes, desc="evaluation")
            if self.verbose
            else range(num_episodes)
        )
        for _ in iterator:
            eval_env.reset(self.rng.randrange(2**32))
            while not eval_env.done:
                self.model.explore_step(eval_env, epsilon)
            self.logger.push(
                {"eval_" + k: v for k, v in eval_env.episode_metrics.items()}
            )
        self.logger.step()
        self.logger.generate_plots()

    def evaluate_with_render(self, epsilon=0):
        from .env.rendering import Renderer

        renderer = Renderer()
        eval_env = copy.deepcopy(self.env)
        eval_env.reset(self.rng.randrange(2**32))

        with tqdm.tqdm(total=2000) as pbar:
            while not eval_env.done:
                renderer.render(eval_env)
                self.model.explore_step(eval_env, epsilon)
                pbar.update(1)
