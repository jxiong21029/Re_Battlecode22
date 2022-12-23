import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from .env import Archon, BattlecodeEnv, Builder, Miner, Sage, Soldier
from .env.rendering import Renderer
from .utils import Logger


@dataclass
class Transition:
    env_state: BattlecodeEnv
    actions: list[int]
    reward: float


class Model(nn.Module):
    def __init__(
        self,
        augment_inputs=True,
        augment_targets=True,
        target_augmentations=4,
        logger=None,
        seed=None,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
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

    def explore_step(self, env: BattlecodeEnv, epsilon=0.01) -> tuple[list[int], float]:
        actions = []
        with torch.no_grad():
            for bot, obs, action_mask in env.iter_agents():
                if self.rng.random() < epsilon:
                    action = self.rng.choice(
                        np.arange(action_mask.shape[0])[action_mask]
                    )
                else:
                    action_qs = self.utility_nets[bot.__class__.__name__](
                        torch.tensor(obs)
                    )
                    if bot.team == 0:
                        action_qs[~action_mask] = -1e8
                        action = torch.argmax(action_qs).item()
                    else:
                        assert bot.team == 1
                        action_qs[~action_mask] = 1e8
                        action = torch.argmin(action_qs).item()
                actions.append(action)
                env.step(bot, action)

        reward = env.get_team_reward()
        env.push_round_metrics(self.logger)
        return actions, reward

    def forward(
        self, env_state: BattlecodeEnv, actions: list[int]
    ) -> tuple[torch.Tensor, BattlecodeEnv]:
        x = self.conv1(torch.tensor(env_state.global_observation()))
        x = self.squeeze_and_excite(x)[:, None, None] * x
        weights = F.softplus(self.head(x).squeeze(0))  # (1, H, W), then (H, W)
        assert weights.shape == env_state.rubble.shape

        global_q = torch.tensor(0.0)
        curr_env = copy.deepcopy(env_state)
        for i, (bot, obs, action_mask) in enumerate(curr_env.iter_agents()):
            action_qs = self.utility_nets[bot.__class__.__name__](torch.tensor(obs))

            local_q = action_qs[actions[i]]
            global_q += weights[bot.y, bot.x] * local_q  # linear QMIX
            curr_env.step(bot, actions[i])

        return global_q, curr_env


class Trainer:
    def __init__(
        self,
        env: BattlecodeEnv,
        discount_factor=0.99,
        buffer_size=128_000,
        steps_per_update=4,
        minibatch_size=32,
        lr=1e-4,
        target_update_period=8000,
        target_update_polyak=1,
        pre_learning_steps=8_000,
        epsilon_decrease_steps=64_000,
        epsilon_max=1,
        epsilon_min=0.01,
        logging_interval=1000,
        verbose=False,
        seed=None,
    ):
        assert pre_learning_steps + epsilon_decrease_steps <= buffer_size

        self.rng = np.random.default_rng(seed)
        self.logging_interval = logging_interval
        self.logger = Logger()
        self.verbose = verbose

        self.env = env
        self.env.reset()
        self.discount_factor = discount_factor

        self.buffer_size = buffer_size
        self.steps_per_update = steps_per_update
        self.minibatch_size = minibatch_size

        self.model = Model(seed=self.rng.integers(2**32), logger=self.logger)
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
                    transition = self.rng.choice(self.replay_buffer)
                    y, next_state = self.model(transition.env_state, transition.actions)

                    if next_state.done:
                        target = transition.reward
                    else:
                        next_actions, _ = self.model.explore_step(
                            copy.deepcopy(next_state), epsilon=0
                        )
                        next_q, _ = self.target_model(next_state, actions=next_actions)
                        target = transition.reward + self.discount_factor * next_q

                    loss += ((y - target) ** 2) / self.minibatch_size

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

            if (
                self.t < self.pre_learning_steps
                and (self.t + 1) % self.logging_interval == 0
            ) or (
                self.t >= self.pre_learning_steps
                and (self.t - self.pre_learning_steps + 1) % self.logging_interval == 0
            ):
                self.logger.step()
                self.logger.generate_plots()

            self.t += 1

    def eval_with_render(self, eps=0):
        renderer = Renderer()
        eval_env = copy.deepcopy(self.env)
        eval_env.reset()

        with tqdm.tqdm(total=2000) as pbar:
            while not eval_env.done:
                renderer.render(eval_env)
                self.model.explore_step(eval_env, eps)
                pbar.update(1)
