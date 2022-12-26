import glob
import os
import random
import typing
from collections import defaultdict

import numpy as np

from ..utils import DIRECTIONS, within_radius
from .entities import (
    Archon,
    Builder,
    Building,
    Entity,
    Laboratory,
    Miner,
    Sage,
    Soldier,
    Watchtower,
)


class BattlecodeEnv:
    def __init__(
        self,
        map_selection: str | None = None,
        max_episode_length: int = 2000,
        # augment_obs: bool = False,  # TODO
        gold_reward_shaping_factor: float = 5.0,
        reward_shaping_depends_hp: bool = True,
        reward_scaling_factor: float = 0.1,
    ):
        self.map_selection = map_selection
        self.max_episode_length = max_episode_length
        self.map_name = None
        # self.augment_obs = augment_obs
        self.gold_reward_shaping_factor = gold_reward_shaping_factor
        self.reward_shaping_depends_hp = reward_shaping_depends_hp
        self.reward_scaling_factor = reward_scaling_factor
        self.rng = None

        self.height = None
        self.width = None
        self.rubble = None
        self.lead = None
        self.gold = None
        self.t = None
        self.lead_banks = None
        self.gold_banks = None
        self.units: list[Entity] = None
        self.pos_map = None
        self.archon_counts = None
        self.done = None
        self.curr_idx = None
        self.prev_value_differential = None
        self.cached_obs = None
        self.episode_metrics = None

    def in_bounds(self, y, x):
        return 0 <= y < self.height and 0 <= x < self.width

    def nearby_bots(self, y, x, radsq, team=None, prev_move: int = 0):
        for (dy, dx) in within_radius(radsq, prev_move=prev_move):
            if (y + dy, x + dx) in self.pos_map and (
                (yld := self.pos_map[(y + dy, x + dx)]).team == team or team is None
            ):
                yield yld

    def reset(self, seed=None):
        self.rng = random.Random(seed)

        filenames = glob.glob("Re_Battlecode22/maps/data/*.npz")
        filenames = [a for a in filenames if "maptestsmall" not in a]
        selected_file = (
            self.rng.choice(filenames)
            if self.map_selection is None
            else f"Re_Battlecode22/maps/data/{self.map_selection}.npz"
        )
        data = np.load(selected_file)
        self.map_name = os.path.splitext(os.path.basename(selected_file))[0]

        self.rubble = data["rubble"].astype(np.uint8)
        self.lead = data["lead"].astype(np.uint8)
        self.gold = np.zeros_like(self.lead)
        self.height = self.rubble.shape[0]
        self.width = self.rubble.shape[1]

        self.t = 0
        self.lead_banks = [200, 200]
        self.gold_banks = [0, 0]
        self.units = []
        self.pos_map = {}
        for row0, row1 in zip(data["team0_archon_pos"], data["team1_archon_pos"]):
            self.units.append(Archon(y=row0[0], x=row0[1], team=0))
            self.units.append(Archon(y=row1[0], x=row1[1], team=1))
        for unit in self.units:
            self.pos_map[(unit.y, unit.x)] = unit
        self.archon_counts = [len(self.units) // 2, len(self.units) // 2]
        self.prev_value_differential = 0
        self.episode_metrics = {
            "lead_mined": 0,
            "dmg_dealt": 0,
            "units_killed": 0,
            "spawned_miner": 0,
            "spawned_builder": 0,
            "spawned_soldier": 0,
            "spawned_sage": 0,
            "spawned_laboratory": 0,
        }
        self.done = False

    # TODO: fix archon observations
    def observations(self):
        if self.cached_obs is not None:
            return self.cached_obs

        counts = defaultdict(lambda: [0, 0])
        for unit in self.units:
            counts[unit.__class__.__name__.lower()][unit.team] += 1

        i = 0
        j = 0
        archon_obs = np.zeros((2, 11), dtype=np.float32)
        archon_obs

        tile_obs = np.zeros((7, self.height, self.width), dtype=np.float32)
        tile_obs[0] = self.rubble / 100
        for unit in self.units:
            tile_obs[1 + unit.team, unit.y, unit.x] = np.log1p(unit.curr_hp) / 2
            tile_obs[3 + unit.team, unit.y, unit.x] = 1 if isinstance(unit, Building) else 0
        tile_obs[5] = self.lead / 64
        tile_obs[6] = self.gold / 16
        self.cached_obs = tile_obs
        return tile_obs

    def state(self):
        # timestep, lead bank, gold bank for both teams, map size, x, y coordinate (9)
        # rubble, lead, gold  (3)
        # ally type one-hot, HP, move_cd, act_cd (6 + 3 = 9)
        # same for opponent (9)
        # SUM: 9 + 3 + 9 + 9 = 30
        symmetry = 0  # TODO
        swapax = 0

        ret = np.zeros(
            (30, self.rubble.shape[0], self.rubble.shape[1]),
            dtype=np.float32,
        )

        # global information:
        # timestep, lead+gold banks, map size.
        ret[0] = self.t / 2000
        ret[1:3] = (
            np.array(
                self.lead_banks if symmetry < 8 else self.lead_banks[::-1]
            ).reshape((-1, 1, 1))
            / 256
        )
        ret[3:5] = (
            np.array(
                self.gold_banks if symmetry < 8 else self.gold_banks[::-1]
            ).reshape((-1, 1, 1))
            / 64
        )
        ret[5:7] = (
            np.array(
                self.rubble.shape if not swapax else self.rubble.shape[::-1]
            ).reshape((-1, 1, 1))
            / 60
        )

        # each cell stores own y and x position relative to origin
        ret[7 + swapax] = (
            np.arange(self.rubble.shape[0]) - self.rubble.shape[0] / 2 + 0.5
        ).reshape((-1, 1)) / 30
        ret[8 - swapax] = (
            np.arange(self.rubble.shape[1]) - self.rubble.shape[1] / 2 + 0.5
        ).reshape((1, -1)) / 30

        # terrain: rubble, lead, gold
        ret[9] = self.rubble / 100
        ret[10] = self.lead / 128
        ret[11] = self.gold / 32

        # units: ally HP, type one-hot, move_cd, act_cd
        unit_type_map = {
            Miner: 0,
            Builder: 1,
            Soldier: 2,
            Sage: 3,
            Archon: 4,
            Laboratory: 5,
        }
        for y in range(self.rubble.shape[0]):
            for x in range(self.rubble.shape[1]):
                if (y, x) not in self.pos_map:
                    continue
                unit: Entity = self.pos_map[(y, x)]
                unit_type_id = unit_type_map[unit.__class__]
                ut = unit.team if symmetry < 8 else 1 - unit.team
                ret[12 + 2 * unit_type_id + ut, y, x] = 1
                ret[24 + ut, y, x] = unit.curr_hp / 512
                ret[26 + ut, y, x] = unit.move_cd / 100
                ret[28 + ut, y, x] = unit.act_cd / 100
        return ret

    def legal_action_mask(self, bot):
        ret = np.zeros(bot.action_space.n, dtype=bool)

        ret[0] = True
        if not isinstance(bot, Building):
            # move actions: [0, 8] inclusive for all bots
            for i, (dy, dx) in enumerate(DIRECTIONS):  # TODO action mask symmetry
                if i == 0:
                    continue
                elif (
                    bot.move_cd < 10
                    and self.in_bounds(bot.y + dy, bot.x + dx)
                    and (bot.y + dy, bot.x + dx) not in self.pos_map
                ):
                    ret[i] = True

        adj_available = False
        for dy, dx in DIRECTIONS[1:]:
            if (
                self.in_bounds(bot.y + dy, bot.x + dx)
                and (bot.y + dy, bot.x + dx) not in self.pos_map
            ):
                adj_available = True
                break

        if isinstance(bot, Builder):
            assert ret.shape == (11,)
            if (
                self.lead_banks[bot.team] >= Laboratory.lead_value
                and bot.act_cd < 10
                and adj_available
            ):
                ret[9] = True
            if self.lead[bot.y, bot.x] == 0:
                ret[10] = True  # disintegrate for lead farm
        elif isinstance(bot, (Soldier, Sage)):
            assert ret.shape == (9,)
            pass  # TODO attack actions, eventually
        elif isinstance(bot, Archon):
            assert ret.shape == (4,)
            # TODO support archon movement (some maps put archons on rubble)
            # [repair / idle, spawn miner, spawn builder, spawn combat]
            if bot.act_cd < 10:
                ret[1] = adj_available and (
                    self.lead_banks[bot.team] >= Miner.lead_value
                )
                ret[2] = adj_available and (
                    self.lead_banks[bot.team] >= Builder.lead_value
                )
                ret[3] = adj_available and (
                    (self.lead_banks[bot.team] >= Soldier.lead_value)
                    | (self.gold_banks[bot.team] >= Sage.gold_value)
                )
        elif isinstance(bot, Laboratory):
            assert ret.shape == (2,)

            if bot.act_cd < 10:
                lead_cost = bot.lead_ratio(
                    len(list(self.nearby_bots(bot.y, bot.x, bot.vis_rad, bot.team))) - 1
                )
                ret[1] = self.lead_banks[bot.team] >= lead_cost
        elif isinstance(bot, Watchtower):
            return NotImplementedError  # TODO

        return ret

    def move(self, bot: Entity, action: int):
        assert bot.move_cd < 10
        dy, dx = DIRECTIONS[action]
        assert (bot.y + dy, bot.x + dx) not in self.pos_map
        assert self.in_bounds(bot.y + dy, bot.x + dx)

        self.cached_obs = None
        del self.pos_map[(bot.y, bot.x)]
        bot.y += dy
        bot.x += dx
        self.pos_map[(bot.y, bot.x)] = bot
        bot.add_move_cost(self.rubble[bot.y, bot.x])

    def attack(self, bot: Entity, target: Entity):
        if isinstance(bot, (Archon, Builder)):
            assert bot.team == target.team
        elif isinstance(bot, (Soldier, Sage, Watchtower)):
            assert bot.team == 1 - target.team
        else:
            raise TypeError

        assert bot.act_cd < 10

        self.cached_obs = None
        bot.add_act_cost(self.rubble[bot.y, bot.x])

        old_hp = target.curr_hp
        target.curr_hp = min(target.curr_hp - bot.dmg, target.max_hp)
        dmg_dealt = old_hp - max(target.curr_hp, 0)
        killed = target.curr_hp <= 0
        self.episode_metrics["dmg_dealt"] += dmg_dealt
        if killed:
            if isinstance(target, Archon):
                self.archon_counts[target.team] -= 1

            self.lead[target.y, target.x] += int(0.2 * target.lead_value)
            self.gold[target.y, target.x] += int(0.2 * target.gold_value)

            idx = self.units.index(target)
            del self.units[idx]
            del self.pos_map[(target.y, target.x)]
            if idx <= self.curr_idx:
                self.curr_idx -= 1

            self.episode_metrics["units_killed"] += 1

        return dmg_dealt, killed

    def disintegrate(self, bot):
        assert isinstance(bot, Builder)
        assert self.lead[bot.y, bot.x] == 0  # heuristic constraint

        self.cached_obs = None
        self.lead[bot.y, bot.x] += int(0.2 * bot.lead_value)
        idx = self.units.index(bot)

        del self.units[idx]
        del self.pos_map[(bot.y, bot.x)]
        if idx <= self.curr_idx:
            self.curr_idx -= 1

    def spawn(self, unit_class, pos, team):
        assert pos not in self.pos_map
        assert self.in_bounds(pos[0], pos[1])
        assert isinstance(pos, tuple)
        assert (
            self.lead_banks[team] >= unit_class.lead_value
        ), f"{self.lead_banks[team]} {unit_class.__name__}"
        assert self.gold_banks[team] >= unit_class.gold_value

        self.cached_obs = None
        new_unit = unit_class(y=pos[0], x=pos[1], team=team)
        self.units.append(new_unit)
        self.pos_map[pos] = new_unit
        self.lead_banks[team] -= new_unit.lead_value
        self.gold_banks[team] -= new_unit.gold_value

        self.episode_metrics[f"spawned_{unit_class.__name__.lower()}"] += 1

    def step(self, bot, action):
        if not isinstance(bot, Building) and 1 <= action <= 8:
            self.move(bot, action)

        # auto-mining
        if isinstance(bot, Miner) and bot.act_cd < 10:
            available_lead = [
                pos
                for (dy, dx) in within_radius(bot.act_rad, prev_move=action)
                if self.in_bounds(bot.y + dy, bot.x + dx)
                and self.lead[pos := (bot.y + dy, bot.x + dx)] >= 1
            ]
            while available_lead and bot.act_cd < 10:
                self.cached_obs = None
                selected = self.rng.choice(available_lead)
                self.lead[selected] -= 1
                self.lead_banks[bot.team] += 1
                if self.lead[selected] <= 1:
                    available_lead.remove(selected)
                bot.add_act_cost(self.rubble[bot.y, bot.x])
                self.episode_metrics[f"lead_mined"] += 1

        # auto-repair
        if (
            (isinstance(bot, Builder) and 0 <= action <= 8)
            or (isinstance(bot, Archon) and action == 0)
        ) and bot.act_cd < 10:
            targets = list(
                self.nearby_bots(bot.y, bot.x, bot.act_rad, bot.team, prev_move=action)
            )
            targets = [
                target
                for target in targets
                if target.curr_hp < target.max_hp and target != bot
            ]
            if targets:
                self.cached_obs = None
                selected = self.rng.choice(targets)
                self.attack(bot, selected)

        # lab construction
        if isinstance(bot, Builder) and action == 9:
            assert self.lead_banks[bot.team] >= Laboratory.lead_value
            available_pos = [
                (bot.y + dy, bot.x + dx)
                for (dy, dx) in DIRECTIONS[1:]
                if (bot.y + dy, bot.x + dx) not in self.pos_map
                and self.in_bounds(bot.y + dy, bot.x + dx)
            ]
            assert len(available_pos) > 0
            chosen_pos = self.rng.choice(available_pos)
            self.spawn(Laboratory, chosen_pos, bot.team)
            bot.add_act_cost(self.rubble[bot.y, bot.x])

        # disintegrate
        if isinstance(bot, Builder) and action == 10:
            self.disintegrate(bot)

        # auto-attacking
        if isinstance(bot, (Soldier, Sage)) and bot.act_cd < 10:
            nearby_enemies = list(
                self.nearby_bots(
                    bot.y, bot.x, bot.act_rad, 1 - bot.team, prev_move=action
                )
            )
            if nearby_enemies:

                def score(enemy: Entity):
                    num_shots = enemy.curr_hp // bot.dmg
                    return -num_shots, enemy.curr_hp, -bot.distsq(enemy)

                best_score = max(score(enemy) for enemy in nearby_enemies)
                chosen_enemy = self.rng.choice(
                    [enemy for enemy in nearby_enemies if score(enemy) == best_score]
                )
                self.attack(bot, chosen_enemy)

        # droid creation
        if isinstance(bot, Archon) and action != 0:
            assert bot.act_cd < 10
            map_center = (self.height // 2, self.width // 2)
            all_spawn_pos = []
            good_spawn_pos = []
            for dy, dx in DIRECTIONS[1:]:
                y = bot.y + dy
                x = bot.x + dx
                if (y, x) not in self.pos_map and self.in_bounds(y, x):
                    all_spawn_pos.append((y, x))
                    if abs(y - map_center[0]) + abs(x - map_center[1]) < abs(
                        bot.y - map_center[0]
                    ) + abs(bot.x - map_center[1]):
                        good_spawn_pos.append((y, x))

            assert all_spawn_pos
            chosen_pos = self.rng.choice(
                good_spawn_pos if good_spawn_pos else all_spawn_pos
            )

            unit_class_map = {
                1: Miner,
                2: Builder,
                3: Sage if self.gold_banks[bot.team] >= Sage.gold_value else Soldier,
            }
            new_unit_class = unit_class_map[action]
            self.spawn(new_unit_class, chosen_pos, bot.team)
            bot.add_act_cost(self.rubble[bot.y, bot.x])

        # auto-transmute
        if isinstance(bot, Laboratory) and action == 1:
            lead_cost = bot.lead_ratio(
                len(list(self.nearby_bots(bot.y, bot.x, bot.vis_rad, bot.team))) - 1
            )
            assert (
                self.lead_banks[bot.team] >= lead_cost
            ), f"{self.lead_banks[bot.team]}, {lead_cost}"
            self.lead_banks[bot.team] -= lead_cost
            self.gold_banks[bot.team] += 1
            bot.add_act_cost(self.rubble[(bot.y, bot.x)])

        if min(self.archon_counts) == 0:
            self.done = True

    def iter_agents(
        self,
    ) -> typing.Generator[tuple[Entity, np.ndarray], None, None]:
        """Yields Entity objects and action masks"""
        assert self.curr_idx is None
        assert self.t < self.max_episode_length

        self.curr_idx = 0
        while self.curr_idx < len(self.units):
            bot = self.units[self.curr_idx]

            # temporary heuristic behavior, not policy-controlled
            if isinstance(bot, Laboratory):
                if self.legal_action_mask(bot)[1]:
                    self.step(bot, action=1)
                else:
                    self.step(bot, action=0)
                self.curr_idx += 1
                continue

            assert not isinstance(bot, Watchtower)  # TODO

            yield bot, self.legal_action_mask(bot)

            self.curr_idx += 1
        self.curr_idx = None
        self.cached_obs = None

        for bot in self.units:
            bot.move_cd = max(0, bot.move_cd - 10)
            bot.act_cd = max(0, bot.act_cd - 10)

        self.t += 1
        if self.t == self.max_episode_length:
            self.done = True
        else:
            self.lead_banks[0] += 2
            self.lead_banks[1] += 2
            if self.t % 20 == 0:
                self.lead[self.lead > 0] += 5

    def winner(self):
        if self.archon_counts[0] != self.archon_counts[1]:
            return 0 if self.archon_counts[0] > self.archon_counts[1] else 1
        lead_values = self.lead_banks.copy()
        gold_values = self.gold_banks.copy()
        for unit in self.units:
            lead_values[unit.team] += unit.lead_value
            gold_values[unit.team] += unit.gold_value
        if gold_values[0] != gold_values[1]:
            return 0 if gold_values[0] > gold_values[1] else 1
        return 0 if lead_values[0] > lead_values[1] else 1

    # reward for team 0 shared reward (zero sum --> equal to negative team 1 reward)
    def get_team_reward(self):
        assert self.curr_idx is None  # not in middle of units pass

        value_differential = self.lead_banks[0] - self.lead_banks[1]
        value_differential += self.gold_reward_shaping_factor * (
            self.gold_banks[0] - self.gold_banks[1]
        )

        for unit in self.units:
            unit_value = (
                unit.lead_value + self.gold_reward_shaping_factor * unit.gold_value
            )
            if self.reward_shaping_depends_hp:
                unit_value *= unit.curr_hp / unit.max_hp

            if unit.team == 0:
                value_differential += unit_value
            else:
                value_differential -= unit_value

        ret = (
            value_differential - self.prev_value_differential
        ) * self.reward_scaling_factor
        self.prev_value_differential = value_differential

        # TODO win reward

        return ret
