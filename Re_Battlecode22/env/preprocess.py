from collections import defaultdict

import numpy as np
import torch

from ..utils import DIRECTIONS, within_radius
from .engine import BattlecodeEngine
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

CLS_MAP = {
    Miner: 0,
    Builder: 1,
    Soldier: 2,
    Sage: 3,
    Archon: 4,
    Laboratory: 5,
}


@profile
def observe(engine: BattlecodeEngine, for_units=None):
    units = engine.units if for_units is None else for_units
    hp = np.zeros((2, engine.height, engine.width), dtype=np.float32)
    is_building = np.zeros((2, engine.height, engine.width), dtype=np.float32)

    all_unit_counts = {cls: [0, 0] for cls in CLS_MAP.keys()}
    for i, unit in enumerate(engine.units):
        all_unit_counts[unit.__class__][unit.team] += 1
        hp[unit.team, unit.y, unit.x] = np.log1p(unit.curr_hp)
        if isinstance(unit, Building):
            is_building[unit.team, unit.y, unit.x] = 1

    unit_counts = {cls: 0 for cls in CLS_MAP.keys()}
    for unit in units:
        unit_counts[unit.__class__] += 1
    ret = {
        cls: np.zeros(
            (unit_counts[cls], cls.observation_space.shape[0]),
            dtype=np.float32,
        )
        for cls in CLS_MAP.keys()
        if cls != Laboratory
    }

    # map height, map width, timestep, lead bank, gold bank, 4 unit counts = 11
    archon_obs = ret[Archon]
    archon_obs[:, 0:2] = engine.rubble.shape
    archon_obs[:, 2] = engine.t / 2000
    i = 0
    for unit in units:
        if isinstance(unit, Archon):
            archon_obs[i, 3:11] = [
                (unit.y - engine.height / 2 + 0.5) / 30,
                (unit.x - engine.width / 2 + 0.5) / 30,
                engine.lead_banks[unit.team] / 128,
                engine.gold_banks[unit.team] / 32,
                all_unit_counts[Miner][unit.team] / 16,
                all_unit_counts[Builder][unit.team] / 16,
                all_unit_counts[Soldier][unit.team] / 16,
                all_unit_counts[Sage][unit.team] / 16,
            ]
            i += 1

    cls_coords = {cls: [] for cls in CLS_MAP.keys()}
    cls_teams = {cls: [] for cls in CLS_MAP.keys()}
    for unit in units:
        cls_coords[unit.__class__].append(np.array([unit.y, unit.x]))
        cls_teams[unit.__class__].append(unit.team)
    cls_coords = {
        cls: np.stack(coords, axis=0)
        for cls, coords in cls_coords.items()
        if len(coords) > 0
    }
    cls_teams = {
        cls: np.array(teams) for cls, teams in cls_teams.items() if len(teams) > 0
    }

    for cls in (Miner, Builder, Soldier, Sage):
        if cls not in cls_teams:
            continue
        cls_obs = ret[cls]

        for i, (dy, dx) in enumerate(within_radius(cls.vis_rad)):
            shifted = cls_coords[cls] + np.array([dy, dx])
            in_bounds_mask = (
                (shifted[:, 0] >= 0)
                & (shifted[:, 0] < engine.height)
                & (shifted[:, 1] >= 0)
                & (shifted[:, 1] < engine.width)
            )
            shifted = shifted[in_bounds_mask]
            tm = cls_teams[cls][in_bounds_mask]

            if cls == Miner:
                cls_obs[in_bounds_mask, 7 * i : 7 * i + 7] = np.stack(
                    [
                        engine.rubble[shifted[:, 0], shifted[:, 1]] / 100,
                        hp[tm, shifted[:, 0], shifted[:, 1]],
                        hp[1 - tm, shifted[:, 0], shifted[:, 1]],
                        is_building[tm, shifted[:, 0], shifted[:, 1]],
                        is_building[1 - tm, shifted[:, 0], shifted[:, 1]],
                        engine.lead[shifted[:, 0], shifted[:, 1]] / 64,
                        engine.gold[shifted[:, 0], shifted[:, 1]] / 16,
                    ],
                    axis=1,
                )
            else:
                cls_obs[in_bounds_mask, 5 * i : 5 * i + 5] = np.stack(
                    [
                        engine.rubble[shifted[:, 0], shifted[:, 1]] / 100,
                        hp[tm, shifted[:, 0], shifted[:, 1]],
                        hp[1 - tm, shifted[:, 0], shifted[:, 1]],
                        is_building[tm, shifted[:, 0], shifted[:, 1]],
                        is_building[1 - tm, shifted[:, 0], shifted[:, 1]],
                    ],
                    axis=1,
                )

    return ret


def trial(seed):
    import random
    import timeit

    random.seed(seed)
    global engine
    engine = BattlecodeEngine("eckleburg")
    engine.reset()
    engine.lead_banks = [1000, 1000]
    engine.gold_banks = [100, 100]
    for _ in range(100):
        for bot, act_mask in engine.iter_agents():
            selected_action = random.choice(np.arange(bot.action_space.n)[act_mask])
            engine.step(bot, selected_action)
    print(
        f"unit count: {len(engine.units)}, "
        f"unique types: {len(set(unit.__class__ for unit in engine.units))}"
    )
    print(
        "elapsed (1000 iter, sec):",
        timeit.timeit(
            "observe(engine, for_units=engine.units[-5:])",
            number=1000,
            globals=globals(),
        ),
    )


    # def global_observation(self, symmetry=0):
    #     # timestep, lead bank, gold bank for both teams, map size, x, y coordinate (9)
    #     # rubble, lead, gold  (3)
    #     # ally type one-hot, HP, move_cd, act_cd (6 + 3 = 9)
    #     # same for opponent (9)
    #     # SUM: 9 + 3 + 9 + 9 = 30
    #
    #     swapax = int(symmetry % 8 >= 4)
    #     ret = np.zeros(
    #         (30, self.rubble.shape[0], self.rubble.shape[1]),
    #         dtype=np.float32,
    #     )
    #
    #     # global information:
    #     # timestep, lead+gold banks, map size.
    #     ret[0] = self.t / 2000
    #     ret[1:3] = (
    #         np.array(
    #             self.lead_banks if symmetry < 8 else self.lead_banks[::-1]
    #         ).reshape((-1, 1, 1))
    #         / 256
    #     )
    #     ret[3:5] = (
    #         np.array(
    #             self.gold_banks if symmetry < 8 else self.gold_banks[::-1]
    #         ).reshape((-1, 1, 1))
    #         / 64
    #     )
    #     ret[5:7] = (
    #         np.array(
    #             self.rubble.shape if not swapax else self.rubble.shape[::-1]
    #         ).reshape((-1, 1, 1))
    #         / 60
    #     )
    #
    #     # each cell stores own y and x position relative to origin
    #     ret[7 + swapax] = (
    #         np.arange(self.rubble.shape[0]) - self.rubble.shape[0] / 2 + 0.5
    #     ).reshape((-1, 1)) / 30
    #     ret[8 - swapax] = (
    #         np.arange(self.rubble.shape[1]) - self.rubble.shape[1] / 2 + 0.5
    #     ).reshape((1, -1)) / 30
    #
    #     # terrain: rubble, lead, gold
    #     ret[9] = self.rubble / 100
    #     ret[10] = self.lead / 128
    #     ret[11] = self.gold / 32
    #
    #     # units: ally HP, type one-hot, move_cd, act_cd
    #     unit_type_map = {
    #         Miner: 0,
    #         Builder: 1,
    #         Soldier: 2,
    #         Sage: 3,
    #         Archon: 4,
    #         Laboratory: 5,
    #     }
    #     for y in range(self.rubble.shape[0]):
    #         for x in range(self.rubble.shape[1]):
    #             if (y, x) not in self.pos_map:
    #                 continue
    #             unit: Entity = self.pos_map[(y, x)]
    #             unit_type_id = unit_type_map[unit.__class__]
    #             ut = unit.team if symmetry < 8 else 1 - unit.team
    #             ret[12 + 2 * unit_type_id + ut, y, x] = 1
    #             ret[24 + ut, y, x] = unit.curr_hp / 512
    #             ret[26 + ut, y, x] = unit.move_cd / 100
    #             ret[28 + ut, y, x] = unit.act_cd / 100
    #     return global_symmetry_transform(ret, symmetry)
    #
    # def legal_action_mask(self, bot):
    #     ret = np.zeros(bot.action_space.n, dtype=bool)
    #
    #     ret[0] = True
    #     if not isinstance(bot, Building):
    #         # move actions: [0, 8] inclusive for all bots
    #         for i, (dy, dx) in enumerate(DIRECTIONS):  # TODO action mask symmetry
    #             if i == 0:
    #                 continue
    #             elif (
    #                 bot.move_cd < 10
    #                 and self.in_bounds(bot.y + dy, bot.x + dx)
    #                 and (bot.y + dy, bot.x + dx) not in self.pos_map
    #             ):
    #                 ret[i] = True
    #
    #     adj_available = False
    #     for dy, dx in DIRECTIONS[1:]:
    #         if (
    #             self.in_bounds(bot.y + dy, bot.x + dx)
    #             and (bot.y + dy, bot.x + dx) not in self.pos_map
    #         ):
    #             adj_available = True
    #             break
    #
    #     if isinstance(bot, Builder):
    #         assert ret.shape == (11,)
    #         if (
    #             self.lead_banks[bot.team] >= Laboratory.lead_value
    #             and bot.act_cd < 10
    #             and adj_available
    #         ):
    #             ret[9] = True
    #         if self.lead[bot.y, bot.x] == 0:
    #             ret[10] = True  # disintegrate for lead farm
    #     elif isinstance(bot, (Soldier, Sage)):
    #         assert ret.shape == (9,)
    #         pass  # TODO attack actions, eventually
    #     elif isinstance(bot, Archon):
    #         assert ret.shape == (4,)
    #         # TODO support archon movement (some maps put archons on rubble)
    #         # [repair / idle, spawn miner, spawn builder, spawn combat]
    #         if bot.act_cd < 10:
    #             ret[1] = adj_available and (
    #                 self.lead_banks[bot.team] >= Miner.lead_value
    #             )
    #             ret[2] = adj_available and (
    #                 self.lead_banks[bot.team] >= Builder.lead_value
    #             )
    #             ret[3] = adj_available and (
    #                 (self.lead_banks[bot.team] >= Soldier.lead_value)
    #                 | (self.gold_banks[bot.team] >= Sage.gold_value)
    #             )
    #     elif isinstance(bot, Laboratory):
    #         assert ret.shape == (2,)
    #
    #         if bot.act_cd < 10:
    #             lead_cost = bot.lead_ratio(
    #                 len(list(self.nearby_bots(bot.y, bot.x, bot.vis_rad, bot.team)))
    #             )
    #             ret[1] = lead_cost >= self.lead_banks[bot.team]
    #     elif isinstance(bot, Watchtower):
    #         return NotImplementedError  # TODO
    #
    #     return ret
    #
    # def process_move(self, bot: Entity, action: int):
    #     assert bot.move_cd < 10
    #     dy, dx = DIRECTIONS[action]
    #     assert (bot.y + dy, bot.x + dx) not in self.pos_map
    #     assert self.in_bounds(bot.y + dy, bot.x + dx)
    #     del self.pos_map[(bot.y, bot.x)]
    #     bot.y += dy
    #     bot.x += dx
    #     self.pos_map[(bot.y, bot.x)] = bot
    #     bot.add_move_cost(self.rubble[bot.y, bot.x])
    #
    # def process_attack(self, bot: Entity, target: Entity):
    #     if isinstance(bot, (Archon, Builder)):
    #         assert bot.team == target.team
    #     elif isinstance(bot, (Soldier, Sage, Watchtower)):
    #         assert bot.team == 1 - target.team
    #     else:
    #         raise TypeError
    #
    #     assert bot.act_cd < 10
    #
    #     bot.add_act_cost(self.rubble[bot.y, bot.x])
    #
    #     old_hp = target.curr_hp
    #     target.curr_hp = min(target.curr_hp - bot.dmg, target.max_hp)
    #     dmg_dealt = old_hp - max(target.curr_hp, 0)
    #     killed = target.curr_hp <= 0
    #     self.episode_metrics["dmg_dealt"] += dmg_dealt
    #     if killed:
    #         if isinstance(target, Archon):
    #             self.archon_counts[target.team] -= 1
    #
    #         self.lead[target.y, target.x] += int(0.2 * target.lead_value)
    #         self.gold[target.y, target.x] += int(0.2 * target.gold_value)
    #
    #         idx = self.units.index(target)
    #         del self.units[idx]
    #         del self.pos_map[(target.y, target.x)]
    #         if idx <= self.curr_idx:
    #             self.curr_idx -= 1
    #
    #         self.episode_metrics["units_killed"] += 1
    #
    #     return dmg_dealt, killed
    #
    # def disintegrate(self, bot):
    #     assert isinstance(bot, Builder)
    #     self.lead[bot.y, bot.x] += int(0.2 * bot.lead_value)
    #     idx = self.units.index(bot)
    #
    #     del self.units[idx]
    #     del self.pos_map[(bot.y, bot.x)]
    #     if idx <= self.curr_idx:
    #         self.curr_idx -= 1
    #
    # def create_unit(self, unit_class, pos, team):
    #     assert pos not in self.pos_map
    #     assert self.in_bounds(pos[0], pos[1])
    #     assert isinstance(pos, tuple)
    #
    #     new_unit = unit_class(y=pos[0], x=pos[1], team=team)
    #     self.units.append(new_unit)
    #     self.pos_map[pos] = new_unit
    #     self.lead_banks[team] -= new_unit.lead_value
    #     self.gold_banks[team] -= new_unit.gold_value
    #
    # def iter_agents(
    #     self,
    # ) -> typing.Generator[tuple[Entity, np.ndarray], None, None]:
    #     """Yields Entity objects, observations, and action masks"""
    #     assert self.curr_idx is None
    #     assert self.t < self.max_episode_length
    #
    #     self.curr_idx = 0
    #     while self.curr_idx < len(self.units):
    #         bot = self.units[self.curr_idx]
    #
    #         # temporary heuristic behavior, not policy-controlled
    #         if isinstance(bot, Laboratory):
    #             self.step(bot, action=1)
    #             self.curr_idx += 1
    #             continue
    #
    #         assert not isinstance(bot, Watchtower)  # TODO
    #
    #         yield bot, self.legal_action_mask(bot)
    #
    #         self.curr_idx += 1
    #     self.curr_idx = None
    #
    #     for bot in self.units:
    #         bot.move_cd = max(0, bot.move_cd - 10)
    #         bot.act_cd = max(0, bot.act_cd - 10)
    #
    #     self.t += 1
    #     if self.t == self.max_episode_length:
    #         self.done = True
    #     else:
    #         self.lead_banks[0] += 2
    #         self.lead_banks[1] += 2
    #         if self.t % 20 == 0:
    #             self.lead[self.lead > 0] += 5
    #
    # def step(self, bot, action):
    #     if not isinstance(bot, Building) and 1 <= action <= 8:
    #         self.process_move(bot, action)
    #
    #     # auto-mining
    #     if isinstance(bot, Miner) and bot.act_cd < 10:
    #         available_lead = [
    #             pos
    #             for (dy, dx) in within_radius(bot.act_rad, prev_move=action)
    #             if self.in_bounds(bot.y + dy, bot.x + dx)
    #             and self.lead[pos := (bot.y + dy, bot.x + dx)] >= 1
    #         ]
    #         while available_lead and bot.act_cd < 10:
    #             selected = self.rng.choice(available_lead)
    #             self.lead[selected] -= 1
    #             self.lead_banks[bot.team] += 1
    #             if self.lead[selected] <= 1:
    #                 available_lead.remove(selected)
    #             bot.add_act_cost(self.rubble[bot.y, bot.x])
    #             self.episode_metrics[f"lead_mined"] += 1
    #
    #     # auto-repair
    #     if (
    #         (isinstance(bot, Builder) and 0 <= action <= 8)
    #         or (isinstance(bot, Archon) and action == 0)
    #     ) and bot.act_cd < 10:
    #         targets = list(
    #             self.nearby_bots(bot.y, bot.x, bot.act_rad, bot.team, prev_move=action)
    #         )
    #         targets = [
    #             target
    #             for target in targets
    #             if target.curr_hp < target.max_hp and target != bot
    #         ]
    #         if targets:
    #             selected = self.rng.choice(targets)
    #             self.process_attack(bot, selected)
    #
    #     # lab construction
    #     if isinstance(bot, Builder) and action == 9:
    #         assert self.lead_banks[bot.team] >= Laboratory.lead_value
    #         available_pos = [
    #             (bot.y + dy, bot.x + dx)
    #             for (dy, dx) in DIRECTIONS[1:]
    #             if (bot.y + dy, bot.x + dx) not in self.pos_map
    #             and self.in_bounds(bot.y + dy, bot.x + dx)
    #         ]
    #         assert len(available_pos) > 0
    #         chosen_pos = self.rng.choice(available_pos)
    #         self.create_unit(Laboratory, chosen_pos, bot.team)
    #         bot.add_act_cost(self.rubble[bot.y, bot.x])
    #
    #     # disintegrate
    #     if isinstance(bot, Builder) and action == 10:
    #         self.disintegrate(bot)
    #
    #     # auto-attacking
    #     if isinstance(bot, (Soldier, Sage)) and bot.act_cd < 10:
    #         nearby_enemies = list(
    #             self.nearby_bots(
    #                 bot.y, bot.x, bot.act_rad, 1 - bot.team, prev_move=action
    #             )
    #         )
    #         if nearby_enemies:
    #
    #             def score(enemy: Entity):
    #                 num_shots = enemy.curr_hp // bot.dmg
    #                 return -num_shots, enemy.curr_hp, -bot.distsq(enemy)
    #
    #             best_score = max(score(enemy) for enemy in nearby_enemies)
    #             chosen_enemy = self.rng.choice(
    #                 [enemy for enemy in nearby_enemies if score(enemy) == best_score]
    #             )
    #             self.process_attack(bot, chosen_enemy)
    #
    #     # droid creation
    #     if isinstance(bot, Archon) and action != 0:
    #         map_center = (self.rubble.shape[0] // 2, self.rubble.shape[1] // 2)
    #         all_spawn_pos = []
    #         good_spawn_pos = []
    #         for dy, dx in DIRECTIONS[1:]:
    #             y = bot.y + dy
    #             x = bot.x + dx
    #             if (y, x) not in self.pos_map and self.in_bounds(y, x):
    #                 all_spawn_pos.append((y, x))
    #                 if abs(y - map_center[0]) + abs(x - map_center[1]) < abs(
    #                     bot.y - map_center[0]
    #                 ) + abs(bot.x - map_center[1]):
    #                     good_spawn_pos.append((y, x))
    #
    #         chosen_pos = self.rng.choice(
    #             good_spawn_pos if good_spawn_pos else all_spawn_pos
    #         )
    #
    #         unit_class_map = {
    #             1: Miner,
    #             2: Builder,
    #             3: Sage if self.gold_banks[bot.team] >= Sage.gold_value else Soldier,
    #         }
    #         new_unit_class = unit_class_map[action]
    #         self.create_unit(new_unit_class, chosen_pos, bot.team)
    #         bot.add_act_cost(self.rubble[bot.y, bot.x])
    #         self.episode_metrics[f"spawned_{new_unit_class.__name__.lower()}"] += 1
    #
    #     # auto-transmute
    #     if isinstance(bot, Laboratory) and action == 1:
    #         nearby = 0
    #         for (dy, dx) in within_radius(bot.vis_rad):
    #             y, x = bot.y + dy, bot.x + dx
    #             if (y, x) in self.pos_map and self.pos_map[y, x].team == bot.team:
    #                 nearby += 1
    #
    #         lead_cost = bot.lead_ratio(nearby)
    #         self.lead_banks[bot.team] -= lead_cost
    #         self.gold_banks[bot.team] += 1
    #         bot.add_act_cost(self.rubble[(bot.y, bot.x)])
    #
    #     if min(self.archon_counts) == 0:
    #         self.done = True
    #
    # def winner(self):
    #     if self.archon_counts[0] != self.archon_counts[1]:
    #         return 0 if self.archon_counts[0] > self.archon_counts[1] else 1
    #     lead_values = self.lead_banks.copy()
    #     gold_values = self.gold_banks.copy()
    #     for unit in self.units:
    #         lead_values[unit.team] += unit.lead_value
    #         gold_values[unit.team] += unit.gold_value
    #     if gold_values[0] != gold_values[1]:
    #         return 0 if gold_values[0] > gold_values[1] else 1
    #     return 0 if lead_values[0] > lead_values[1] else 1
    #
    # # reward for team 0 shared reward (zero sum --> equal to negative team 1 reward)
    # def get_team_reward(self):
    #     assert self.curr_idx is None  # not in middle of units pass
    #
    #     value_differential = 0
    #     for unit in self.units:
    #         unit_value = (
    #             unit.lead_value + self.gold_reward_shaping_factor * unit.gold_value
    #         )
    #         if self.reward_shaping_depends_hp:
    #             unit_value *= unit.curr_hp / unit.max_hp
    #
    #         if unit.team == 0:
    #             value_differential += unit_value
    #         else:
    #             value_differential -= unit_value
    #
    #     ret = (
    #         value_differential - self.prev_value_differential
    #     ) * self.reward_scaling_factor
    #     self.prev_value_differential = value_differential
    #
    #     # TODO win reward
    #
    #     return ret
    #
    # def push_episode_metrics(self, logger):
    #     logger.push(self.episode_metrics)
