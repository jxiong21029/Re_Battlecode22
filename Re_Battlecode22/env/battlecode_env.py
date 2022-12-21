import glob
import typing
from collections import defaultdict

from .entities import *


class BattlecodeEnv:
    def __init__(
        self,
        map_selection: str | None = None,
        augment_obs: bool = True,
        gold_reward_shaping_factor: float = 5.0,
        reward_shaping_depends_hp: bool = True,
        seed: int | None = None,
    ):
        self.map_selection = map_selection
        self.augment_obs = augment_obs
        self.gold_reward_shaping_factor = gold_reward_shaping_factor
        self.reward_shaping_depends_hp = reward_shaping_depends_hp
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.map_selection = self.map_selection

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

    def in_bounds(self, y, x):
        return 0 <= y < self.rubble.shape[0] and 0 <= x < self.rubble.shape[1]

    def reset(self):
        filenames = glob.glob("Re_Battlecode22/maps/data/*.npz")
        data = np.load(
            self.rng.choice(filenames)
            if self.map_selection is None
            else f"maps/data/{self.map_selection}.npz"
        )

        self.rubble = data["rubble"]
        self.lead = data["lead"]
        self.gold = np.zeros_like(self.lead)

        self.t = 0
        self.lead_banks = [200, 200]
        self.gold_banks = [0, 0]
        self.units = []
        self.pos_map = {}
        for row0, row1 in zip(data["team0_archon_pos"], data["team1_archon_pos"]):
            self.units.append(Archon(y=row0[1], x=row0[0], team=0))
            self.units.append(Archon(y=row1[1], x=row1[0], team=1))
        for unit in self.units:
            self.pos_map[(unit.y, unit.x)] = unit
        self.archon_counts = [len(self.units) // 2, len(self.units) // 2]
        self.prev_value_differential = 0
        self.done = False

    # basic observation features, observed by all bots (simplified for now)
    # - own x, own y
    # - own move & action cd, // 10, (+log scale?)
    # - TODO: own is turret / is portable

    # cell observation features, observed by all non-building bots (simplified for now):
    # - rubble
    # - ally log hp, ally is building
    # - enemy log hp, enemy is building
    # - TODO: other unit move and action cd

    # other observations:
    # - miner: cell observations also include log lead and log gold
    # - archon: only sees timestep, lead bank, gold bank, allied count of each unit

    # action space (simplified for now):
    # - miner: Discrete(9) movements
    #       - auto-mines nearby resources
    # - builder: Discrete(9) movements x Discrete(2) [repair / idle, spawn lab]
    # - soldier: Discrete(9) movements
    #       - auto-attacks enemy (heuristic)
    # - sage: Discrete(9) movements
    #       - auto-attacks enemy (heuristic)
    # - archon: Discrete(4) [repair / idle, spawn miner, spawn builder, spawn combat]
    # - lab: auto-converts lead to gold

    def nearby_bots(self, y, x, radsq, team=None, prev_move: int = 0):
        for (dy, dx) in within_radius(radsq, prev_move=prev_move):
            if (y + dy, x + dx) in self.pos_map and (
                (yld := self.pos_map[(y + dy, x + dx)]).team == team or team is None
            ):
                yield yld

    def observe(self, bot, symmetry=0):  # 4 rotations * 2 reflections * 2 team swaps
        if isinstance(bot, Archon):
            counts = defaultdict(int)
            for unit in self.units:
                if unit.team == bot.team:
                    counts[unit.__class__.__name__.lower()] += 1
            for key in counts.keys():
                assert key in (
                    "miner",
                    "builder",
                    "soldier",
                    "sage",
                    "laboratory",
                    "archon",
                )

            # TODO observation normalization
            return np.array(
                [
                    (bot.y - self.rubble.shape[0] / 2 + 0.5) / 30,
                    (bot.x - self.rubble.shape[1] / 2 + 0.5) / 30,
                    self.rubble.shape[0] / 60,
                    self.rubble.shape[1] / 60,
                    self.t / 2000,
                    np.log1p(self.t) / np.log1p(2000),
                    np.log1p(2000 - self.t) / np.log1p(2000),
                    self.lead_banks[bot.team] / 256,
                    np.log1p(self.lead_banks[bot.team]) / np.log1p(256),
                    self.gold_banks[bot.team] / 64,
                    np.log1p(self.gold_banks[bot.team]) / np.log1p(64),
                    counts["miner"] / 32,
                    counts["builder"] / 32,
                    counts["soldier"] / 32,
                    counts["sage"] / 32,
                ],
                dtype=np.float32,
            )
        elif isinstance(bot, (Laboratory, Watchtower)):
            raise NotImplementedError
        else:
            assert isinstance(bot, (Miner, Builder, Soldier, Sage))
            ret = [bot.x, bot.y, bot.curr_hp, bot.move_cd, bot.act_cd]

            for dy, dx in within_radius(bot.vis_rad):
                y, x = bot.y + dy, bot.x + dx
                other: Entity = self.pos_map[(y, x)] if (y, x) in self.pos_map else None
                is_ally = other is not None and other.team == bot.team
                is_enemy = other is not None and other.team != bot.team
                is_building = float(isinstance(other, Building))
                ret.extend(
                    [
                        self.rubble[y, x] / 100 if self.in_bounds(y, x) else 0,
                        other.curr_hp / 512 if is_ally else 0,
                        np.log1p(other.curr_hp) / np.log1p(512) if is_ally else 0,
                        is_building if is_ally else 0,
                        other.curr_hp / 512 if is_enemy else 0,
                        np.log1p(other.curr_hp) / np.log1p(512) if is_enemy else 0,
                        is_building if is_enemy else 0,
                    ]
                )
                if isinstance(bot, Miner):
                    ret.extend(
                        [
                            self.lead[y, x] / 128 if self.in_bounds(y, x) else 0,
                            np.log1p(self.lead[y, x]) / np.log1p(128)
                            if self.in_bounds(y, x)
                            else 0,
                            self.gold[y, x] / 32 if self.in_bounds(y, x) else 0,
                            np.log1p(self.gold[y, x]) / np.log1p(32)
                            if self.in_bounds(y, x)
                            else 0,
                        ]
                    )
            return np.array(ret, dtype=np.float32)

    # TODO augmentations: map flips, rotations. team swap should negate q-value.
    def global_observation(self):
        # timestep, lead bank, gold bank for both teams, map size, x, y coordinate (8)
        # rubble, lead, gold (3)
        # ally HP, type one-hot, move_cd, act_cd (1 + 6 + 2 = 9)
        # same for opponent (9)
        # NUM CHANNELS: 8 + 3 + 9 + 9 = 29

        ret = np.zeros(
            (29, self.rubble.shape[0], self.rubble.shape[1]), dtype=np.float32
        )

        # global information: copied into every x,y coordinate
        # timestep, lead+gold banks, map size. also includes each cell's x, y coordinate
        ret[0] = self.t / 2000
        ret[1:3] = np.array(self.lead_banks).reshape((-1, 1, 1)) / 256
        ret[3:5] = np.array(self.gold_banks).reshape((-1, 1, 1)) / 64
        ret[5:7] = np.array(self.rubble.shape).reshape((-1, 1, 1)) / 60
        ret[8] = (
            np.arange(self.rubble.shape[0]) - self.rubble.shape[0] / 2 + 0.5
        ).reshape((-1, 1)) / 30
        ret[9] = (
            np.arange(self.rubble.shape[1]) - self.rubble.shape[1] / 2 + 0.5
        ).reshape((1, -1)) / 30

        # terrain: rubble, lead, gold
        ret[10] = self.rubble / 100
        ret[11] = self.lead / 128
        ret[12] = self.gold / 32

        # units: ally HP, type one-hot, move_cd, act_cd, x2 for opp

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
                ret[13 + 2 * unit_type_id + unit.team] = 1

                ret[25 + unit.team] = unit.move_cd
                ret[27 + unit.team] = unit.act_cd
        return ret

    def legal_action_mask(self, bot, symmetry=0):
        ret = np.zeros(bot.action_space.n, dtype=bool)

        ret[0] = True
        if not isinstance(bot, Building):
            # move actions: [0, 8] inclusive for all bots
            for i, (dy, dx) in enumerate(DIRECTIONS):
                if i == 0:
                    continue
                elif (
                    bot.move_cd < 10
                    and 0 <= bot.y + dy < self.rubble.shape[0]
                    and 1 <= bot.x + dx < self.rubble.shape[1]
                    and (bot.y + dy, bot.x + dx) not in self.pos_map
                ):
                    ret[i] = True

        adj_available = False
        for dy, dx in DIRECTIONS[1:]:
            if (
                0 <= bot.y + dy < self.rubble.shape[0]
                and 1 <= bot.x + dx < self.rubble.shape[1]
                and (bot.y + dy, bot.x + dx) not in self.pos_map
            ):
                adj_available = True
                break

        if isinstance(bot, Builder):
            assert ret.shape == (10,)
            if (
                self.lead_banks[bot.team] >= Laboratory.lead_value
                and bot.act_cd < 10
                and adj_available
            ):
                ret[9] = True
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
                    len(list(self.nearby_bots(bot.y, bot.x, bot.vis_rad, bot.team)))
                )
                ret[1] = lead_cost >= self.lead_banks[bot.team]
        elif isinstance(bot, Watchtower):
            return NotImplementedError  # TODO

        return ret

    def process_move(self, bot: Entity, action: int):
        assert bot.move_cd < 10
        dy, dx = DIRECTIONS[action]
        assert (bot.y + dy, bot.x + dx) not in self.pos_map
        del self.pos_map[(bot.y, bot.x)]
        bot.y += dy
        bot.x += dx
        self.pos_map[(bot.y, bot.x)] = bot
        bot.add_move_cost(self.rubble[bot.y, bot.x])

    def process_attack(self, bot: Entity, target: Entity):
        if isinstance(bot, (Archon, Builder)):
            assert bot.team == target.team
        elif isinstance(bot, (Soldier, Sage, Watchtower)):
            assert bot.team == 1 - target.team
        else:
            raise TypeError

        assert bot.act_cd < 10
        assert (target.y - bot.y) ** 2 + (target.x - bot.x) ** 2 <= bot.act_rad

        bot.add_act_cost(self.rubble[bot.y, bot.x])

        old_hp = target.curr_hp
        target.curr_hp = min(target.curr_hp - bot.dmg, target.max_hp)
        dmg_dealt = old_hp - max(target.curr_hp, 0)
        killed = target.curr_hp <= 0
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

        return dmg_dealt, killed

    def create_unit(self, unit_class, pos, team):
        assert pos not in self.pos_map
        assert isinstance(pos, tuple)

        new_unit = unit_class(y=pos[0], x=pos[1], team=team)
        self.units.append(new_unit)
        self.pos_map[pos] = new_unit
        self.lead_banks[team] -= new_unit.lead_value
        self.gold_banks[team] -= new_unit.gold_value

    # passes through all agents once in order, yielding Entity objects, observations,
    # and action masks
    def agent_inputs(
        self,
    ) -> typing.Generator[tuple[Entity, np.ndarray, np.ndarray], None, None]:
        assert self.curr_idx is None
        assert self.t < 2000

        self.curr_idx = 0
        while self.curr_idx < len(self.units):
            bot = self.units[self.curr_idx]

            # temporary heuristic behavior, not policy-controlled
            if isinstance(bot, Laboratory):
                self.step(bot, action=1)
                continue

            assert not isinstance(bot, Watchtower)  # TODO

            symmetry = self.rng.integers(16) if self.augment_obs else 0
            yield bot, self.observe(bot, symmetry), self.legal_action_mask(
                bot, symmetry
            )

            self.curr_idx += 1
        self.curr_idx = None

        self.t += 1
        if self.t == 2000:
            self.done = True
        else:
            self.lead_banks[0] += 2
            self.lead_banks[1] += 2
            if self.t % 20 == 0:
                self.rubble[self.rubble > 0] += 5

    def step(self, bot, action):
        if not isinstance(bot, Building) and 1 <= action <= 8:
            self.process_move(bot, action)

        # auto-mining
        if isinstance(bot, Miner) and bot.act_cd < 10:
            available_lead = [
                pos
                for (dy, dx) in within_radius(bot.act_rad, prev_move=action)
                if self.in_bounds(bot.y + dy, bot.x + dx)
                and (self.lead[pos := (bot.y + dy, bot.x + dx)]) > 0
            ]
            while available_lead and bot.act_cd < 10:
                selected = tuple(self.rng.choice(available_lead))
                self.lead[selected] -= 1
                self.lead_banks[bot.team] += 1
                if self.lead[selected] == 0:
                    available_lead.remove(selected)
                bot.add_act_cost(self.rubble[bot.y, bot.x])

        # auto-repair
        if (
            (isinstance(bot, Builder) and 0 <= action <= 8)
            or (isinstance(bot, Archon) and action == 0)
        ) and bot.act_cd < 10:
            targets = list(
                self.nearby_bots(bot.y, bot.x, bot.act_rad, bot.team, prev_move=action)
            )
            targets = [target for target in targets if target.curr_hp < target.max_hp]
            if targets:
                selected = self.rng.choice(targets)
                self.process_attack(bot, selected)

        # lab construction
        if isinstance(bot, Builder) and action == 9:
            assert self.lead_banks[bot.team] >= Laboratory.lead_value
            available_pos = [
                (bot.y + dy, bot.x + dx)
                for (dy, dx) in DIRECTIONS[1:]
                if (bot.y + dy, bot.x + dx) not in self.pos_map
            ]
            assert len(available_pos) > 0
            chosen_pos = tuple(self.rng.choice(available_pos))
            self.create_unit(Laboratory, chosen_pos, bot.team)
            bot.add_act_cost(self.rubble[bot.y, bot.x])

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
                self.process_attack(bot, chosen_enemy)

        if isinstance(bot, Archon) and action != 0:
            all_spawn_pos = [
                pos
                for dy, dx in DIRECTIONS[1:]
                if (pos := (bot.y + dy, bot.x + dx)) not in self.pos_map
            ]
            map_center = (self.rubble.shape[0] // 2, self.rubble.shape[1] // 2)
            good_spawn_pos = [
                pos
                for pos in all_spawn_pos
                if abs(pos[0] - map_center[0]) + abs(pos[1] - map_center[1])
                < abs(bot.y - map_center[0]) + abs(bot.x - map_center[1])
            ]
            chosen_pos = tuple(
                self.rng.choice(good_spawn_pos if good_spawn_pos else all_spawn_pos)
            )

            unit_class_map = {
                1: Miner,
                2: Builder,
                3: Sage if self.gold_banks[bot.team] >= Sage.gold_value else Soldier,
            }
            new_unit_class = unit_class_map[action]
            self.create_unit(new_unit_class, chosen_pos, bot.team)
            bot.add_act_cost(self.rubble[bot.y, bot.x])

        if isinstance(bot, Laboratory) and action == 1:
            nearby = 0
            for (dy, dx) in within_radius(bot.vis_rad):
                y, x = bot.y + dy, bot.x + dx
                if (y, x) in self.pos_map and self.pos_map[y, x].team == bot.team:
                    nearby += 1
            lead_cost = bot.lead_ratio(nearby)
            self.lead_banks[bot.team] -= lead_cost
            self.gold_banks[bot.team] += 1
            bot.add_act_cost(self.rubble[(bot.y, bot.x)])

        if min(self.archon_counts) == 0:
            self.done = True

    # reward for team 0 shared reward (zero sum --> equal to negative team 1 reward)
    def get_team_reward(self):
        assert self.curr_idx is None  # not in middle of units pass

        value_differential = 0
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

        ret = value_differential - self.prev_value_differential
        self.prev_value_differential = value_differential

        # TODO win reward

        return ret
