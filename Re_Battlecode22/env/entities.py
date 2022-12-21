import math
from abc import ABC, abstractmethod

import gym
import numpy as np

from Re_Battlecode22.utils import within_radius

DIRECTIONS = [
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
]


class Entity(ABC):
    def __init__(self, x: int, y: int, team: int):
        self.x: int = x
        self.y: int = y
        self.team: int = team

        self.curr_hp: int = self.max_hp
        self.move_cd: int = 0
        self.act_cd: int = 0

    @property
    @abstractmethod
    def lead_value(self) -> int:
        pass

    @property
    @abstractmethod
    def gold_value(self) -> int:
        pass

    @property
    @abstractmethod
    def max_hp(self) -> int:
        pass

    @property
    @abstractmethod
    def dmg(self) -> int:
        pass

    @property
    @abstractmethod
    def move_cost(self) -> int:
        pass

    @property
    @abstractmethod
    def act_cost(self) -> int:
        pass

    @property
    @abstractmethod
    def act_rad(self) -> int:
        pass

    @property
    @abstractmethod
    def vis_rad(self) -> int:
        pass

    @property
    @abstractmethod
    def sprite(self) -> str:
        pass

    @property
    def observation_space(self) -> gym.spaces.Box:
        raise NotImplementedError

    @property
    def action_space(self) -> gym.spaces.Discrete:
        raise NotImplementedError

    def distsq(self, other: "Entity"):
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2

    def add_move_cost(self, rubble: int):
        self.move_cd += math.floor(1 + rubble / 10) * self.move_cost

    def add_act_cost(self, rubble: int):
        self.act_cd += math.floor(1 + rubble / 10) * self.act_cost


class Miner(Entity):
    lead_value = 50
    gold_value = 0
    max_hp = 40
    dmg = 0
    move_cost = 20
    act_cost = 2
    act_rad = 2
    vis_rad = 20
    sprite = "m"

    observation_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(len(within_radius(20)) * 11,)
    )
    action_space = gym.spaces.Discrete(9)  # TODO: add leave-1-remaining decision


class Builder(Entity):
    lead_value = 40
    gold_value = 0
    max_hp = 30
    dmg = -2
    move_cost = 20
    act_cost = 10
    act_rad = 5
    vis_rad = 20
    sprite = "b"

    observation_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(len(within_radius(20)) * 7,)
    )
    action_space = gym.spaces.Discrete(10)


class Soldier(Entity):
    lead_value = 75
    gold_value = 0
    max_hp = 50
    dmg = 3
    move_cost = 16
    act_cost = 10
    act_rad = 13
    vis_rad = 20
    sprite = "s"

    observation_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(len(within_radius(20)) * 7,)
    )
    action_space = gym.spaces.Discrete(9)


class Sage(Entity):
    lead_value = 0
    gold_value = 20
    max_hp = 100
    dmg = 45
    move_cost = 25
    act_cost = 200
    act_rad = 25
    vis_rad = 34
    sprite = "g"

    observation_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(len(within_radius(34)) * 7,)
    )
    action_space = gym.spaces.Discrete(9)


class Building(Entity):
    def __init__(
        self,
        x: int,
        y: int,
        team: int,
        mode: str,
    ):
        super().__init__(x, y, team)
        self.mode = mode
        self.level = 1

    @abstractmethod
    def level_to(self, level: int):
        pass


class Archon(Building):
    lead_value = 0
    gold_value = 100

    max_hp = 600
    dmg = -2
    move_cost = 24
    act_cost = 10
    act_rad = 20
    vis_rad = 34
    sprite = "A"

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(15,))
    action_space = gym.spaces.Discrete(4)

    def __init__(self, x, y, team):
        super().__init__(x, y, team, mode="turret")

    def level_to(self, level: int):
        assert self.level == 1 and level == 2 or self.level == 2 and level == 3
        self.level += 1

        self.dmg -= 2
        if level == 2:
            self.lead_value += 300
            self.curr_hp += 480
            self.max_hp += 480
        else:
            self.gold_value += 80
            self.curr_hp += 864
            self.max_hp += 864


class Laboratory(Building):
    lead_value = 180
    gold_value = 0

    max_hp = 100
    dmg = 0
    move_cost = 24
    act_cost = 10
    act_rad = 0
    vis_rad = 53
    sprite = "L"

    observation_space = None
    action_space = gym.spaces.Discrete(2)

    def __init__(self, x, y, team):
        super().__init__(x, y, team, mode="prototype")
        self.curr_hp = round(0.8 * self.max_hp)

    def level_to(self, level: int):
        assert self.level == 1 and level == 2 or self.level == 2 and level == 3
        self.level += 1

        if level == 2:
            self.lead_value += 150
            self.curr_hp += 80
            self.max_hp += 80
        else:
            self.gold_value += 25
            self.curr_hp += 144
            self.max_hp += 144

    def lead_ratio(self, num_nearby_ally: int):
        if self.level == 1:
            k = 0.02
        elif self.level == 2:
            k = 0.01
        else:
            k = 0.005

        return math.floor(20 - 18 * math.exp(-k * num_nearby_ally))


class Watchtower(Building):
    lead_value = 150
    gold_value = 0

    max_hp = 150
    dmg = 4
    move_cost = 24
    act_cost = 10
    act_rad = 25
    vis_rad = 34
    sprite = "W"

    # ACTIONS
    # [0, 8] moves (incl. idle)
    # [9, 89] attacks (can't attack self)
    # 90 turret mode
    # 91 portable mode
    _ATK_MAP = {
        entry: 9 + i for i, entry in enumerate(within_radius(25)) if entry != (0, 0)
    }

    def __init__(self, x, y, team):
        super().__init__(x, y, team, mode="prototype")
        self.curr_hp = round(0.8 * self.max_hp)

    def level_to(self, level: int):
        assert self.level == 1 and level == 2 or self.level == 2 and level == 3
        self.level += 1

        self.dmg += 4
        if level == 2:
            self.lead_value += 150
            self.curr_hp += 120
            self.max_hp += 120
        else:
            self.gold_value += 60
            self.curr_hp += 216
            self.max_hp += 216

    def attack_action(self, target_y, target_x) -> int:
        return self._ATK_MAP[(target_y - self.y, target_x - self.x)]
