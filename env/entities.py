from abc import ABC, abstractmethod


class Entity(ABC):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.curr_hp = self.max_hp
        self.move_cd = 0
        self.act_cd = 0

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

    def __init__(self, x, y):
        super().__init__(x, y)


class Builder(Entity):
    lead_value: 40
    gold_value: 0
    max_hp = 30
    dmg = -2
    move_cost = 20
    act_cost = 10
    act_rad = 5
    vis_rad = 20
    sprite = "b"

    def __init__(self, x, y):
        super().__init__(x, y)


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

    def __init__(self, x, y):
        super().__init__(x, y)


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

    def __init__(self, x, y):
        super().__init__(x, y)


class Building(Entity):
    def __init__(
        self,
        x: int,
        y: int,
        mode: str,
    ):
        super().__init__(x, y)
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

    def __init__(self, x, y):
        super().__init__(x, y, mode="turret")

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

    def __init__(self, x, y):
        super().__init__(x, y, mode="prototype")
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
