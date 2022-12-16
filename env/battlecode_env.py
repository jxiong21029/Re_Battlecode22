import glob
from dataclasses import dataclass

import numpy as np

from .entities import *


@dataclass
class BattlecodeConfig:
    map_selection: str | None = None


class BattlecodeEnv:
    def __init__(self, config: BattlecodeConfig):
        self.rng = np.random.default_rng()
        self.cfg = config

        self.map_selection = self.cfg.map_selection
        self.rubble = np.empty(())
        self.lead = np.empty(())
        self.gold = np.empty(())

        self.t = 0
        self.units0 = []
        self.units1 = []

    def reset(self):
        filenames = glob.glob("maps/data/*.npz")
        data = np.load(
            self.rng.choice(filenames)
            if self.map_selection is None
            else f"maps/data/{self.map_selection}.npz"
        )

        self.rubble = data["rubble"]
        self.lead = data["lead"]
        self.gold = np.zeros_like(self.lead)

        self.t = 0
        self.units0 = [Archon(row[1], row[0]) for row in data["team0_archon_pos"]]
        self.units1 = [Archon(row[1], row[0]) for row in data["team1_archon_pos"]]

        for unit in self.units0 + self.units1:
            assert 0 <= unit.y < self.lead.shape[0]
            assert 0 <= unit.x < self.lead.shape[1]
