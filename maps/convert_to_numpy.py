import glob
import os

import numpy as np
from battlecode.schema.GameMap import GameMap

dirname = "data"
if os.path.isdir(dirname):
    for filename in os.listdir(dirname):
        file_path = os.path.join(dirname, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)

for filename in glob.glob("raw/*.map22"):
    name = filename[4:-6]
    print(f"Starting map {name}")
    with open(filename, "rb") as f:
        buf = bytearray(f.read())
    game_map = GameMap.GetRootAsGameMap(buf, 0)

    shape = (game_map.MaxCorner().Y(), game_map.MaxCorner().X())

    rubble = game_map.RubbleAsNumpy().reshape(shape).astype(int)
    lead = game_map.LeadAsNumpy()[: shape[0] * shape[1]].reshape(shape).astype(int)

    num_archons_per_team = game_map.Bodies().Locs().XsLength() // 2
    team1_archon_locs = np.zeros((num_archons_per_team, 2), dtype=int)
    team2_archon_locs = np.zeros((num_archons_per_team, 2), dtype=int)

    idx1 = 0
    idx2 = 0
    assert game_map.Bodies().TeamIdsLength() == 2 * num_archons_per_team
    for i in range(2 * num_archons_per_team):
        if game_map.Bodies().TeamIds(i) == 1:
            team1_archon_locs[idx1, 0] = game_map.Bodies().Locs().Ys(i)
            team1_archon_locs[idx1, 1] = game_map.Bodies().Locs().Xs(i)
            idx1 += 1
        else:
            team2_archon_locs[idx2, 0] = game_map.Bodies().Locs().Ys(i)
            team2_archon_locs[idx2, 1] = game_map.Bodies().Locs().Xs(i)
            idx2 += 1

    # NOTE: dim 0 is the y-axis, dim 1 is the x-axis
    np.savez(
        f"{dirname}/{name}.npz",
        rubble=rubble,
        lead=lead,
        team0_archon_pos=team1_archon_locs,
        team1_archon_pos=team2_archon_locs,
    )
