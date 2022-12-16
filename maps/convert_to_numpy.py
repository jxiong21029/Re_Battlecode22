import glob

import numpy as np

from battlecode.schema.GameMap import GameMap

for filename in glob.glob("maps/*.map22"):
    name = filename[5:-6]
    print(f"Starting map {name}")
    with open(filename, "rb") as f:
        buf = bytearray(f.read())
    game_map = GameMap.GetRootAsGameMap(buf, 0)

    shape = (game_map.MaxCorner().Y(), game_map.MaxCorner().X())

    rubble = game_map.RubbleAsNumpy().reshape(shape).astype(int)
    if name == "progress":
        lead = game_map.LeadAsNumpy()[: shape[0] * shape[1]].reshape(shape).astype(int)
    else:
        lead = game_map.LeadAsNumpy().reshape(shape).astype(int)

    num_archons_per_team = game_map.Bodies().Locs().XsLength() // 2
    team1_archon_locs = np.zeros((num_archons_per_team, 2), dtype=int)
    team2_archon_locs = np.zeros((num_archons_per_team, 2), dtype=int)

    idx1 = 0
    idx2 = 0
    for i in range(2 * num_archons_per_team):
        if game_map.Bodies().TeamIDs(i) == 1:
            team1_archon_locs[idx1, 0] = game_map.Bodies().Locs().Xs(i)
            team1_archon_locs[idx1, 1] = game_map.Bodies().Locs().Ys(i)
            idx1 += 1
        else:
            team2_archon_locs[idx2, 0] = game_map.Bodies().Locs().Xs(i)
            team2_archon_locs[idx2, 1] = game_map.Bodies().Locs().Ys(i)
            idx2 += 1

    np.savez(
        f"numpy/{name}.npz",
        rubble=rubble,
        lead=lead,
        team0_archon_pos=team1_archon_locs,
        team1_archon_pos=team2_archon_locs,
    )
