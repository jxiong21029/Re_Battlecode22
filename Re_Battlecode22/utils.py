import functools
import math

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


def pos_symmetry_trsfm(
    y: int, x: int, symmetry: int, map_height: int = 1, map_width: int = 1
):
    assert isinstance(symmetry, int) and 0 <= symmetry <= 16
    dy = y - map_height / 2 + 0.5
    dx = x - map_width / 2 + 0.5
    if symmetry % 2 >= 1:
        dy = -dy
    if symmetry % 4 >= 2:
        dx = -dx
    if symmetry % 8 >= 4:
        dy, dx = dx, dy
    return int(dy - 0.5 + map_height / 2), int(dx - 0.5 + map_width / 2)


@functools.cache
def within_radius(radsq, symmetry=0, prev_move=0):
    ret = []

    ay, ax = DIRECTIONS[prev_move]
    dys = list(
        range(
            -int(math.sqrt(radsq)) - max(ay, 0), 1 + int(math.sqrt(radsq)) - min(ay, 0)
        )
    )
    dxs = list(
        range(
            -int(math.sqrt(radsq)) - max(ax, 0), 1 + int(math.sqrt(radsq)) - min(ax, 0)
        )
    )

    for dy in dys:
        for dx in dxs:
            if dy * dy + dx * dx <= radsq or (dy + ay) ** 2 + (dx + ax) ** 2 <= radsq:
                ret.append(pos_symmetry_trsfm(dy, dx, symmetry=symmetry))

    return ret
