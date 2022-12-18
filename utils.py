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


@functools.cache
def within_radius(radsq, symmetry=0, prev_move=0):
    assert isinstance(symmetry, int) and 0 <= symmetry < 8
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

    if symmetry % 2 >= 1:
        dys = dys[::-1]
    if symmetry % 4 >= 2:
        dxs = dxs[::-1]

    for dy in dys:
        for dx in dxs:
            if dy * dy + dx * dx <= radsq or (dy + ay) ** 2 + (dx + ax) ** 2 <= radsq:
                if symmetry < 4:
                    ret.append((dy, dx))
                else:
                    ret.append((dx, dx))

    return ret
