import functools
import math
import os
from collections import defaultdict

import numpy as np

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


def symmetry_transform(
    y: int, x: int, symmetry: int, map_height: int = 1, map_width: int = 1
):
    dy = y - map_height / 2 + 0.5
    dx = x - map_width / 2 + 0.5
    if symmetry % 2 >= 1:
        dy = -dy
    if symmetry % 4 >= 2:
        dx = -dx
    if symmetry % 8 >= 4:
        dy, dx = dx, dy
    return int(dy - 0.5 + map_height / 2), int(dx - 0.5 + map_width / 2)


def global_symmetry_transform(arr, symmetry):
    ret = arr.copy()

    if len(arr.shape) == 2:
        if symmetry % 2 >= 1:
            ret = ret[::-1]
        if symmetry % 4 >= 2:
            ret = ret[:, ::-1]
        if symmetry % 8 >= 4:
            ret = np.transpose(ret, axes=(1, 0))
    else:
        assert len(ret.shape) == 3
        if symmetry % 2 >= 1:
            ret = ret[:, ::-1]
        if symmetry % 4 >= 2:
            ret = ret[:, :, ::-1]
        if symmetry % 8 >= 4:
            ret = np.transpose(ret, axes=(0, 2, 1))
    return ret


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
                ret.append(symmetry_transform(dy, dx, symmetry=symmetry))

    return ret


class Logger:
    def __init__(self):
        self._buffer_data = defaultdict(list)
        self.cumulative_data = defaultdict(list)

        self._cleared_prev_plots = False

    # log metrics, used once per epoch
    def log(self, metrics=None, **kwargs):
        metrics = {} if metrics is None else metrics
        for k, v in (metrics | kwargs).items():
            self.cumulative_data[k].append(v)

    # push metrics logged many times per epoch, e.g. loss, means and stds computed
    def push(self, metrics=None, **kwargs):
        metrics = {} if metrics is None else metrics
        for k, v in (metrics | kwargs).items():
            self._buffer_data[k].append(v)

    def step(self):
        for k, v in self._buffer_data.items():
            self.cumulative_data[k + "_mean"].append(np.mean(v))
            self.cumulative_data[k + "_std"].append(np.std(v))
        self._buffer_data.clear()

    def tune_report(self):
        from ray import tune

        tune.report(**{k: v[-1] for k, v in self.cumulative_data.items()})

    def air_report(self, **kwargs):
        from ray.air import session

        session.report({k: v[-1] for k, v in self.cumulative_data.items()}, **kwargs)

    def generate_plots(self, dirname="plotgen"):
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme()

        if not self._cleared_prev_plots:
            if os.path.isdir(dirname):
                for filename in os.listdir(dirname):
                    file_path = os.path.join(dirname, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
            else:
                os.mkdir(dirname)
            self._cleared_prev_plots = True

        for k, v in self.cumulative_data.items():
            if k.endswith("_std"):
                continue

            fig, ax = plt.subplots()

            x = np.arange(len(self.cumulative_data[k]))
            v = np.array(v)
            if k.endswith("_mean"):
                name = k[:-5]

                (line,) = ax.plot(x, v, label=k)
                stds = np.array(self.cumulative_data[name + "_std"])
                ax.fill_between(
                    x, v - stds, v + stds, color=line.get_color(), alpha=0.15
                )
            else:
                name = k
                (line,) = ax.plot(x, v)
            ax.scatter(x, v, color=line.get_color())

            fig.suptitle(name)
            fig.savefig(os.path.join(dirname, name))
            plt.close(fig)

    def convergence(self, key, smoothing=0.9, eps=1e-8):
        """Estimates the degree to which some metric has converged.
        A custom metric by me (Jerry). Close to zero when the metric is clearly trending
        upwards or downwards, close to one when changes in the metric seem to be
        dominated by noise. Intended for debugging purposes, not for scientific usage.
        """
        assert key in self.cumulative_data

        m = 0
        v = 0
        data = self.cumulative_data[key]
        if len(data) <= 1:
            return 0

        diffs = [data[i + 1] - data[i] for i in range(len(data) - 1)]

        for d in diffs:
            m = smoothing * m + (1 - smoothing) * d
            v = smoothing * v + (1 - smoothing) * d * d

        mh = m / (1 - smoothing ** len(diffs))
        vh = v / (1 - smoothing ** len(diffs))
        return 1 - abs(mh) / (math.sqrt(vh) + eps)
