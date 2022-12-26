import math
import timeit

import torch
import torch.nn as nn
import torch.nn.functional as F

from Re_Battlecode22.env import Miner


class MaskedConv(nn.Conv2d):
    def __init__(self, unit_cls, device=None):
        ksize = 1 + 2 * math.floor(math.sqrt(unit_cls.vis_rad))
        super().__init__(
            in_channels=7,
            out_channels=unit_cls.action_space.n,
            kernel_size=ksize,
            padding="same",
            device=device,
        )
        mask = torch.ones((unit_cls.action_space.n, 7, ksize, ksize))
        mid = ksize // 2
        for row in range(ksize):
            for col in range(ksize):
                if (mid - row) ** 2 + (mid - col) ** 2 > unit_cls.vis_rad:
                    mask[:, :, row, col] = 0
        self.register_buffer("mask", mask)

    def forward(self, x):
        return self._conv_forward(x, self.mask * self.weight, self.bias)


def forward2(input_, mask, weight, y, x):
    return torch.matmul(mask * weight, F.pad(input_, (4, 4))[y : y + 9, x : x + 9])


def main():
    data = torch.randn((7, 30, 30))
    model1 = MaskedConv(Miner)

    mask = model1.mask.clone().detach()
    weight = model1.weight.clone().detach().requires_grad_()
    y, x = 5, 6

    def lam1():
        return model1(data)[:, y, x]

    # def lam2():
    #     return torch.matmul(mask * weight, F.pad(data, (4, 4))[5 : 5 + 9, 5 : 5 + 9])

    # print(lam1().shape)
    # print((mask * weight).shape)
    # print((F.pad(data, (4, 4, 4, 4))[:, y : y + 9, x : x + 9]).shape)
    # print((mask * weight * F.pad(data, (4, 4, 4, 4))[:, 5 : 5 + 9, 5 : 5 + 9]).shape)
    # print(lam1())
    # print((mask * weight * F.pad(data, (4, 4, 4, 4))[:, 5 : 5 + 9, 5 : 5 + 9]).sum(dim=(1, 2, 3)) + model1.bias)

    def lam2():
        return (mask * weight * F.pad(data, (4, 4, 4, 4))[:, y : y + 9, x : x + 9]).sum(dim=(1, 2, 3)) + model1.bias

    def lam3():
        return torch.tensordot(
            mask * weight,
            F.pad(data, (4, 4, 4, 4))[:, y : y + 9, x : x + 9],
            dims=3
        ) + model1.bias
    print(lam1())
    print(lam2())
    print(lam3())

    print(timeit.timeit(lam1, number=1000))
    print(timeit.timeit(lam2, number=1000))
    print(timeit.timeit(lam3, number=1000))


main()
