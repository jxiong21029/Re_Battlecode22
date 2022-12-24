import math
import os
import sys
import time

import numpy as np
import pygame

# render config
MARGIN = 10

# rubble
CLEAR = np.array([235, 225, 190], dtype=np.float32)
RUBBLE = np.array([90, 80, 20], dtype=np.float32)


def rubble_color(rubble_amt):
    color = CLEAR + (rubble_amt / 100) * (RUBBLE - CLEAR)
    return tuple(color)


class Renderer:
    def __init__(self):
        self.screen = pygame.display.set_mode((1000, 700))

    def render(self, env):
        size = math.floor(min(1000 / env.rubble.shape[1], 700 / env.rubble.shape[0]))
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                pygame.quit()
                pygame.display.quit()
                sys.exit()

        self.screen.fill((0, 0, 0))
        h, w = env.rubble.shape
        pygame.draw.line(
            self.screen,
            (150, 150, 150),
            (MARGIN, MARGIN),
            (MARGIN + size * h, MARGIN),
        )
        pygame.draw.line(
            self.screen,
            (150, 150, 150),
            (MARGIN, MARGIN),
            (MARGIN, MARGIN + size * w),
        )
        pygame.draw.line(
            self.screen,
            (150, 150, 150),
            (MARGIN, MARGIN + size * w),
            (MARGIN + size * h, MARGIN + size * w),
        )
        pygame.draw.line(
            self.screen,
            (150, 150, 150),
            (MARGIN + size * h, MARGIN),
            (MARGIN + size * h, MARGIN + size * w),
        )

        img_dir = "Re_Battlecode22/env/images/"

        for y in range(env.rubble.shape[0]):
            for x in range(env.rubble.shape[1]):
                pygame.draw.rect(
                    self.screen,
                    rubble_color(env.rubble[y, x]),
                    pygame.Rect(MARGIN + size * x, MARGIN + size * y, size, size),
                )

                if env.lead[y, x] > 0:
                    image = pygame.image.load(os.path.join(img_dir, "lead.png"))
                    sz = max(2, round(env.lead[y, x] / 100 * size / 2) * 2)
                    image = pygame.transform.smoothscale(image, (sz, sz))
                    self.screen.blit(
                        image,
                        (
                            MARGIN + size * x + size // 2 - sz // 2,
                            MARGIN + size * y + size // 2 - sz // 2,
                        ),
                    )

                if env.gold[y, x] > 0:
                    image = pygame.image.load(os.path.join(img_dir, "gold.png"))
                    sz = max(2, round(env.gold[y, x] / 100 * size / 2) * 2)
                    image = pygame.transform.smoothscale(image, (sz, sz))
                    self.screen.blit(
                        image,
                        (
                            MARGIN + size * x + size // 2 - sz // 2,
                            MARGIN + size * y + size // 2 - sz // 2,
                        ),
                    )

                if (y, x) not in env.pos_map:
                    continue
                bot = env.pos_map[(y, x)]
                team = "blue_" if bot.team == 0 else "red_"
                image = pygame.image.load(
                    os.path.join(
                        img_dir, team + bot.__class__.__name__.lower() + ".png"
                    )
                )
                sz = max(2, round(bot.curr_hp / bot.max_hp * size / 2) * 2)
                image = pygame.transform.smoothscale(image, (sz, sz))
                self.screen.blit(
                    image,
                    (
                        MARGIN + size * bot.x + size // 2 - sz // 2,
                        MARGIN + size * bot.y + size // 2 - sz // 2,
                    ),
                )
        pygame.display.flip()
        time.sleep(0.2)
