import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (polygonShape)
import random
import sys
import numpy as np
from Box2D.b2 import (vec2, pi)

try:
    # Running in PyCharm
    import res.colors as Color
    from Setup import *
    from Util import world_to_pixels
    from task_race.RaceCircle import Triangle, Circle, Quadrilateral
    import Util
except NameError as err:
    print(err, "--> our error message")
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running from command line -> Import libraries as package")
    from ..res import colors as Color
    from ..Setup import *
    from Util import world_to_pixels
    import Util

class RaceCircleV2(object):
    def __init__(self, screen=None, world=None):
        self.screen = screen

        radius = SCREEN_WIDTH / PPM / 2  # 18
        max_x = SCREEN_WIDTH / PPM  # 36
        max_y = SCREEN_HEIGHT / PPM  # 36

        top_right_tri = [Triangle(screen=screen, world=world,
                                  vertices=[
                                      (0, 0),
                                      (np.cos((i + 1) * np.pi / 12) * radius + radius - max_x,
                                       np.sin((i + 1) * np.pi / 12) * radius + radius - max_y),
                                      (np.cos(i * np.pi / 12) * radius + radius - max_x,
                                       np.sin(i * np.pi / 12) * radius + radius - max_y)
                                  ],
                                  x=max_x, y=max_y)
                         for i in range(0, 6)]

        top_left_tri = [Triangle(screen=screen, world=world,
                                 vertices=[
                                     (0, 0),
                                     (max_x - (np.cos((i - 1) * np.pi / 12) * radius + radius),
                                      np.sin((i - 1) * np.pi / 12) * radius + radius - max_y),
                                     (max_x - (np.cos(i * np.pi / 12) * radius + radius),
                                      np.sin(i * np.pi / 12) * radius + radius - max_y)
                                 ],
                                 x=0, y=max_y)
                        for i in range(6, 0, -1)]

        bottom_left_tri = [Triangle(screen=screen, world=world,
                                    vertices=[
                                        (0, 0),
                                        (max_x - (np.cos((i - 1) * np.pi / 12) * radius + radius),
                                         max_y - (np.sin((i - 1) * np.pi / 12) * radius + radius)),
                                        (max_x - (np.cos(i * np.pi / 12) * radius + radius),
                                         max_y - (np.sin(i * np.pi / 12) * radius + radius))
                                    ],
                                    x=0, y=0)
                           for i in range(6, 0, -1)]

        bottom_right_tri = [Triangle(screen=screen, world=world,
                                     vertices=[
                                         (0, 0),
                                         (np.cos((i + 1) * np.pi / 12) * radius + radius - max_x,
                                          max_y - (np.sin((i + 1) * np.pi / 12) * radius + radius)),
                                         (np.cos(i * np.pi / 12) * radius + radius - max_x,
                                          max_y - (np.sin(i * np.pi / 12) * radius + radius))
                                     ],
                                     x=max_x, y=0)
                            for i in range(0, 6)]

        self.triangle_list = top_right_tri + top_left_tri + bottom_left_tri + bottom_right_tri

        self.inner_circle = Circle(screen=self.screen, world=world,
                                   x=radius, y=radius,
                                   radius=radius - 7)

        r = (SCREEN_WIDTH / PPM / 2) - 7  # 11
        c = vec2(max_x / 2, max_y / 2)  # centre

        self.wall = Quadrilateral(screen=screen, world=world,
                                  color=Color.Gray,
                                  x=c.x,
                                  y=c.y,
                                  vertices=[
                                      (-np.cos(2 * np.pi / 12) * r, np.sin(2 * np.pi / 12) * r),
                                      (-np.cos(2 * np.pi / 12) * (r + 7), np.sin(2 * np.pi / 12) * (r + 7)),
                                      (-np.cos(3 * np.pi / 12) * (r + 7), np.sin(3 * np.pi / 12) * (r + 7)),
                                      (-np.cos(3 * np.pi / 12) * r, np.sin(3 * np.pi / 12) * r)
                                  ], )

    def draw(self):
        for triangle in self.triangle_list:
            triangle.draw()

        self.wall.draw()

        self.inner_circle.draw()

        # Goal square
        M1 = vec2(-np.cos(3*(np.pi/12)) * 11 + 18, 18)
        M2 = vec2(0, 18)
        M3 = vec2(0, np.sin(3*(np.pi/12)) * 11 + 18)
        M4 = vec2(-np.cos(3*(np.pi/12)) * 11 + 18, np.sin(3*(np.pi/12)) * 11 + 18)
        pygame.draw.line(self.screen, Color.Orange, world_to_pixels(M1), world_to_pixels(M2))
        pygame.draw.line(self.screen, Color.Orange, world_to_pixels(M2), world_to_pixels(M3))
        pygame.draw.line(self.screen, Color.Orange, world_to_pixels(M3), world_to_pixels(M4))
        pygame.draw.line(self.screen, Color.Orange, world_to_pixels(M4), world_to_pixels(M1))