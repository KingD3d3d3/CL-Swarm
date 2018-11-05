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


class Circle(object):
    def __init__(self, screen=None, world=None, x=0, y=0, radius=1, color=Color.Gray):
        self.screen = screen
        self.id = random.randint(0, 1000000)

        self.radius = radius
        self.body = world.CreateStaticBody(
            position=(x, y), userData=self)
        self.fixture = self.body.CreateCircleFixture(
            radius=radius, density=1, friction=0, restitution=0)
        self.color = color

    def draw(self):
        position = self.body.transform * self.fixture.shape.pos * PPM
        position = (position[0], SCREEN_HEIGHT - position[1])
        pygame.draw.circle(self.screen, self.color, [int(x) for x in position], int(self.radius * PPM))


class Triangle(object):
    def __init__(self, screen=None, world=None, x=0, y=0, vertices=None, color=Color.Gray):
        if vertices is None:
            vertices = [(0, 0), (1, 1), (0, 1)]

        self.screen = screen
        self.body = world.CreateStaticBody(
            position=(x, y),
            shapes=polygonShape(vertices=vertices),
            userData=self
        )
        self.color = color

    def draw(self):
        vertices = [(self.body.transform * v) * PPM for v in self.body.fixtures[0].shape.vertices]
        vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
        pygame.draw.polygon(self.screen, self.color, vertices)


class Quadrilateral(object):
    def __init__(self, screen=None, world=None, x=0, y=0, vertices=None, color=Color.Yellow):
        if vertices is None:
            vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]

        self.screen = screen
        self.body = world.CreateStaticBody(
            position=(x, y),
            shapes=polygonShape(vertices=vertices),
            userData=self
        )
        self.color = color

    def draw(self):
        vertices = [(self.body.transform * v) * PPM for v in self.body.fixtures[0].shape.vertices]
        vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
        pygame.draw.polygon(self.screen, self.color, vertices)


class RaceCircle(object):
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
                                      (np.cos(2 * np.pi / 12) * r, np.sin(2 * np.pi / 12) * r),
                                      (np.cos(2 * np.pi / 12) * (r + 7), np.sin(2 * np.pi / 12) * (r + 7)),
                                      (np.cos(3 * np.pi / 12) * (r + 7), np.sin(3 * np.pi / 12) * (r + 7)),
                                      (np.cos(3 * np.pi / 12) * r, np.sin(3 * np.pi / 12) * r)
                                  ], )
        #
        # self.wall = Quadrilateral(screen=screen, world=world,
        #                           color=Color.Yellow,
        #                           x=np.cos(3 * np.pi / 12) * r + c.x,
        #                           y=np.sin(3 * np.pi / 12) * r + c.y,
        #                           vertices=[
        #                               (0, 0),
        #                               (np.cos(3 * np.pi / 12) * 7, np.sin(3 * np.pi / 12) * 7),
        #                               (np.cos(4 * np.pi / 12) * 7, np.sin(4 * np.pi / 12) * 7),
        #                               (np.cos((np.pi / 2) + (4 * np.pi/12)), np.sin(np.pi / 2) + (4 * np.pi/12)),
        #                           ], )

        # (np.cos((i + 1) * np.pi / 12) * radius + radius - max_x,
        #  np.sin((i + 1) * np.pi / 12) * radius + radius - max_y),
        # (np.cos(i * np.pi / 12) * radius + radius - max_x,
        #  np.sin(i * np.pi / 12) * radius + radius - max_y)


        # self.goal = Quadrilateral(screen=screen, world=world, x=c.x + r - 1, y=c.y,
        #                           vertices=[
        #                      (0, 0),
        #                      (8, 0),
        #                      (8, 0.5),
        #                      (0, 0.5)
        #                  ], )

    def draw(self):
        for triangle in self.triangle_list:
            triangle.draw()

        self.wall.draw()
        # self.goal.draw()

        self.inner_circle.draw()

        # Goal square
        M1 = vec2(np.cos(3*(np.pi/12)) * 11 + 18, 18)
        M2 = vec2(36, 18)
        M3 = vec2(36, np.sin(3*(np.pi/12)) * 11 + 18)
        M4 = vec2(np.cos(3*(np.pi/12)) * 11 + 18, np.sin(3*(np.pi/12)) * 11 + 18)
        pygame.draw.line(self.screen, Color.Orange, world_to_pixels(M1), world_to_pixels(M2))
        pygame.draw.line(self.screen, Color.Orange, world_to_pixels(M2), world_to_pixels(M3))
        pygame.draw.line(self.screen, Color.Orange, world_to_pixels(M3), world_to_pixels(M4))
        pygame.draw.line(self.screen, Color.Orange, world_to_pixels(M4), world_to_pixels(M1))