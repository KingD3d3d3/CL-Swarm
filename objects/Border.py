from __future__ import division
import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (polygonShape)
import random
import sys

try:
    # Running in PyCharm
    import res.colors as Color
    from Setup import *
except:
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running from command line -> Import libraries as package')
    from ..res import colors as Color
    from ..Setup import *


class Wall(object):
    def __init__(self, screen=None, world=None, x=0, y=0, width=50, height=1, color=Color.Gray):
        self.screen = screen
        self.body = world.CreateStaticBody(
            position=(x, y),
            shapes=polygonShape(box=(width, height)),  # (half_width, half_height)
            userData=self
        )
        self.id = random.randint(0, sys.maxint)
        self.color = color

    def draw(self):
        vertices = [(self.body.transform * v) * PPM for v in self.body.fixtures[0].shape.vertices]
        vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
        pygame.draw.polygon(self.screen, self.color, vertices)

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

class Border(object):
    def __init__(self, screen=None, world=None):
        self.screen = screen
        bottom = Wall(screen=screen, world=world,
                      x=int((SCREEN_WIDTH / PPM) / 2),
                      y=0,
                      width=int((SCREEN_WIDTH / PPM) / 2),
                      height=1)
        bottom.id = 0
        right = Wall(screen=screen, world=world,
                     x=int(SCREEN_WIDTH / PPM),
                     y=int((SCREEN_HEIGHT / PPM) / 2),
                     width=1,
                     height=int((SCREEN_HEIGHT / PPM) / 2))
        right.id = 1
        top = Wall(screen=screen, world=world,
                   x=int((SCREEN_WIDTH / PPM) / 2),
                   y=int(SCREEN_HEIGHT / PPM),
                   width=int((SCREEN_WIDTH / PPM) / 2),
                   height=1)
        top.id = 2
        left = Wall(screen=screen, world=world,
                    x=0,
                    y=int((SCREEN_HEIGHT / PPM) / 2),
                    width=1,
                    height=int((SCREEN_HEIGHT / PPM) / 2))
        left.id = 3
        self.box_list = [bottom, right, top, left]

        # Corner Top Left
        top_left_trig1 = Triangle(screen=screen, world=world, vertices=[(0, 0), (2, 2), (0, 2)],
                                 x=1, y=int((SCREEN_HEIGHT / PPM) - 3))
        top_left_trig2 = Triangle(screen=screen, world=world, vertices=[(0, -1), (1, 2), (0, 2)],
                            x=1, y=int((SCREEN_HEIGHT / PPM) - 3))
        top_left_trig3 = Triangle(screen=screen, world=world, vertices=[(0, 1), (3, 2), (0, 2)],
                            x=1, y=int((SCREEN_HEIGHT / PPM) - 3))

        # Corner Top Right
        top_right_trig1 = Triangle(screen=screen, world=world, vertices=[(0, 2), (2, 0), (2, 2)],
                                  x=int((SCREEN_WIDTH / PPM) - 3), y=int((SCREEN_HEIGHT / PPM) - 3))
        top_right_trig2 = Triangle(screen=screen, world=world, vertices=[(-1, 2), (2, 1), (2, 2)],
                                   x=int((SCREEN_WIDTH / PPM) - 3), y=int((SCREEN_HEIGHT / PPM) - 3))
        top_right_trig3 = Triangle(screen=screen, world=world, vertices=[(1, 2), (2, -1), (2, 2)],
                                   x=int((SCREEN_WIDTH / PPM) - 3), y=int((SCREEN_HEIGHT / PPM) - 3))

        # Corner Bottom Left
        bottom_left_trig1 = Triangle(screen=screen, world=world, vertices=[(0, 0), (2, 0), (0, 2)],
                                    x=1, y=1)
        bottom_left_trig2 = Triangle(screen=screen, world=world, vertices=[(0, 0), (1, 0), (0, 3)],
                            x=1, y=1)
        bottom_left_trig3 = Triangle(screen=screen, world=world, vertices=[(0, 0), (3, 0), (0, 1)],
                            x=1, y=1)




        # Corner Bottom Right
        bottom_right_trig1 = Triangle(screen=screen, world=world, vertices=[(0, 0), (2, 0), (2, 2)],
                                     x=int((SCREEN_WIDTH / PPM) - 3), y=1)
        bottom_right_trig2 = Triangle(screen=screen, world=world, vertices=[(-1, 0), (2, 0), (2, 1)],
                            x=int((SCREEN_WIDTH / PPM) - 3), y=1)
        bottom_right_trig3 = Triangle(screen=screen, world=world, vertices=[(1, 0), (2, 0), (2, 3)],
                            x=int((SCREEN_WIDTH / PPM) - 3), y=1)


        self.triangle_list = [top_left_trig1, top_left_trig2, top_left_trig3,
                              top_right_trig1, top_right_trig2, top_right_trig3,
                              bottom_left_trig1, bottom_left_trig2, bottom_left_trig3,
                              bottom_right_trig1, bottom_right_trig2, bottom_right_trig3
                              ]

    def draw(self):
        for box in self.box_list:
            box.draw()

        for triangle in self.triangle_list:
            triangle.draw()
