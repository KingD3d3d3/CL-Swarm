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
    from .res import colors as Color
    from .Setup import *


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
        self.boxList = [bottom, right, top, left]

    def draw(self):
        for box in self.boxList:
            box.draw()

