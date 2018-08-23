from __future__ import division
import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import vec2
from pygame.locals import *
import random
import sys
from Box2D.b2 import (polygonShape)

try:
    from Util import worldToPixels
    from Setup import *
    import res.colors as Color
except:
    # Running in command line
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running from command line -> Import libraries as package')
    from ..Util import worldToPixels
    from ..Setup import *
    from ..res import colors as Color

class Box(object):
    def __init__(self, screen=None, world=None, x=0, y=0, width=1, height=1, angle=0):
        self.screen = screen
        self.height = height
        self.body = world.CreateDynamicBody(
            position=(x, y), angle=angle)
        self.fixture = self.body.CreatePolygonFixture(
            box=(width, height), density=1, friction=0.3, )

    def draw(self):
        vertices = [(self.body.transform * v) * PPM for v in self.fixture.shape.vertices]
        vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
        pygame.draw.polygon(self.screen, Color.Red, vertices)

        p = self.body.GetWorldPoint(localPoint=(0, self.height))  # upper point of the box
        forward = self.body.GetWorldVector((0, 1))  # transform.forward of this box
        pygame.draw.line(self.screen, Color.White, worldToPixels(vec2(p)),
                         worldToPixels(vec2(p) + forward))


class StaticBox(object):
    def __init__(self, screen=None, world=None, x=0, y=0, width=1, height=1, angle=0):
        self.screen = screen
        self.body = world.CreateStaticBody(
            position=(x, y),
            angle=angle,
            shapes=polygonShape(box=(width, height)),
            userData=self
        )
        self.id = random.randint(0, sys.maxint)
        self.color = Color.Lime

    def draw(self):
        vertices = [(self.body.transform * v) * PPM for v in self.body.fixtures[0].shape.vertices]
        pygame.draw.polygon(self.screen, self.color, vertices)
