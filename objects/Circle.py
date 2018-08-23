from __future__ import division
import pygame
from pygame.locals import *
import random, sys

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


class Circle(object):
    def __init__(self, screen=None, world=None, x=0, y=0, radius=1):
        self.screen = screen
        self.radius = radius
        self.body = world.CreateDynamicBody(
            position=(x, y), userData=self)
        self.fixture = self.body.CreateCircleFixture(
            radius=radius, density=1, friction=0.3)
        self.color = Color.Lime

    def draw(self):
        position = self.body.transform * self.fixture.shape.pos * PPM
        position = (position[0], SCREEN_HEIGHT - position[1])
        pygame.draw.circle(self.screen, self.color, [int(x) for x in position], int(self.radius * PPM))

        current_forward_normal = self.body.GetWorldVector((0, 1))
        pygame.draw.line(self.screen, Color.White, worldToPixels(self.body.worldCenter),
                         worldToPixels(self.body.worldCenter + current_forward_normal * self.radius))


class StaticCircle(object):
    def __init__(self, screen=None, world=None, x=0, y=0, radius=1):
        self.screen = screen
        self.id = random.randint(0, sys.maxint)

        self.radius = radius
        self.body = world.CreateStaticBody(
            position=(x, y), userData=self)
        self.fixture = self.body.CreateCircleFixture(
            radius=radius, density=1, friction=0, restitution=0)  #density=1, friction=0.3)
        self.color = Color.Lime

    def draw(self):
        position = self.body.transform * self.fixture.shape.pos * PPM
        position = (position[0], SCREEN_HEIGHT - position[1])
        pygame.draw.circle(self.screen, self.color, [int(x) for x in position], int(self.radius * PPM))

        # Id of the obstacle
        font = pygame.font.SysFont("monospace", 22)
        idText = font.render(str(self.id), True, Color.LightBlack)
        offset = [idText.get_rect().width / 2.0, idText.get_rect().height / 2.0]  # to adjust center
        idPos = (self.body.transform * (0, 0)) * PPM
        idPos = (idPos[0] - offset[0], SCREEN_HEIGHT - idPos[1] - offset[1])
        self.screen.blit(idText, idPos)