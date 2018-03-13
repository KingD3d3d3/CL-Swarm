import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (polygonShape)

import res.colors as Color
from Setup import SCREEN_WIDTH, SCREEN_HEIGHT, PPM


class Wall:
    def __init__(self, screen=None, world=None, x=0, y=0, width=50, height=1):
        self.screen = screen
        self.body = world.CreateStaticBody(
            position=(x, y),
            shapes=polygonShape(box=(width, height)),
            userData=self
        )

    def draw(self):
        vertices = [(self.body.transform * v) * PPM for v in self.body.fixtures[0].shape.vertices]
        vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
        pygame.draw.polygon(self.screen, Color.Gray, vertices)


class Border:
    def __init__(self, screen=None, world=None):
        self.screen = screen
        bottom = Wall(screen=screen, world=world, x=0, y=0, width=int(SCREEN_WIDTH / PPM), height=1)
        top = Wall(screen=screen, world=world, x=0, y=int(SCREEN_HEIGHT / PPM), width=int(SCREEN_WIDTH / PPM), height=1)
        right = Wall(screen=screen, world=world, x=0, y=0, width=1, height=int(SCREEN_HEIGHT / PPM))
        left = Wall(screen=screen, world=world, x=int(SCREEN_WIDTH / PPM), y=0, width=1, height=int(SCREEN_HEIGHT / PPM))
        self.boxList = [bottom, top, right, left]

    def draw(self):
        for box in self.boxList:
            box.draw()

