import pygame
from Box2D.b2 import (polygonShape)
import res.colors as Color

class Circle(object):
    def __init__(self, screen=None, world=None, x=0, y=0, radius=1, color=Color.Gray, screen_height=None, screen_width=None, ppm=20):
        self.screen = screen
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.ppm = ppm

        self.radius = radius
        self.body = world.CreateStaticBody(position=(x, y), userData=self)
        self.fixture = self.body.CreateCircleFixture(radius=radius, density=1, friction=0, restitution=0)
        self.color = color

    def render(self):
        position = self.body.transform * self.fixture.shape.pos * self.ppm
        position = (position[0], self.screen_height - position[1])
        pygame.draw.circle(self.screen, self.color, [int(x) for x in position], int(self.radius * self.ppm))

class Triangle(object):
    def __init__(self, screen=None, world=None, x=0, y=0, vertices=None, color=Color.Gray, screen_height=None, screen_width=None, ppm=20):
        if vertices is None:
            vertices = [(0, 0), (1, 1), (0, 1)]

        self.screen = screen
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.ppm = ppm

        self.body = world.CreateStaticBody(position=(x, y), shapes=polygonShape(vertices=vertices), userData=self)
        self.color = color

    def render(self):
        vertices = [(self.body.transform * v) * self.ppm for v in self.body.fixtures[0].shape.vertices]
        vertices = [(v[0], self.screen_height - v[1]) for v in vertices]
        pygame.draw.polygon(self.screen, self.color, vertices)

class Quadrilateral(object):
    def __init__(self, screen=None, world=None, x=0, y=0, vertices=None, color=Color.Gray, screen_height=None, screen_width=None, ppm=20):
        if vertices is None:
            vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]

        self.screen = screen
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.ppm = ppm

        self.body = world.CreateStaticBody(position=(x, y), shapes=polygonShape(vertices=vertices), userData=self)
        self.color = color

    def render(self):
        vertices = [(self.body.transform * v) * self.ppm for v in self.body.fixtures[0].shape.vertices]
        vertices = [(v[0], self.screen_height - v[1]) for v in vertices]
        pygame.draw.polygon(self.screen, self.color, vertices)
