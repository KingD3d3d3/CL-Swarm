

import random
import argparse
import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, vec2)
from pygame.locals import *
# import matplotlib.pyplot as plt
import time
import os
import errno
import csv
import numpy as np
import sys
import datetime
import gc
import matplotlib.pyplot as plt

try:
    # Running in PyCharm
    from AgentHoming import AgentHoming
    import res.colors as Color
    from objects.Border import Border
    from objects.Circle import StaticCircle
    from objects.Box import StaticBox
    from Setup import *
    from MyContactListener import MyContactListener
    from Util import world_to_pixels, pixels_to_world
    import Util
    import debug_homing
    from res.print_colors import *
    import global_homing
    import res.print_colors as PrintColor
    import Global
except NameError as err:
    print(err, "--> our error message")
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running from command line -> Import libraries as package")
    from .AgentHoming import AgentHoming
    from ..objects.Border import Border
    from ..objects.Circle import StaticCircle
    from ..res import colors as Color
    from .MyContactListener import MyContactListener
    from ..Setup import *
    from ..Util import world_to_pixels, pixels_to_world
    from .. import Util
    import debug_homing
    from ..res.print_colors import *
    import global_homing
    from ..res import print_colors as PrintColor
    from .. import Global


class EnvironmentHoming(object):
    def __init__(self, render=False, fixed_ur_timestep=False, num_agents=1):

        debug_homing.xprint(color=PRINT_GREEN, msg='Creating Environment, Physics Setup')

        self.render = render
        self.fixed_ur_timestep = fixed_ur_timestep

        # -------------------- Pygame Setup ----------------------

        pygame.init()

        self.deltaTime = 1.0 / TARGET_FPS  # 0.016666
        self.fps = TARGET_FPS
        self.accumulator = 0

        self.screen = None
        if self.render:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
            pygame.display.set_caption('Testbed Homing')
        self.clock = pygame.time.Clock()
        self.myfont = pygame.font.SysFont("monospace", 15)

        # -------------------- Environment and PyBox2d World Setup ----------------------

        # Goal positions
        goal1 = (100, 100)
        goal2 = (SCREEN_WIDTH - goal1[0], SCREEN_HEIGHT - goal1[1])  # 1180, 620
        self.goals_pixels = [goal1, goal2]
        goal1_vec = pixels_to_world(goal1)  # b2Vec2(5,31)
        goal2_vec = vec2(SCREEN_WIDTH / PPM - goal1_vec.x, SCREEN_HEIGHT / PPM - goal1_vec.y)  # b2Vec2(59,5)
        self.goals_vec = [goal1_vec, goal2_vec]

        # Create the world
        self.world = world(gravity=(0, 0), doSleep=True, contactListener=MyContactListener())

        # Border of the map
        self.border = Border(screen=self.screen, world=self.world)

        # Agents
        self.numAgents = num_agents  # total numbers of agents in the simulation
        self.agents = []
        for j in range(num_agents):
            start_pos = pixels_to_world(self.goals_pixels[1])  # start from goal 2

            toGoal = Util.normalize(pixels_to_world(self.goals_pixels[0]) - start_pos)
            forward = vec2(0, 1)
            angleDeg = Util.angle_indirect(forward, toGoal)
            angle = Util.deg_to_rad(angleDeg)
            angle = -angle  # start by looking at goal1
            a = AgentHoming(screen=self.screen, world=self.world, x=start_pos.x, y=start_pos.y, angle=angle,
                            radius=1.5, id=j, goals=self.goals_vec, numAgents=num_agents)
            self.agents.append(a)

        # Obstacles
        self.obstacles = []
        radius = 1.5 # 2 --> too big

        # 1
        # circle1 = StaticCircle(screen=self.screen, world=self.world, x=12.75, y=24, radius=radius)
        circle1 = StaticCircle(screen=self.screen, world=self.world, x=17.5, y=21.75, radius=radius)
        circle1.id = 1

        # 7
        # circle7 = StaticCircle(screen=self.screen, world=self.world, x=51.25, y=12, radius=radius)
        circle7 = StaticCircle(screen=self.screen, world=self.world, x=46.5, y=14.25, radius=radius)
        circle7.id = 7

        # 2
        # circle2 = StaticCircle(screen=self.screen, world=self.world, x=26, y=28, radius=radius)
        circle2 = StaticCircle(screen=self.screen, world=self.world, x=30, y=29, radius=radius)
        circle2.id = 2

        # 6
        # circle6 = StaticCircle(screen=self.screen, world=self.world, x=38, y=8, radius=radius)
        circle6 = StaticCircle(screen=self.screen, world=self.world, x=34, y=7, radius=radius)
        circle6.id = 6

        # 3
        circle3 = StaticCircle(screen=self.screen, world=self.world, x=14.75, y=10, radius=radius)
        circle3.id = 3

        # 5
        circle5 = StaticCircle(screen=self.screen, world=self.world, x=49.25, y=26, radius=radius)
        circle5.id = 5

        # 4
        circle4 = StaticCircle(screen=self.screen, world=self.world, x=32, y=18, radius=radius)
        circle4.id = 4

        self.obstacles.extend((circle1, circle2, circle3, circle4, circle5, circle6, circle7))
        self.numObstacles = len(self.obstacles)

    def draw(self):
        """
            Rendering part
        """
        if not self.render:
            return

        # Reset screen's pixels
        self.screen.fill((0, 0, 0, 0))

        goalFont = pygame.font.SysFont("monospace", 25)

        # Draw Goal 1
        pygame.draw.circle(self.screen, Color.Red, self.goals_pixels[0], 20)
        self.screen.blit(goalFont.render('1', True, Color.White),
                         (self.goals_pixels[0][0] - 8, self.goals_pixels[0][1] - 12))

        # Draw Goal 2
        pygame.draw.circle(self.screen, Color.Red, self.goals_pixels[1], 20)
        self.screen.blit(goalFont.render('2', True, Color.White),
                         (self.goals_pixels[1][0] - 8, self.goals_pixels[1][1] - 12))

        # Moving Objects
        for j in range(len(self.agents)):
            self.agents[j].draw()

        # Obstacles
        for i in range(self.numObstacles):
            self.obstacles[i].draw()

        # Boundary
        self.border.draw()

        # Show FPS
        Util.print_fps(self.screen, self.myfont, 'FPS : ' + str('{:3.2f}').format(self.fps))

        # Flip the screen
        pygame.display.flip()

    def update(self):
        """
            Call agent's update
        """
        for j in range(len(self.agents)):
            self.agents[j].update()

    def fps_physic_step(self):
        """
            FPS and Physic's Step Part
        """
        # With rendering
        if self.render:
            self.deltaTime = self.clock.tick(TARGET_FPS) / 1000.0
            self.fps = self.clock.get_fps()

            # Fixed your timestep
            if self.fixed_ur_timestep:
                """
                    'Fixed your timestep' technique.
                    Physics is stepped by a fixed amount e.g. 1/60s.
                    Faster machine e.g. render at 120fps -> step the physics one every 2 frames
                    Slower machine e.g. render at 30fps -> run the physics twice
                """
                self.accumulator += self.deltaTime

                while self.accumulator >= PHYSICS_TIME_STEP:
                    # Physic step
                    self.world.Step(PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS)
                    self.world.ClearForces()

                    self.accumulator -= PHYSICS_TIME_STEP

                return

            # Frame dependent
            else:
                # Physic step
                self.world.Step(PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS)
                self.world.ClearForces()

                return

        # No rendering
        elif not self.render:
            self.deltaTime = self.clock.tick() / 1000.0
            # self.fps = self.clock.get_fps()

            # Frame dependent -- Physic step
            self.world.Step(PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS)
            self.world.ClearForces()

            return
