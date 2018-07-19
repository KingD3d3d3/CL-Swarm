from __future__ import division

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
    from AgentHomingSimple import AgentHomingSimple
    from AgentHomingPerfect import AgentHomingPerfect
    import res.colors as Color
    from Border import Border
    from Circle import StaticCircle
    from Box import StaticBox
    from MyContactListener import MyContactListener
    from Setup import *
    from Util import worldToPixels, pixelsToWorld
    import Util
    import debug_homing_simple
    from res.print_colors import *
    import global_homing_simple
    import res.print_colors as PrintColor
    import Global
except:
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running from command line -> Import libraries as package')
    from .AgentHomingSimple import AgentHomingSimple
    from .AgentHomingPerfect import AgentHomingPerfect
    from ..Border import Border
    from ..Circle import StaticCircle
    from ..res import colors as Color
    from .MyContactListener import MyContactListener
    from ..Setup import *
    from ..Util import worldToPixels, pixelsToWorld
    from .. import Util
    import debug_homing_simple
    from ..res.print_colors import *
    import global_homing_simple
    from ..res import print_colors as PrintColor
    from .. import Global

class EnvironmentHomingSimple(object):

    def __init__(self, render=False, fixed_ur_timestep=False, num_agents=1):

        debug_homing_simple.xprint(color=PRINT_GREEN, msg='Creating Environment, Physics Setup')

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
            pygame.display.set_caption('Testbed Parameters Sharing')
        self.clock = pygame.time.Clock()
        self.myfont = pygame.font.SysFont("monospace", 15)

        # -------------------- Environment and PyBox2d World Setup ----------------------

        # Goal positions
        goal1 = (100, 100)
        goal2 = (SCREEN_WIDTH - goal1[0], SCREEN_HEIGHT - goal1[1]) # 1180, 620
        self.goals_pixels = [goal1, goal2]
        goal1_vec = pixelsToWorld(goal1) # b2Vec2(5,31)
        goal2_vec = vec2(SCREEN_WIDTH / PPM - goal1_vec.x, SCREEN_HEIGHT / PPM - goal1_vec.y) # b2Vec2(59,5)
        self.goals_vec = [goal1_vec, goal2_vec]

        # Create the world
        self.world = world(gravity=(0, 0), doSleep=True)

        # Border of the map
        self.border = Border(screen=self.screen, world=self.world)

        # Agents
        self.numAgents = num_agents  # total numbers of agents in the simulation
        self.agents = []
        for j in xrange(num_agents):
            start_pos = pixelsToWorld(self.goals_pixels[1])  # start from goal 2

            toGoal = Util.normalize(pixelsToWorld(self.goals_pixels[0]) - start_pos)
            forward = vec2(0, 1)
            angleDeg = Util.angle(forward, toGoal)
            angle = Util.degToRad(angleDeg)
            angle = -angle  # start by looking at goal1
            a = AgentHomingSimple(screen=self.screen, world=self.world, x=start_pos.x, y=start_pos.y, angle=angle,
                                  radius=1.5, id=j, goals=self.goals_vec)
            self.agents.append(a)

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
        for j in xrange(len(self.agents)):
            self.agents[j].draw()

        # Boundary
        self.border.draw()

        # Flip the screen
        pygame.display.flip()

    def update(self):
        """
            Call agent's update
        """
        for j in xrange(len(self.agents)):
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
                while self.accumulator >= PHYSICS_TIME_STEP:

                    # Physic step
                    self.world.Step(PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS)
                    self.world.ClearForces()

                    self.accumulator -= PHYSICS_TIME_STEP

            # Frame dependent
            else:
                # Physic step
                self.world.Step(PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS)
                self.world.ClearForces()

        # No rendering
        elif not self.render:
            self.deltaTime = self.clock.tick() / 1000.0
            self.fps = self.clock.get_fps()

            # Frame dependent
            if self.deltaTime <= TARGET_FPS:  # Frame is faster than target (60fps) -> simulation run faster

                # Physic step
                self.world.Step(PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS)
                self.world.ClearForces()

            # Fixed your timestep
            elif self.fixed_ur_timestep:
                self.accumulator += self.deltaTime

                while self.accumulator >= PHYSICS_TIME_STEP:

                    # Physic step
                    self.world.Step(PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS)
                    self.world.ClearForces()

                    self.accumulator -= PHYSICS_TIME_STEP