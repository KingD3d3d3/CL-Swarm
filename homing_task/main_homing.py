#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import argparse
import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world)
from pygame.locals import *
import os
import errno
import time
import csv

try:
    # Running in PyCharm
    from AgentHoming import AgentHoming
    import res.colors as Color
    from Border import Border
    from Circle import StaticCircle
    from Box import StaticBox
    from MyContactListener import MyContactListener
    from Setup import *
    from Util import *
    from experimental.EntityManager import EntityManager
    from experimental.Entity import Entity
    import homing_global
except:
    # Running in command line
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running from command line -> Import libraries as package')
    from .AgentHoming import AgentHoming
    from ..Border import Border
    from ..Circle import StaticCircle
    from ..res import colors as Color
    from .MyContactListener import MyContactListener
    from ..Setup import *
    from ..Util import *
    import homing_global

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Homing Task')
    parser.add_argument('--render', help='render the simulation', default='True')
    parser.add_argument('--print_fps', help='print fps', default='False')
    parser.add_argument('--debug', help='print simulation log', default='True')
    parser.add_argument('--record', help='record simulation log in file', default='False')
    args = parser.parse_args()
    render = args.render == 'True'
    print_fps = args.print_fps == 'True'
    homing_global.debug = args.debug == 'True'
    homing_global.record = args.record == 'True'

    deltaTime = 1.0 / TARGET_FPS
    fps = 1.0 / deltaTime

    accumulator = 0
    interpolated = False

    # -------------------- Pygame Setup ----------------------

    pygame.init()
    screen = None
    if render:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
        pygame.display.set_caption('Homing Task Testbed')
    clock = pygame.time.Clock()

    myfont = pygame.font.SysFont("monospace", 15)
    def PrintFPS(text, color=(255, 0, 0, 255)):
        """
        Draw some text at the top status lines
        and advance to the next line.
        """
        screen.blit(myfont.render(
            text, True, color), (10, 3))


    # -------------------- PyBox2d World Setup ----------------------

    # Create the world
    world = world(gravity=(0, 0), doSleep=True, contactListener=MyContactListener())  # gravity = (0, -10)

    # Border of the map
    border = Border(screen=screen, world=world)

    # Agent
    numAgents = 2
    goal_threshold = 100
    agents = []
    for i in xrange(numAgents):
        randX = random.randint(2, SCREEN_WIDTH / PPM - 2)
        randY = random.randint(2, SCREEN_HEIGHT / PPM - 2)
        randAngle = degToRad(random.randint(0, 360))
        a = AgentHoming(screen=screen, world=world, x=randX, y=randY, angle=randAngle,
                        radius=1.5, goal_threshold=goal_threshold, id=i)
        agents.append(a)

    # Obstacles
    # box1 = StaticBox(screen=screen, world=world, x=20, y=17, width=2, height=2)
    # box1.id = 1
    # box2 = StaticBox(screen=screen, world=world, x=40, y=20, width=2, height=2)
    # box2.id = 2
    # box3 = StaticBox(screen=screen, world=world, x=50, y=10, width=2, height=2)
    # box3.id = 3
    # box4 = StaticBox(screen=screen, world=world, x=10, y=25, width=2, height=2)
    # box4.id = 4

    circle1 = StaticCircle(screen=screen, world=world, x=20, y=17, radius=2)
    circle1.id = 1
    circle2 = StaticCircle(screen=screen, world=world, x=40, y=20, radius=2)
    circle2.id = 2
    circle3 = StaticCircle(screen=screen, world=world, x=50, y=10, radius=2)
    circle3.id = 3
    circle4 = StaticCircle(screen=screen, world=world, x=10, y=25, radius=2)
    circle4.id = 4
    # -------------------- Main Game Loop ----------------------

    running = True
    pause = False
    recording = False
    f = None
    while running:
        if render:
            # Check the event queue
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    # The user closed the window or pressed escape
                    running = False
                if event.type == KEYDOWN and event.key == K_p:
                    pause = not pause  # Pause the game
                if event.type == KEYDOWN and event.key == K_r:
                    if not homing_global.record:  # Record simulation
                        print("start recording")
                        homing_global.record = True
                        filename = homing_global.fileCreate()
                        homing_global.fo = open(filename, 'a')
                    else:                         # Stop recording
                        print("stop recording")
                        homing_global.record = False
                        homing_global.fo.close()

        # Pause the game
        if pause:
            # clock.tick compute how many milliseconds have passed since the previous call.
            # If we don't call that during pause, clock.tick will compute time spend during pause
            # thus timer is counting during simulation pause -> we want to avoid that !
            if render:
                deltaTime = clock.tick(TARGET_FPS) / 1000.0
            elif not render:
                deltaTime = clock.tick() / 1000.0
            continue  # go back to loop entry

        # if recording:
        #     f.write("hello world {}\n".format(homing_global.timestep))

        if render:
            screen.fill((0, 0, 0, 0))

        # Update the agents
        for i in xrange(numAgents):
            agents[i].update()

        # ---------------------- FPS Physics Step Part -----------
        if render:
            deltaTime = clock.tick(TARGET_FPS) / 1000.0
            fps = clock.get_fps()

            # "Fixed your timestep" technique
            # Physics is stepped by a fixed amount i.e. 1/60s.
            # Faster machine i.e. render at 120fps -> step the physics one every 2 frames
            # Slower machine i.e. render at 30fps -> run the physics twice
            accumulator += deltaTime
            while accumulator >= PHYSICS_TIME_STEP:

                # Physic step
                world.Step(PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS)
                world.ClearForces()

                accumulator -= PHYSICS_TIME_STEP

        elif not render:
            deltaTime = clock.tick() / 1000.0
            fps = clock.get_fps()

            if deltaTime >= TARGET_FPS: # Frame is faster than target (60fps) -> simulation run faster
                accumulator = 0

                # Physic step
                world.Step(PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS)
                world.ClearForces()
            else:
                accumulator += deltaTime

                while accumulator >= PHYSICS_TIME_STEP:

                    # Physic step
                    world.Step(PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS)
                    world.ClearForces()

                    accumulator -= PHYSICS_TIME_STEP

        # ---------------------------------------------------------

        # ---------------------- Rendering Part -------------------
        if render:
            # Draw goals
            pygame.draw.circle(screen, Color.Red, (goal_threshold, goal_threshold), 20)
            goalFont = pygame.font.SysFont("monospace", 25)
            screen.blit(goalFont.render('1', True, Color.White), (goal_threshold - 8, goal_threshold - 12))
            pygame.draw.circle(screen, Color.Red, (SCREEN_WIDTH - goal_threshold, SCREEN_HEIGHT - goal_threshold),
                               20)
            screen.blit(goalFont.render('2', True, Color.White),
                        (SCREEN_WIDTH - goal_threshold - 8, SCREEN_HEIGHT - goal_threshold - 12))

            # Moving Objects
            for i in xrange(numAgents):
                agents[i].draw()

            # Obstacles
            circle1.draw()
            circle2.draw()
            circle3.draw()
            circle4.draw()

            # Boundary
            border.draw()

            # Show FPS
            PrintFPS('FPS : ' + str('{:3.2f}').format(fps))
        # ---------------------------------------------------------

        # Flip the screen
        if render:
            pygame.display.flip()

        if print_fps:
            print('FPS : ' + str('{:3.2f}').format(fps))

        # Time counter
        homing_global.timer += deltaTime

        # Step counter
        homing_global.timestep += 1

    pygame.quit()
    print('Done!')

    if homing_global.record:
        homing_global.record = False
        homing_global.fo.close()
        print("stop recording")

    # Close file
    #if recording:
    #    homing_global.fo.close()
            #= open("hello.txt", "r")
        #f.close()

