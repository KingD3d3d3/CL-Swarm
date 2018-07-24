from __future__ import division
import numpy as np
import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (vec2)
from enum import Enum
from pygame.locals import *
import sys
from collections import deque
import time
import os
import errno
import csv
import pandas as pd
import keras.optimizers

try:
    # Running in PyCharm
    import res.colors as Color
    # from ..res import colors as Color
    from AI.DQN import DQN
    from Agent import Agent
    from Setup import *
    from Util import worldToPixels, pixelsToWorld
    import Util
    from task_homing_simple.RayCastCallback import RayCastCallback
    import res.print_colors as PrintColor
    import debug_homing_simple
    import global_homing_simple
    import Global
except:
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running from command line -> Import libraries as package')
    from ..res import colors as Color
    from ..AI.DQN import DQN
    from ..Agent import Agent
    from ..Setup import *
    from ..Util import worldToPixels, pixelsToWorld
    from .. import Util
    from .RayCastCallback import RayCastCallback
    from ..res import print_colors as PrintColor
    import debug_homing_simple
    import global_homing_simple
    from .. import Global


# Rewards Mechanism
class Reward:
    GETTING_CLOSER = 0.1

    LIVING_PENALTY = 0.  # -0.01 # 0 #-0.5
    GOAL_REACHED = 0.  # 1.0 # 10.0 # useless reward

    @classmethod
    def getting_closer(cls, angle):
        """
            Getting Closer (GC)
            angle_travelled = 0 deg -> reward = GETTING_CLOSER = 0.1 (Maximum)
            angle_travelled = 45 deg -> reward = 0.1 * sqrt(2) / 2 = 0.07
            angle_travelled = 90 deg -> reward = 0
        """
        reward = Reward.GETTING_CLOSER * np.cos(angle)
        return reward

class AgentHomingPerfect(Agent):
    def __init__(self, screen=None, world=None, x=0, y=0, angle=0, radius=1.5, id=-1, numAgents=0):
        # numAgents : the total number of agents in the simulation -> create a vector of collision's time for each agent
        super(AgentHomingPerfect, self).__init__(screen, world, x, y, angle, radius)

        # Agent's ID
        self.id = id

        # Default spec of the agent
        self.speed = 10 # 9.5 m/s
        self.rotation_speed = 10 # # 10.5 rad/s

        # Goal
        goal1 = pixelsToWorld((100, 100)) # b2Vec2(5,31)
        goal2 = vec2(SCREEN_WIDTH / PPM - goal1.x, SCREEN_HEIGHT / PPM - goal1.y) # b2Vec2(59,5)

        distance = Util.distance(goal1, goal2)

        self.currentGoalIndex = 0
        self.goals = [goal1, goal2]
        self.goalReachedThreshold = 2.5  # If agent-to-goal distance is less than this value then agent reached the goal
        self.num_goals = len(self.goals) # number of goals

        # Collision with obstacles (walls, obstacles in path)
        self.t2GCollisionCount = 0
        self.collisionCount = 0
        self.elapsedTimestepObstacleCollision = 0.00  # timestep passed between collision (for same objects collision)
        self.startTimestepObstacleCollision = 0.00  # start timestep since a collision (for same objects collision)
        self.lastObstacleCollide = None

        # Collision with Agents
        self.elapsedTimestepAgentCollision = np.zeros(
            numAgents)  # timestep passed between collision (for same objects collision)
        self.startTimestepAgentCollision = np.zeros(
            numAgents)  # start timestep since a collision (for same objects collision)
        self.t2GAgentCollisionCount = 0
        self.agentCollisionCount = 0

        self.currentCollisionCount = 0  # number of objects the agent is colliding at the current timestep

        # Event features
        self.goalReachedCount = 0
        self.startTime = 0.0
        self.startTimestep = 0
        self.elapsedTime = 0.00
        self.elapsedTimestep = 0

        self.maxReward = Reward.GETTING_CLOSER
        self.last_reward = 0.0  # last agent's reward
        self.last_distance = 0.0  # last agent's distance to the goal
        self.distance = self.distanceToGoal()  # 0.0  # current distance to the goal
        self.last_position = vec2(self.body.position.x, self.body.position.y)  # keep track of previous position
        self.last_orientation = self.orientationToGoal()

        # Get initial orientation to the goal
        orientation = self.orientationToGoal()
        self.facingGoal = True  # default
        if (0.0 <= orientation < 0.5) or (-0.5 <= orientation < 0.0):
            self.facingGoal = True
        elif (0.5 <= orientation < 1.0) or (-1.0 <= orientation < -0.5):
            self.facingGoal = False

    def learning_score(self):
        """
            Score is the mean of the reward in the sliding window
        """
        return 1

    def draw(self):

        # Circle of collision
        position = self.body.transform * self.fixture.shape.pos * PPM
        position = (position[0], SCREEN_HEIGHT - position[1])
        pygame.draw.circle(self.screen, Color.Blue, [int(x) for x in position], int(self.radius * PPM))

        # Triangle Shape
        vertex = [(-1, -1), (1, -1), (0, 1.5)]
        vertices = [(self.body.transform * v) * PPM for v in vertex]
        vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
        pygame.draw.polygon(self.screen, self.color, vertices)

        # Id of the agent
        font = pygame.font.SysFont("monospace", 22, bold=True)
        idText = font.render(str(self.id), True, Color.White)
        offset = [idText.get_rect().width / 2.0, idText.get_rect().height / 2.0]  # to adjust center
        idPos = (self.body.transform * (0, 0)) * PPM
        idPos = (idPos[0] - offset[0], SCREEN_HEIGHT - idPos[1] - offset[1])
        self.screen.blit(idText, idPos)


    def updateDrive(self):
        """
            Perform agent's movement based on the input action
        """
        speed = self.speed
        orientation = self.orientationToGoal()

        # Rotate to goal
        if orientation == 0.0:
            # Keep orientation
            pass
        elif 0.0 < orientation <= 1:
            # Goal is on the right -> turn right
            self.body.angularVelocity = -self.rotation_speed
        elif -1.0 <= orientation < 0.0:
            # Goal is on the left -> turn left
            self.body.angularVelocity = self.rotation_speed

        # Move forward
        forward_vec = self.body.GetWorldVector((0, 1))
        self.body.linearVelocity = forward_vec * speed


    def orientationToGoal(self):
        """
            Get agent orientation to the goal
            Angle in degrees, normalized between [-1,1] and counter-clockwise
            0 degree -> 0, perfectly aligned in direction to the goal
            90 deg -> 0.5, goal is on the right
            180 deg -> 1 , reverse facing
            -90 deg -> -0.5, goal is on the left
        """
        toGoal = Util.normalize(self.goals[self.currentGoalIndex] - self.body.position)
        forward = Util.normalize(self.body.GetWorldVector((0, 1)))
        orientation = Util.angle(forward, toGoal) / 180.0
        orientation = round(orientation, 2)  # only 3 decimals
        return orientation

    def distanceToGoal(self):
        distance = np.sqrt((self.body.position.x - self.goals[self.currentGoalIndex].x) ** 2 +
                           (self.body.position.y - self.goals[self.currentGoalIndex].y) ** 2)
        return round(distance, 2)  # 5) # 2

    def computeGoalReached(self):
        self.goalReachedCount += 1
        self.elapsedTime = global_homing_simple.timer - self.startTime
        self.elapsedTimestep = Global.timestep - self.startTimestep

        # self.timeToGoal_window.append(self.elapsedTimestep)

        # if self.elapsedTimestep <= 500 and Global.timestep >= 100000 \
        #         and round(self.learning_score(), 2) >= 0.10:
        #     self.ready_to_save = True

        # Goal reached event
        sys.stdout.write(PrintColor.PRINT_RED)
        debug_homing_simple.printEvent(self, "reached goal: {}".format(self.currentGoalIndex + 1))
        sys.stdout.write(PrintColor.PRINT_RESET)

        # Reset, Update
        self.startTime = global_homing_simple.timer
        self.startTimestep = Global.timestep
        self.currentGoalIndex = (self.currentGoalIndex + 1) % self.num_goals  # change goal
        self.distance = self.distanceToGoal()

    def rewardFunction(self):
        # Check Getting closer reward
        flagGC = False
        getting_closer = 0.
        if self.distance < self.last_distance:  # getting closer
            flagGC = True

            # Process getting_closer reward

            # Calculate beta
            goal_position = self.goals[self.currentGoalIndex]
            v1 = (goal_position - self.last_position)  # goal to last position vector
            v2 = (self.body.position - self.last_position) # current position to last position vector
            angle_deg = Util.angle(v1, v2)
            angle = Util.degToRad(angle_deg)
            getting_closer = Reward.getting_closer(angle)

        # Check Goal Reached
        flagGR = False
        if self.distance < self.goalReachedThreshold:
            flagGR = True

        # Overall reward
        reward = flagGC * getting_closer + flagGR * Reward.GOAL_REACHED + Reward.LIVING_PENALTY

        return reward

    def update(self):
        """
            Main function of the agent
        """
        super(AgentHomingPerfect, self).update()

        # Get orientation to the goal
        orientation = self.orientationToGoal()
        if (0.0 <= orientation < 0.5) or (-0.5 <= orientation < 0.0):
            if not self.facingGoal:
                self.facingGoal = True

        elif (0.5 <= orientation < 1.0) or (-1.0 <= orientation < -0.5):
            if self.facingGoal:
                self.facingGoal = False

        # Select action using CHEAT
        self.updateFriction()
        self.updateDrive()

        # Calculate agent's distance to the goal
        self.distance = self.distanceToGoal()

        # Agent's Reward
        self.last_reward = self.rewardFunction()

        # Reached Goal
        if self.distance < self.goalReachedThreshold:
            self.computeGoalReached()

        self.last_distance = self.distance
        self.elapsedTime = global_homing_simple.timer - self.startTime
        self.elapsedTimestep = Global.timestep - self.startTimestep
        self.last_position = vec2(self.body.position.x, self.body.position.y)
        self.last_orientation = self.body.angle

