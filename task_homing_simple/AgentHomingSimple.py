from __future__ import division
import numpy as np
import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (vec2)
from enum import Enum
from keras.layers import Dense
from keras.layers import Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
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
    from ..res import print_colors as PrintColor
    import debug_homing_simple
    import global_homing_simple
    from .. import Global

# ----------- Agent's brain Neural Network Config ----------------

class DQNHomingSimple(DQN):
    def build_model(self):

        model = Sequential()

        h1 = 8  # 1st hidden layer's size
        h2 = 5  # 2nd hidden layer's size

        model.add(Dense(h1, input_dim=self.inputCnt))  # input -> hidden
        model.add(Activation('relu'))

        if h2 != 0:
            model.add(Dense(h2))  # hidden -> hidden
            model.add(Activation('relu'))

        model.add(Dense(self.actionCnt, activation='linear'))  # hidden -> output

        # Optimizer
        optimizer = Adam(lr=self.lr)

        # Compile model
        model.compile(loss='mse', optimizer=optimizer)

        return model


# -------------------- Agent ----------------------

# Agent's possible actions
class Action(Enum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    KEEP_ORIENTATION = 2

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


class AgentHomingSimple(Agent):
    def __init__(self, screen=None, world=None, x=0, y=0, angle=0, radius=1.5, id=-1, goals=vec2(0, 0)):
        super(AgentHomingSimple, self).__init__(screen, world, x, y, angle, radius)

        # Agent's ID
        self.id = id

        # Input size
        self.input_size = 1

        # Goal
        self.currentGoalIndex = 0
        self.goals = goals
        self.goalReachedThreshold = 2.4  # If agent-to-goal distance is less than this value then agent reached the goal
        self.num_goals = len(self.goals) # number of goals

        # Event features
        self.goalReachedCount = 0
        self.startTime = 0.0
        self.startTimestep = 0
        self.elapsedTime = 0.00
        self.elapsedTimestep = 0

        self.last_reward = 0.0  # last agent's reward
        self.last_distance = 0.0  # last agent's distance to the goal
        self.distance = self.distanceToGoal()  # 0.0  # current distance to the goal
        self.last_position = vec2(self.body.position.x, self.body.position.y)  # keep track of previous position
        self.last_orientation = self.orientationToGoal()

    def setup(self, training=True, random_agent=False):

        # Update the agent's flag
        self.training = training

        if random_agent: # can't train when random
            self.training = False
            self.random_agent = random_agent

        # Create agent's brain
        self.brain = DQNHomingSimple(inputCnt=self.input_size, actionCnt=len(list(Action)), id=self.id,
                                     ratio_update=1, training=self.training, random_agent=self.random_agent)

        # Set agent's position : Start from goal 2
        start_pos = self.goals[1]
        self.body.position = start_pos

        # Set agent's orientation : Start by looking at goal 1
        toGoal = Util.normalize(pixelsToWorld(self.goals[0]) - start_pos)
        forward = vec2(0, 1)
        angleDeg = Util.angle(forward, toGoal)
        angle = Util.degToRad(angleDeg)
        angle = -angle
        self.body.angle = angle

        # -------------------- Reset agent's variables --------------------

        # Goals
        self.currentGoalIndex = 0

        # Event features
        self.goalReachedCount = 0
        self.startTime = 0.0
        self.startTimestep = 0
        self.elapsedTime = 0.00
        self.elapsedTimestep = 0

        self.last_reward = 0.0
        self.last_distance = 0.0
        self.distance = self.distanceToGoal() # current distance to the goal
        self.last_position = vec2(self.body.position.x, self.body.position.y)
        self.last_orientation = self.orientationToGoal()

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

    def updateDrive(self, action):
        """
            Perform agent's movement based on the input action
        """
        speed = self.speed

        if action == Action.TURN_LEFT:  # Turn Left
            self.body.angularVelocity = self.rotation_speed
        elif action == Action.TURN_RIGHT:  # Turn Right
            self.body.angularVelocity = -self.rotation_speed
        elif action == Action.KEEP_ORIENTATION:  # Don't turn
            pass

        # Move Agent
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

        # Process getting_closer reward
        flagGC = False
        getting_closer = 0.
        if self.distance < self.last_distance:
            flagGC = True # getting closer

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
        super(AgentHomingSimple, self).update()

        # Orientation to the goal
        orientation = self.orientationToGoal()

        # Agent's signal
        last_signal = np.asarray([orientation])

        # Select action using AI
        action_num = self.brain.update(self.last_reward, last_signal)
        self.updateFriction()
        # self.remainStatic()
        self.updateDrive(Action(action_num))
        # self.updateManualDrive()

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