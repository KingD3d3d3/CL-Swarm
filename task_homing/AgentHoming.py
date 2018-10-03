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
from pygame.locals import *
import sys
from collections import deque
import time
import os
import errno
import csv
import pandas as pd

try:
    # Running in PyCharm
    import res.colors as Color
    # from ..res import colors as Color
    from AI.DQN import DQN
    from objects.Agent import Agent
    from Setup import *
    from Util import worldToPixels, pixelsToWorld
    import Util
    from task_homing.RayCastCallback import RayCastCallback
    import res.print_colors as PrintColor
    import debug_homing
    import global_homing
    import Global
except:
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running from command line -> Import libraries as package')
    from ..res import colors as Color
    from ..AI.DQN import DQN
    from ..objects.Agent import Agent
    from ..Setup import *
    from ..Util import worldToPixels, pixelsToWorld
    from .. import Util
    from .RayCastCallback import RayCastCallback
    from ..res import print_colors as PrintColor
    import debug_homing
    import global_homing
    from .. import Global


# ----------- Agent's brain Neural Network Config ----------------

class DQNHoming(DQN):
    def build_model(self, h1=-1, h2=-1):
        model = Sequential()

        if h1 == -1 and h2 == -1:
            # Default values -- need to find the good hyperparam
            h1 = 10  # 1st hidden layer's size
            h2 = 10  # 2nd hidden layer's size

        model.add(Dense(h1, input_dim=self.inputCnt))  # input -> hidden
        model.add(Activation('relu'))

        if h2 != 0:
            model.add(Dense(h2))  # hidden -> hidden
            model.add(Activation('relu'))

        model.add(Dense(self.actionCnt, activation='linear'))  # hidden -> output

        # Optimizer
        optimizer = Adam(lr=self.lr)
        # optimizer = keras.optimizers.SGD(lr=self.lr, momentum=0.9, decay=0.0, nesterov=True)

        # Compile model
        model.compile(loss='mse', optimizer=optimizer)

        return model


# -------------------- Agent ----------------------

# Agent's possible actions
class Action(Enum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    KEEP_ORIENTATION = 2
    STOP = 3
    STOP_TURN_LEFT = 4
    STOP_TURN_RIGHT = 5

# Rewards Mechanism
class Reward:
    GETTING_CLOSER = 0.1
    LIVING_PENALTY = 0.
    GOAL_REACHED = 0.

    @classmethod
    def sensorReward(cls, x):
        """
            Custom cubic regression made with https://mycurvefit.com/
            Check sensor_reward_task_homing.xlsx file
        """
        # Simple line
        # y = 0.5 * x - 1
        # return round(y, 1)

        # Old with minimum -0.5 at x=0
        # y = -0.5 + 0.5909357 * x - 0.2114035 * np.power(x, 2) + 0.02046784 * np.power(x, 3)

        y = -1 + (1.566667 * x) - (0.8 * np.power(x, 2)) + (0.1333333 * np.power(x, 3))
        y = round(y, 1)
        if y == -0 or y == 0 or y >= 0:
            y = 0.
        return y

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


class AgentHoming(Agent):
    def __init__(self, screen=None, world=None, x=0, y=0, angle=0, radius=1.5, id=-1, goals=vec2(0, 0), numAgents=0):
        super(AgentHoming, self).__init__(screen, world, x, y, angle, radius)

        # Agent's ID
        self.id = id

        # Proximity Sensors
        self.raycastLength = 2.0
        self.initial_raycastDiagonalColor = Color.Yellow
        self.raycastDiagonalColor = Color.Yellow
        self.initial_raycastStraightColor = Color.Red
        self.raycastStraightColor = Color.Red
        top_left = vec2(-0.70, 0.70)  # Raycast Front Left
        forward_vec = vec2(0, 1)  # Raycast Front Middle
        top_right = vec2(0.70, 0.70)  # Raycast Front Right # sqr(2) / 2
        left = vec2(-1, 0)  # Raycast Left Side
        right = vec2(1, 0)  # Raycast Right Side
        bottom_left = vec2(-0.70, -0.70)  # Raycast Left Back
        backward = vec2(0, -1)  # Raycast Middle Back
        bottom_right = vec2(0.70, -0.70)  # Raycast Right Back
        self.raycast_vectors = (top_left, forward_vec, top_right, right, bottom_right, backward, bottom_left,
                                left)  # Store raycast direction vector for each raycast
        self.numSensors = len(self.raycast_vectors)
        self.sensors = np.ones(self.numSensors) * self.raycastLength # store the value of each sensors
        # print("num sensors : {}".format(self.numSensors))

        # Input size
        self.input_size = self.numSensors + 1  # 8 sensors + 1 orientation

        # Number of agents
        self.numAgents = numAgents

        # Goal
        self.currentGoalIndex = 0
        self.goals = goals
        self.goalReachedThreshold = 2.4  # If agent-to-goal distance is less than this value then agent reached the goal
        self.num_goals = len(self.goals)  # number of goals

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

        self.last_reward = 0.0  # last agent's reward
        self.last_distance = 0.0  # last agent's distance to the goal
        self.distance = self.distanceToGoal()  # 0.0  # current distance to the goal
        self.last_position = vec2(self.body.position.x, self.body.position.y)  # keep track of previous position
        self.last_orientation = self.orientationToGoal()

    def setup(self, training=True, random_agent=False, h1=-1, h2=-1):

        # Update the agent's flag
        self.training = training

        if random_agent:  # can't train when random
            self.training = False
            self.random_agent = random_agent

        # Create agent's brain
        self.brain = DQNHoming(inputCnt=self.input_size, actionCnt=len(list(Action)), id=self.id,
                               ratio_update=1, training=self.training, random_agent=self.random_agent, h1=h1, h2=h2)

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

        # Collision with obstacles (walls, obstacles in path, agents)
        self.t2GCollisionCount = 0
        self.collisionCount = 0
        self.elapsedTimestepObstacleCollision = 0.00  # timestep passed between collision (for same objects collision)
        self.startTimestepObstacleCollision = 0.00  # start timestep since a collision (for same objects collision)
        self.lastObstacleCollide = None

        # Collision with Agents
        self.elapsedTimestepAgentCollision = np.zeros(
            self.numAgents)  # timestep passed between collision (for same objects collision)
        self.startTimestepAgentCollision = np.zeros(
            self.numAgents)  # start timestep since a collision (for same objects collision)
        self.t2GAgentCollisionCount = 0
        self.agentCollisionCount = 0

        self.currentCollisionCount = 0  # number of objects the agent is colliding at the current timestep

        # Event features
        self.goalReachedCount = 0
        self.startTime = 0.0
        self.startTimestep = 0
        self.elapsedTime = 0.00
        self.elapsedTimestep = 0

        self.last_reward = 0.0
        self.last_distance = 0.0
        self.distance = self.distanceToGoal()  # current distance to the goal
        self.last_position = vec2(self.body.position.x, self.body.position.y)
        self.last_orientation = self.orientationToGoal()

        debug_homing.printEvent(color=PrintColor.PRINT_CYAN, agent=self,
                                event_message="agent is ready")

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

        # Draw raycasts
        for i in xrange(self.numSensors):
            v = self.body.GetWorldVector(self.raycast_vectors[i])
            p1 = self.body.worldCenter + v * self.radius
            p2 = p1 + v * self.raycastLength
            if i % 2 == 0:
                ray_color = self.raycastDiagonalColor
            else:
                ray_color = self.raycastStraightColor
            pygame.draw.line(self.screen, ray_color, worldToPixels(p1), worldToPixels(p2))

    def readSensors(self):

        # Read raycasts value
        for i in xrange(self.numSensors):
            raycast = RayCastCallback()
            v = self.body.GetWorldVector(self.raycast_vectors[i])
            p1 = self.body.worldCenter + v * self.radius
            p2 = p1 + v * self.raycastLength
            self.world.RayCast(raycast, p1, p2)
            if raycast.hit:
                dist = (p1 - raycast.point).length  # distance to the hit point
                self.sensors[i] = round(dist, 2)
            else:
                self.sensors[i] = self.raycastLength  # default value is raycastLength

    def normalizeSensorsValue(self, val):
        return Util.minMaxNormalization_m1_1(val, _min=0.0, _max=self.raycastLength)

    def updateDrive(self, action):
        """
            Perform agent's movement based on the input action
        """
        speed = self.speed
        move = True

        if action == Action.TURN_LEFT:  # Turn Left
            self.body.angularVelocity = self.rotation_speed
        elif action == Action.TURN_RIGHT:  # Turn Right
            self.body.angularVelocity = -self.rotation_speed
        elif action == Action.KEEP_ORIENTATION:  # Don't turn
            pass
        elif action == Action.STOP:  # Stop moving
            move = False
            speed = 0.
        elif action == Action.STOP_TURN_LEFT:  # Stop and turn left
            move = False
            speed = 0.
            self.body.angularVelocity = self.rotation_speed
        elif action == Action.STOP_TURN_RIGHT:  # Stop and turn right
            move = False
            speed = 0.
            self.body.angularVelocity = -self.rotation_speed

        if move:
            forward_vec = self.body.GetWorldVector((0, 1))
            self.body.linearVelocity = forward_vec * speed
        else:
            # Kill velocity
            impulse = -self.getForwardVelocity() * self.body.mass * (2. / 3.)
            self.body.ApplyLinearImpulse(impulse, self.body.worldCenter, True)  # kill forward

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
        return round(distance, 2)

    def computeGoalReached(self):
        self.goalReachedCount += 1
        self.elapsedTime = global_homing.timer - self.startTime
        self.elapsedTimestep = Global.timestep - self.startTimestep

        # Goal reached event
        debug_homing.printEvent(color=PrintColor.PRINT_RED, agent=self,
                                event_message="reached goal: {}".format(self.currentGoalIndex + 1))

        # Reset, Update
        self.startTime = global_homing.timer
        self.startTimestep = Global.timestep
        self.currentGoalIndex = (self.currentGoalIndex + 1) % self.num_goals  # change goal
        self.distance = self.distanceToGoal()
        self.t2GCollisionCount = 0
        self.t2GAgentCollisionCount = 0

    def rewardFunction(self):

        # Process getting_closer reward
        flagGC = False
        getting_closer = 0.
        if self.distance < self.last_distance:
            flagGC = True  # getting closer

            # Calculate beta
            goal_position = self.goals[self.currentGoalIndex]
            v1 = (goal_position - self.last_position)  # goal to last position vector
            v2 = (self.body.position - self.last_position)  # current position to last position vector
            angle_deg = Util.angle(v1, v2)
            angle = Util.degToRad(angle_deg)

            getting_closer = Reward.getting_closer(angle)

        # Check Goal Reached
        flagGR = False
        if self.distance < self.goalReachedThreshold:
            flagGR = True

        # Process sensor's value
        r = np.fromiter((Reward.sensorReward(s) for s in self.sensors), self.sensors.dtype,
                        count=len(self.sensors))  # Apply sensor reward function to all sensors value
        sensorReward = np.amin(r)  # take the min value

        # Overall reward
        reward = flagGC * getting_closer + sensorReward  # + flagGR * Reward.GOAL_REACHED + Reward.LIVING_PENALTY

        return reward

    def update(self):
        """
            Main function of the agent
        """
        super(AgentHoming, self).update()

        last_signal = []

        # Orientation to the goal
        orientation = self.orientationToGoal()
        last_signal.append(orientation)

        # Read sensor's value
        self.readSensors()

        # Normalize sensor's value
        for i in xrange(self.numSensors):
            normed_sensor = self.normalizeSensorsValue(self.sensors[i])
            last_signal.append(normed_sensor)

        # Array form
        last_signal = np.asarray(last_signal)

        # Select action using AI
        action_num = self.brain.update(self.last_reward, last_signal)
        self.updateFriction()
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
        self.elapsedTime = global_homing.timer - self.startTime
        self.elapsedTimestep = Global.timestep - self.startTimestep
        self.last_position = vec2(self.body.position.x, self.body.position.y)
        self.last_orientation = self.body.angle

    def collisionColor(self):
        """
            Change agent color during collision
        """
        self.currentCollisionCount += 1

        self.color = Color.DeepSkyBlue
        self.raycastDiagonalColor = Color.Magenta
        self.raycastStraightColor = Color.Magenta

    def endCollisionColor(self):
        """
            Reset agent color when end of collision
        """
        self.currentCollisionCount -= 1


        # if self.currentCollisionCount < 0:
        #     # sys.exit('Error! Cannot have currentCollisionCount of negative value')
        #     self.currentCollisionCount = 0
        #     debug_homing.xprint(msg='Error! Cannot have currentCollisionCount of negative value')
        #     self.color = self.initial_color
        #     self.raycastStraightColor = self.initial_raycastStraightColor
        #     self.raycastDiagonalColor = self.initial_raycastDiagonalColor

        if self.currentCollisionCount <= 0:
            self.currentCollisionCount = 0
            self.color = self.initial_color
            self.raycastStraightColor = self.initial_raycastStraightColor
            self.raycastDiagonalColor = self.initial_raycastDiagonalColor
