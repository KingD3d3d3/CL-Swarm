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


# ----------- Agent's brain Neural Network Config ----------------

class DQNHomingSimple(DQN):
    def build_model(self):
        # Sequential() creates the foundation of the layers.
        model = Sequential()

        # Neural Network's Architecture
        # Without collision avoidance
        h1 = 8  # 1st hidden layer's size
        h2 = 5  # 2nd hidden layer's size
        #h3 = 0

        model.add(Dense(h1, input_dim=self.inputCnt))  # input -> hidden  # activation='relu'
        model.add(Activation('relu'))

        if h2 != 0:
            model.add(Dense(h2))  # hidden -> hidden  # activation='relu'
            model.add(Activation('relu'))

        # if h3 != 0:
        #     model.add(Dense(h3))  # hidden -> hidden  # activation='relu'
        #     model.add(Activation('relu'))

        model.add(Dense(self.actionCnt, activation='linear'))  # hidden -> output

        # Using Sensors' input
        # model.add(Dense(24, activation='relu', input_dim=self.inputCnt))  # input -> hidden
        # model.add(Dense(24, activation='relu'))  # hidden -> hidden
        # model.add(Dense(self.actionCnt, activation='linear'))  # hidden -> output

        # Optimizer
        # optimizer = keras.optimizers.SGD(lr=self.lr, momentum=0.0, decay=0.0, nesterov=False)
        # optimizer = keras.optimizers.SGD(lr=self.lr, momentum=0.9, decay=0.0, nesterov=True)
        optimizer = Adam(lr=self.lr)

        # Compile model
        model.compile(loss='mse', optimizer=optimizer)  # optimizer for stochastic gradient descent

        return model


# -------------------- Agent ----------------------

# Agent's possible actions
class Action(Enum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    KEEP_ORIENTATION = 2
    # STOP = 3
    # STOP_TURN_LEFT = 4
    # STOP_TURN_RIGHT = 5


# Rewards Mechanism
class Reward:
    GETTING_CLOSER = 0.1

    LIVING_PENALTY = 0.  # -0.01 # 0 #-0.5
    GOAL_REACHED = 0.  # 1.0 # 10.0 # useless reward

    @classmethod
    def sensorReward(cls, x):
        """
            Sensor value (SV):
            x = 2 -> y = 0
            x = 1.9 -> y = 0
            x = 1 -> y = -0.1
            x = 2 -> y =-0.5
        """
        # y = -1 + 1.155556 * x - 0.7666667 * np.power(x, 2) + 0.1111111 * np.power(x, 3)
        # y = 0.5 * x - 1
        # return round(y, 1)

        y = -0.5 + 0.5909357 * x - 0.2114035 * np.power(x, 2) + 0.02046784 * np.power(x, 3)
        y = round(y, 1)
        if y == -0 or y == 0:
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


class AgentHomingSimple(Agent):
    def __init__(self, screen=None, world=None, x=0, y=0, angle=0, radius=1.5, id=-1, numAgents=0, training=True,
                 collision_avoidance=True, random_agent=False):
        # numAgents : the total number of agents in the simulation -> create a vector of collision's time for each agent
        super(AgentHomingSimple, self).__init__(screen, world, x, y, angle, radius, training, random_agent)

        # Agent's ID
        self.id = id

        # Training flag
        self.training = training

        # Default spec of the agent
        self.speed = 10 # 9.5 m/s
        self.rotation_speed = 10 # # 10.5 rad/s

        # Collision avoidance
        self.collision_avoidance = collision_avoidance

        # Proximity sensors
        if not collision_avoidance:
            self.numSensors = 0  # no need proximity sensors when no collision avoidance behavior
        else:
            self.numSensors = 0  # total numbers of proximity sensors by agent
        self.raycastLength = 2.0
        self.sensors = np.ones(self.numSensors) * self.raycastLength

        # Colors of raycast
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

        # Store raycast direction vector for each raycast
        self.raycast_vectors = (top_left, forward_vec, top_right, right, bottom_right, backward, bottom_left, left)

        # Brain of the Agent
        input_size = self.numSensors + 1  # 2
        self.brain = DQNHomingSimple(inputCnt=input_size, actionCnt=len(list(Action)), id=self.id,
                                     training=self.training, ratio_update=1, random_agent=random_agent)

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

        # Goal
        goal1 = pixelsToWorld((100, 100)) # b2Vec2(5,31)
        goal2 = vec2(SCREEN_WIDTH / PPM - goal1.x, SCREEN_HEIGHT / PPM - goal1.y) # b2Vec2(59,5)

        distance = Util.distance(goal1, goal2)

        self.currentGoalIndex = 0
        self.goals = [goal1, goal2]
        self.goalReachedThreshold = 2.5  # If agent-to-goal distance is less than this value then agent reached the goal
        self.num_goals = len(self.goals) # number of goals

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

        # Keep track of the time it took for agent to reach goal
        # self.timeToGoal_window = deque(maxlen=100)

        # Get initial orientation to the goal
        orientation = self.orientationToGoal()
        self.facingGoal = True  # default
        if (0.0 <= orientation < 0.5) or (-0.5 <= orientation < 0.0):
            self.facingGoal = True
        elif (0.5 <= orientation < 1.0) or (-1.0 <= orientation < -0.5):
            self.facingGoal = False

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
        if self.numSensors != 0:
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

        if not self.collision_avoidance:
            return

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
        return Util.minMaxNormalization_m1_1(val, 0.0, self.raycastLength)

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

        # elif action == Action.STOP:  # Stop moving
        #     move = False
        #     speed = 0.
        # elif action == Action.STOP_TURN_LEFT:  # Stop and turn left
        #     move = False
        #     speed = 0.
        #     self.body.angularVelocity = self.rotation_speed
        # elif action == Action.STOP_TURN_RIGHT:  # Stop and turn right
        #     move = False
        #     speed = 0.
        #     self.body.angularVelocity = -self.rotation_speed

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
        self.t2GCollisionCount = 0
        self.t2GAgentCollisionCount = 0
        self.distance = self.distanceToGoal()

    def rewardFunction(self):
        # Check Getting closer reward
        flagGC = False
        getting_closer = 0.
        if self.distance < self.last_distance:  # getting closer
            flagGC = True

            # Process getting_closer reward

            # Calculate alpha
            # goal_position = self.goals[self.currentGoalIndex]
            # v1 = (self.last_position - goal_position)  # last position to goal
            # v2 = (self.body.position - goal_position)  # current position to goal
            # max_angle = np.arctan(
            #     self.delta_dist / self.distance)  # maximum angle to goal an agent can make at each timestep
            # angle_deg = Util.angle(v1, v2)
            # angle_rad = Util.degToRad(angle_deg)
            # angle = Util.minMaxNormalization(angle_rad, -max_angle, max_angle, -np.pi / 2, np.pi / 2)

            # Calculate beta
            goal_position = self.goals[self.currentGoalIndex]
            v1 = (goal_position - self.last_position)  # goal to last position vector
            v2 = (self.body.position - self.last_position) # current position to last position vector
            angle_deg = Util.angle(v1, v2)
            angle = Util.degToRad(angle_deg)

            # Easy reward, using directly agent's orientation_to_goal
            # angle = self.orientationToGoal()
            # angle_rescaled = Util.minMaxNormalization(angle, -1, 1, -180, 180)
            # angle = Util.degToRad(angle_rescaled)

            getting_closer = Reward.getting_closer(angle)

            #print('getting_closer', getting_closer)
        # Check Goal Reached
        flagGR = False
        if self.distance < self.goalReachedThreshold:
            flagGR = True

        # Process sensor's value
        sensorReward = 0.
        if not self.sensors.shape[0] == 0:  # Check array length
            # Apply sensor reward function to all sensors value
            r = np.fromiter((Reward.sensorReward(s) for s in self.sensors), self.sensors.dtype, count=len(self.sensors))
            sensorReward = np.amin(r)  # take the min value

        # Overall reward
        # old # reward = flagGC * Reward.GETTING_CLOSER + sensorReward + flagGR * Reward.GOAL_REACHED + Reward.LIVING_PENALTY
        reward = flagGC * getting_closer + sensorReward + flagGR * Reward.GOAL_REACHED + Reward.LIVING_PENALTY

        return reward

    def update(self):
        """
            Main function of the agent
        """
        super(AgentHomingSimple, self).update()

        # Get orientation to the goal
        orientation = self.orientationToGoal()
        if (0.0 <= orientation < 0.5) or (-0.5 <= orientation < 0.0):
            if not self.facingGoal:
                self.facingGoal = True

        elif (0.5 <= orientation < 1.0) or (-1.0 <= orientation < -0.5):
            if self.facingGoal:
                self.facingGoal = False

        # Agent's signal
        if self.collision_avoidance:

            # Read sensor's value
            self.readSensors()

            # Normalize sensor's value
            normSensor1 = self.normalizeSensorsValue(self.sensors[0])
            normSensor2 = self.normalizeSensorsValue(self.sensors[1])
            normSensor3 = self.normalizeSensorsValue(self.sensors[2])
            normSensor4 = self.normalizeSensorsValue(self.sensors[3])
            normSensor5 = self.normalizeSensorsValue(self.sensors[4])
            normSensor6 = self.normalizeSensorsValue(self.sensors[5])
            normSensor7 = self.normalizeSensorsValue(self.sensors[6])
            normSensor8 = self.normalizeSensorsValue(self.sensors[7])

            last_signal = np.asarray([normSensor1, normSensor2, normSensor3,
                                      normSensor4, normSensor5,
                                      normSensor6, normSensor7, normSensor8,
                                      orientation, -orientation])
        else:
            # last_signal = np.asarray([orientation, -orientation]) # states when no collision avoidance
            last_signal = np.asarray([orientation])

            # Testing distance as input
            # ('distance goal to goal', '59.93329625508679')
            # dist = Util.minMaxNormalization_m1_1(self.distance, 0.0, 60.0)
            # last_signal = np.asarray([orientation, dist])

        # Select action using AI
        action_num = self.brain.update(self.last_reward, last_signal)
        self.updateFriction()
        # self.remainStatic()
        self.updateDrive(Action(action_num))
        # self.updateManualDrive()
        # self.updateManualDriveTestAngle(10.5)  # 10

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

        return

    def collisionColor(self):
        """
            Change agent color during collision
        """
        self.currentCollisionCount += 1

        if self.collision_avoidance:
            self.color = Color.DeepSkyBlue
            self.raycastDiagonalColor = Color.Magenta
            self.raycastStraightColor = Color.Magenta

    def endCollisionColor(self):
        """
            Reset agent color when end of collision
        """
        self.currentCollisionCount -= 1
        if self.currentCollisionCount < 0:
            sys.exit('Error! Cannot have currentCollisionCount of negative value')
        elif self.currentCollisionCount == 0:
            if self.collision_avoidance:
                self.color = self.initial_color
                self.raycastStraightColor = self.initial_raycastStraightColor
                self.raycastDiagonalColor = self.initial_raycastDiagonalColor
