import numpy as np
import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (vec2)
from enum import Enum
from keras.layers import Dense
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
    from Agent import Agent
    from Setup import *
    from Util import worldToPixels, pixelsToWorld
    import Util
    from homing_task.RayCastCallback import RayCastCallback
    import res.print_colors as PrintColor
    import homing_debug
    import homing_global
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
    import homing_debug
    import homing_global


# ----------- Agent's brain Neural Network Config ----------------

class DQNHoming(DQN):
    def build_model(self):
        # Sequential() creates the foundation of the layers.
        model = Sequential()

        # 'Dense' define fully connected layers
        model.add(Dense(24, activation='relu', input_dim=self.inputCnt))  # input -> hidden
        #model.add(Dense(24, activation='relu'))  # hidden -> hidden
        model.add(Dense(self.actionCnt, activation='linear'))  # hidden -> output

        # # 'Dense' define fully connected layers
        # model.add(Dense(4, activation='relu', input_dim=self.inputCnt))  # input -> hidden
        # model.add(Dense(4, activation='relu'))  # hidden -> hidden
        # model.add(Dense(self.actionCnt, activation='linear'))  # hidden -> output

        # Compile model
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))  # optimizer for stochastic gradient descent

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
    LIVING_PENALTY = -0.5
    GETTING_CLOSER = 0.1
    GOAL_REACHED = 1.0

    @classmethod
    def sensorReward(cls, x):
        """
            Custom cubic regression made with https://mycurvefit.com/
            Check sensor-reward-function-HomingTask.xls file in /res/ directory
        """
        if x == 2.0:
            return cls.LIVING_PENALTY
        else:
            y = -1 + 0.5909357 * x - 0.2114035 * np.power(x, 2) + 0.02046784 * np.power(x, 3)
            return y


class AgentHoming(Agent):
    def __init__(self, screen=None, world=None, x=0, y=0, angle=0, radius=1.5, id=-1, numAgents=0, training=True):
        # numAgents : the total number of agents in the simulation -> create a vector of collision's time for each agent
        super(AgentHoming, self).__init__(screen, world, x, y, angle, radius)

        # Agent's ID
        self.id = id

        # Training flag
        self.training = training

        # Default speed of the agent
        self.initialSpeed = 9.5  # 12 m/s

        self.sensor1 = 0.0  # left raycast
        self.sensor2 = 0.0  # front raycast
        self.sensor3 = 0.0  # right raycast
        self.raycastLength = 2.0
        self.brain = DQNHoming(inputCnt=5, actionCnt=len(list(Action)), id=self.id, training=self.training)

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
        goal1 = pixelsToWorld((100, 100))
        goal2 = vec2(SCREEN_WIDTH / PPM - goal1.x, SCREEN_HEIGHT / PPM - goal1.y)
        self.currentGoalIndex = 0
        self.goals = [goal1, goal2]
        self.goalReachedThreshold = 2.5  # If agent-to-goal distance is less than this value then agent reached the goal

        # Event features
        self.goalReachedCount = 0
        self.startTime = 0.0
        self.startTimestep = 0
        self.elapsedTime = 0.00
        self.elapsedTimestep = 0

        self.last_reward = 0.0  # last agent's reward
        self.last_distance = 0.0  # last agent's distance to the goal
        self.distance = 0.0  # current distance to the goal

        # Raycast Left
        top_left = self.body.GetWorldVector(vec2(-0.70, 0.70))
        self.raycastLeft_point1 = self.body.worldCenter + top_left * self.radius
        self.raycastLeft_point2 = self.raycastLeft_point1 + top_left * self.raycastLength
        self.initial_raycastSideColor = Color.Yellow
        self.raycastSideColor = Color.Yellow

        # Raycast Front
        forward_vec = self.body.GetWorldVector(vec2(0, 1))
        self.raycastFront_point1 = self.body.worldCenter + forward_vec * self.radius
        self.raycastFront_point2 = self.raycastFront_point1 + forward_vec * self.raycastLength
        self.initial_raycastFrontColor = Color.Red
        self.raycastFrontColor = Color.Red

        # Raycast Right
        top_right = self.body.GetWorldVector(vec2(0.70, 0.70))  # sqr(2) / 2
        self.raycastRight_point1 = self.body.worldCenter + top_right * self.radius
        self.raycastRight_point2 = self.raycastRight_point1 + top_right * self.raycastLength

        # Keep track of the time it took for agent to reach goal
        self.timeToGoal_window = deque(maxlen=100)

        # Get initial orientation to the goal
        orientation = self.orientationToGoal()
        self.facingGoal = True  # default
        if (0.0 <= orientation < 0.5) or (-0.5 <= orientation < 0.0):
            self.facingGoal = True
            homing_debug.printEvent(self, "facing goal: {}".format(self.currentGoalIndex + 1))
        elif (0.5 <= orientation < 1.0) or (-1.0 <= orientation < -0.5):
            self.facingGoal = False
            homing_debug.printEvent(self, "reverse facing goal: {}".format(self.currentGoalIndex + 1))

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

        # Forward Line
        # current_forward_normal = self.body.GetWorldVector(vec2(0, 1))
        # pygame.draw.line(self.screen, Color.White, worldToPixels(self.body.worldCenter),
        #                  worldToPixels(self.body.worldCenter + current_forward_normal * self.radius))

        # Raycast Left
        top_left = self.body.GetWorldVector(vec2(-0.70, 0.70))
        self.raycastLeft_point1 = self.body.worldCenter + top_left * self.radius
        self.raycastLeft_point2 = self.raycastLeft_point1 + top_left * self.raycastLength
        pygame.draw.line(self.screen, self.raycastSideColor, worldToPixels(self.raycastLeft_point1),
                         worldToPixels(self.raycastLeft_point2))  # draw the raycast

        # Raycast Front
        forward_vec = self.body.GetWorldVector(vec2(0, 1))
        self.raycastFront_point1 = self.body.worldCenter + forward_vec * self.radius
        self.raycastFront_point2 = self.raycastFront_point1 + forward_vec * self.raycastLength
        pygame.draw.line(self.screen, self.raycastFrontColor, worldToPixels(self.raycastFront_point1),
                         worldToPixels(self.raycastFront_point2))  # draw the raycast

        # Raycast Right
        top_right = self.body.GetWorldVector(vec2(0.70, 0.70))  # sqr(2) / 2
        self.raycastRight_point1 = self.body.worldCenter + top_right * self.radius
        self.raycastRight_point2 = self.raycastRight_point1 + top_right * self.raycastLength
        pygame.draw.line(self.screen, self.raycastSideColor, worldToPixels(self.raycastRight_point1),
                         worldToPixels(self.raycastRight_point2))  # draw the raycast

    def readSensors(self):
        # TODO refactor raycast method, multiple copies of the same code

        # Raycast Left
        rayCastLeft = RayCastCallback()
        self.world.RayCast(rayCastLeft, self.raycastLeft_point1, self.raycastLeft_point2)
        if rayCastLeft.hit:
            dist1 = (self.raycastLeft_point1 - rayCastLeft.point).length  # distance to the hit point
            self.sensor1 = round(dist1, 2)
        else:
            self.sensor1 = self.raycastLength

        # Raycast Front
        rayCastFront = RayCastCallback()
        self.world.RayCast(rayCastFront, self.raycastFront_point1, self.raycastFront_point2)
        if rayCastFront.hit:
            dist2 = (self.raycastFront_point1 - rayCastFront.point).length  # distance to the hit point
            self.sensor2 = round(dist2, 2)
            # print('val sensor front', self.sensor2)
        else:
            self.sensor2 = self.raycastLength

        # Raycast Right
        rayCastRight = RayCastCallback()
        self.world.RayCast(rayCastRight, self.raycastRight_point1, self.raycastRight_point2)
        if rayCastRight.hit:
            dist3 = (self.raycastRight_point1 - rayCastRight.point).length  # distance to the hit point
            self.sensor3 = round(dist3, 2)
        else:
            self.sensor3 = self.raycastLength

    def normalizeSensorsValue(self, val):
        return Util.minMaxNormalizationScale(val, minX=0.0, maxX=self.raycastLength)

    def updateDrive(self, action):
        """
            Perform agent's movement based on the input action
        """
        speed = self.initialSpeed

        if action == Action.TURN_LEFT:  # Turn Left
            self.body.angularVelocity = 10.5  # 5
        elif action == Action.TURN_RIGHT:  # Turn Right
            self.body.angularVelocity = -10.5  # -5
        elif action == Action.KEEP_ORIENTATION:  # Don't turn
            pass
        elif action == Action.STOP:  # Stop moving
            speed = 0
        elif action == Action.STOP_TURN_LEFT:  # Stop and turn left
            speed = 0
            self.body.angularVelocity = 10.5
        elif action == Action.STOP_TURN_RIGHT:  # Stop and turn right
            speed = 0
            self.body.angularVelocity = -10.5

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
        return distance

    def computeGoalReached(self):
        self.goalReachedCount += 1
        self.elapsedTime = homing_global.timer - self.startTime
        self.elapsedTimestep = homing_global.timestep - self.startTimestep

        self.timeToGoal_window.append(self.elapsedTimestep)

        # Goal reached event
        sys.stdout.write(PrintColor.PRINT_RED)
        homing_debug.printEvent(self, "reached goal: {}".format(self.currentGoalIndex + 1))
        sys.stdout.write(PrintColor.PRINT_RESET)

        # Reset, Update
        self.startTime = homing_global.timer
        self.startTimestep = homing_global.timestep
        self.currentGoalIndex = (self.currentGoalIndex + 1) % len(self.goals)  # change goal
        self.t2GCollisionCount = 0
        self.t2GAgentCollisionCount = 0

    def update(self):
        """
            Main function of the agent
        """
        super(AgentHoming, self).update()

        # Read sensor's value
        self.readSensors()

        # Normalize sensor's value
        normSensor1 = self.normalizeSensorsValue(self.sensor1)
        normSensor2 = self.normalizeSensorsValue(self.sensor2)
        normSensor3 = self.normalizeSensorsValue(self.sensor3)

        # Get orientation to the goal
        orientation = self.orientationToGoal()
        if (0.0 <= orientation < 0.5) or (-0.5 <= orientation < 0.0):
            if not self.facingGoal:
                self.facingGoal = True
                homing_debug.printEvent(self, "facing goal: {}".format(self.currentGoalIndex + 1))

        elif (0.5 <= orientation < 1.0) or (-1.0 <= orientation < -0.5):
            if self.facingGoal:
                self.facingGoal = False
                homing_debug.printEvent(self, "reverse facing goal: {}".format(self.currentGoalIndex + 1))

        # Select action using AI
        # last_signal = np.asarray([self.sensor1, self.sensor2, self.sensor3, orientation])
        last_signal = np.asarray([normSensor1, normSensor2, normSensor3, orientation, -orientation])
        action_num = self.brain.update(self.last_reward, last_signal)
        self.updateFriction()
        self.updateDrive(Action(action_num))
        # self.updateManualDrive()
        # self.updateManualDriveTestAngle(10.5)  # 10

        # Calculate agent's distance to the goal
        self.distance = self.distanceToGoal()

        # Living Penalty
        self.last_reward = Reward.LIVING_PENALTY

        # Process agent's distance to goal
        if self.distance < self.last_distance:  # getting closer
            self.last_reward = Reward.GETTING_CLOSER

        # Process sensor's value
        if self.sensor1 < self.raycastLength or self.sensor2 < self.raycastLength or self.sensor3 < self.raycastLength:
            r1 = Reward.sensorReward(self.sensor1)
            r2 = Reward.sensorReward(self.sensor2)
            r3 = Reward.sensorReward(self.sensor3)
            self.last_reward = np.amin(np.array([r1, r2, r3]))

        # Reached Goal
        if self.distance < self.goalReachedThreshold:
            self.computeGoalReached()
            self.last_reward = Reward.GOAL_REACHED
            # self.brain.replay()  # experience replay

        self.last_distance = self.distance
        self.elapsedTime = homing_global.timer - self.startTime
        self.elapsedTimestep = homing_global.timestep - self.startTimestep

        return

    def learning_score(self):
        """
            Score is the mean of the reward in the sliding window
        """
        learning_score = self.brain.learning_score()
        return learning_score

    def collisionColor(self):
        """
            Change agent color during collision
        """
        self.currentCollisionCount += 1

        self.color = Color.DeepSkyBlue
        self.raycastSideColor = Color.Magenta
        self.raycastFrontColor = Color.Magenta

    def endCollisionColor(self):
        """
            Reset agent color when end of collision
        """
        self.currentCollisionCount -= 1
        if self.currentCollisionCount < 0:
            sys.exit('Error! Cannot have currentCollisionCount of negative value')
        elif self.currentCollisionCount == 0:
            self.color = self.initial_color
            self.raycastFrontColor = self.initial_raycastFrontColor
            self.raycastSideColor = self.initial_raycastSideColor

    def save_brain(self):
        """
            Save agent's brain (model : neural network, optimizer, loss, etc) in file
            Also create the /brain_files/ directory if it doesn't exist
        """
        timestr = time.strftime("%Y_%m_%d_%H%M%S")
        directory = "./brain_files/"
        network_model = directory + timestr + "_model.h5"  # neural network model file

        if not os.path.exists(os.path.dirname(directory)):
            try:
                os.makedirs(os.path.dirname(directory))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        self.brain.save_model(network_model)

    def load_model(self):
        """
            Load Agent's model config from file
            Everything : NN architecture, optimizer, weights, ...
        """
        directory = "./brain_files/"
        model_file = directory + "brain" + "_model.h5"  # neural network model file

        self.brain.load_model(model_file)

    def load_weights(self):
        """
            Load Agent's weights from file
        """
        directory = "./brain_files/"
        model_file = directory + "brain" + "_model.h5"  # neural network model file

        self.brain.load_weights(model_file)

    def load_lower_layers_weights(self):
        """
            Load Agent's lowe layers' weights from file
        """
        directory = "./brain_files/"
        model_file = directory + "brain" + "_model.h5"  # neural network model file

        self.brain.load_lower_layers_weights(model_file)

    def save_memory(self):
        """
            Save Agent's memory (experiences) in csv file
            Also create the /brain_files/ directory if it doesn't exist
        """
        timestr = time.strftime("%Y_%m_%d_%H%M%S")
        directory = "./brain_files/"
        memory_file = directory + timestr + "_memory.csv"  # neural network model file

        if not os.path.exists(os.path.dirname(directory)):
            try:
                os.makedirs(os.path.dirname(directory))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        self.brain.save_memory(memory_file)

    def load_memory(self):
        """
            Load memory from file
        """
        directory = "./brain_files/"
        memory_file = directory + "brain" + "_memory.csv"  # neural network model file

        memory_list = []
        data = pd.read_csv(memory_file)

        remove_bracket = lambda x: x.replace('[', '').replace(']', '')
        string_to_array = lambda x: np.expand_dims(np.fromstring(x, sep=' '), axis=0)

        data['state'] = data['state'].map(remove_bracket).map(string_to_array)
        data['next_state'] = data['next_state'].map(remove_bracket).map(string_to_array)

        for i, row in data.iterrows():
            exp = (row['state'], row['action'], row['reward'], row['next_state'])
            memory_list.append(exp)

        return

    def stop_training(self):
        """
            Stop training the brain (Neural Network)
            Stop exploration -> only exploitation
        """
        self.training = False
        self.brain.stop_training()