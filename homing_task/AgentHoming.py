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

try:
    # Running in PyCharm
    import res.colors as Color
    # from ..res import colors as Color
    from AI.DQN import Dqn
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
    from ..AI.DQN import Dqn
    from ..Agent import Agent
    from ..Setup import *
    from ..Util import worldToPixels, pixelsToWorld
    from .. import Util
    from .RayCastCallback import RayCastCallback
    from ..res import print_colors as PrintColor
    import homing_debug
    import homing_global


# ----------- Neural Network Config ----------------

class HomingDqn(Dqn):
    def build_model(self):
        # Sequential() creates the foundation of the layers.
        model = Sequential()

        # # # 'Dense' define fully connected layers
        # model.add(Dense(24, activation='relu', input_dim=self.inputCnt))  # input -> hidden
        # #model.add(Dense(24, activation='relu'))  # hidden -> hidden
        # model.add(Dense(self.actionCnt, activation='linear'))  # hidden -> output

        # 'Dense' define fully connected layers
        model.add(Dense(4, activation='relu', input_dim=self.inputCnt))  # input -> hidden
        model.add(Dense(4, activation='relu'))  # hidden -> hidden
        model.add(Dense(self.actionCnt, activation='linear'))  # hidden -> output

        # Compile model
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))  # optimizer for stochastic gradient descent

        return model


# -------------------- Agent ----------------------

# Agent's possible actions
class Action(Enum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    NOTHING = 2


class AgentHoming(Agent):
    def __init__(self, screen=None, world=None, x=0, y=0, angle=0, radius=2, goal_threshold=100, id=-1, numAgents=0):
        super(AgentHoming, self).__init__(screen, world, x, y, angle, radius)

        if id != -1:  # id == -1 if not set
            self.id = id  # id is set

        self.sensor1 = 0  # left raycast
        self.sensor2 = 0  # middle raycast
        self.sensor3 = 0  # right raycast
        self.brain = HomingDqn(inputCnt=5, actionCnt=len(list(Action)))
        #self.brain = HomingDqn(inputCnt=4, actionCnt=len(list(Action)))

        # Collision with obstacles (walls, obstacles in path)
        self.t2GCollisionCount = 0
        self.collisionCount = 0
        self.elapsedTimestepObstacleCollision = 0.00  # timestep passed between collision (for same objects collision)
        self.startTimestepObstacleCollision = 0.00  # start timestep since a collision (for same objects collision)
        self.lastObstacleCollide = None

        # Collision Agent
        self.lastAgentCollide = None
        self.elapsedTimestepAgentCollision = np.zeros(numAgents) # timestep passed between collision (for same objects collision)
        self.startTimestepAgentCollision = np.zeros(numAgents)   # start timestep since a collision (for same objects collision)
        self.t2GAgentCollisionCount = 0
        self.agentCollisionCount = 0

        # Goal
        goal1 = pixelsToWorld((goal_threshold, goal_threshold))
        goal2 = vec2(SCREEN_WIDTH / PPM - goal1.x, SCREEN_HEIGHT / PPM - goal1.y)
        self.currentGoalIndex = 0
        self.goals = [goal1, goal2]

        self.goalReachedCount = 0
        self.startTime = 0.0
        self.startTimestep = 0
        self.elapsedTime = 0.00
        self.elapsedTimestep = 0

        self.last_reward = 0
        self.last_distance = 0
        self.distance = 0  # current distance to the goal
        self.goalReachedThreshold = 2.5

        self.raycastLength = 2.0

        top_left = self.body.GetWorldVector(vec2(-0.70, 0.70))
        self.raycastLeft_point1 = self.body.worldCenter + top_left * self.radius
        self.raycastLeft_point2 = self.raycastLeft_point1 + top_left * self.raycastLength
        self.initial_raycastSideColor = Color.Yellow
        self.raycastSideColor = Color.Yellow

        forward_vec = self.body.GetWorldVector(vec2(0, 1))
        self.raycastFront_point1 = self.body.worldCenter + forward_vec * self.radius
        self.raycastFront_point2 = self.raycastFront_point1 + forward_vec * self.raycastLength
        self.initial_raycastFrontColor = Color.Red
        self.raycastFrontColor = Color.Red

        top_right = self.body.GetWorldVector(vec2(0.70, 0.70))  # sqr(2) / 2
        self.raycastRight_point1 = self.body.worldCenter + top_right * self.radius
        self.raycastRight_point2 = self.raycastRight_point1 + top_right * self.raycastLength

        self.timeToGoal_window = deque(maxlen=100)

        # Get initial orientation to the goal
        orientation = self.orientationToGoal()
        if (0.0 <= orientation < 0.5) or (-0.5 <= orientation < 0.0):
            self.facingGoal = True
            homing_debug.xprint(self, "facing goal: {}".format(self.currentGoalIndex + 1))
        elif (0.5 <= orientation < 1.0) or (-1.0 <= orientation < -0.5):
            self.facingGoal = False
            homing_debug.xprint(self, "reverse facing goal: {}".format(self.currentGoalIndex + 1))


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

        # Raycasts
        top_left = self.body.GetWorldVector(vec2(-0.70, 0.70))
        self.raycastLeft_point1 = self.body.worldCenter + top_left * self.radius
        self.raycastLeft_point2 = self.raycastLeft_point1 + top_left * 2
        pygame.draw.line(self.screen, self.raycastSideColor, worldToPixels(self.raycastLeft_point1),
                         worldToPixels(self.raycastLeft_point2))  # draw the raycast

        forward_vec = self.body.GetWorldVector(vec2(0, 1))
        self.raycastFront_point1 = self.body.worldCenter + forward_vec * self.radius
        self.raycastFront_point2 = self.raycastFront_point1 + forward_vec * 2
        pygame.draw.line(self.screen, self.raycastFrontColor, worldToPixels(self.raycastFront_point1),
                         worldToPixels(self.raycastFront_point2))  # draw the raycast

        top_right = self.body.GetWorldVector(vec2(0.70, 0.70))  # sqr(2) / 2
        self.raycastRight_point1 = self.body.worldCenter + top_right * self.radius
        self.raycastRight_point2 = self.raycastRight_point1 + top_right * 2
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
            #print('val sensor front', self.sensor2)
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

    def updateDrive(self, action):
        speed = 12

        if action == Action.TURN_LEFT:  # Turn Left
            self.body.angularVelocity = 10  # 5
            pass
        if action == Action.TURN_RIGHT:  # Turn Right
            self.body.angularVelocity = -10  # -5
            pass
        if action == Action.NOTHING:  # Don't turn
            pass

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

        sys.stdout.write(PrintColor.RED)
        homing_debug.xprint(self, "reached goal: {}".format(self.currentGoalIndex + 1))
        sys.stdout.write(PrintColor.RESET)

        # Reset, Update
        self.startTime = homing_global.timer
        self.startTimestep = homing_global.timestep
        self.currentGoalIndex = (self.currentGoalIndex + 1) % len(self.goals)  # change goal
        self.t2GCollisionCount = 0
        self.t2GAgentCollisionCount = 0

    def update(self):
        super(AgentHoming, self).update()

        # Update sensors value
        self.readSensors()

        # Get orientation to the goal
        orientation = self.orientationToGoal()
        if (0.0 <= orientation < 0.5) or (-0.5 <= orientation < 0.0):
            if not self.facingGoal:
                self.facingGoal = True
                homing_debug.xprint(self, "facing goal: {}".format(self.currentGoalIndex + 1))

        elif (0.5 <= orientation < 1.0) or (-1.0 <= orientation < -0.5):
            if self.facingGoal:
                self.facingGoal = False
                homing_debug.xprint(self, "reverse facing goal: {}".format(self.currentGoalIndex + 1))

        # Select action using AI
        #last_signal = np.asarray([self.sensor1, self.sensor2, self.sensor3, orientation])
        last_signal = np.asarray([self.sensor1, self.sensor2, self.sensor3, orientation, -orientation])
        #action_num = self.brain.update(self.last_reward, last_signal)
        #self.updateDrive(Action(action_num))
        self.updateManualDrive()
        self.updateFriction()

        # Calculate agent's distance to the goal
        self.distance = self.distanceToGoal()

        # Reward mechanism
        self.last_reward = -0.5  # Living penalty
        if self.distance < self.last_distance:  # getting closer
            self.last_reward = 0.1
        if self.sensor1 < 0.1:
            self.last_reward = -1
        if self.sensor2 < 0.1:
            self.last_reward = -1
        if self.sensor3 < 0.1:
            self.last_reward = -1

        # Reached Goal
        if self.distance < self.goalReachedThreshold:
            self.computeGoalReached()
            self.last_reward = 1
            #self.brain.replay()  # experience replay

        self.last_distance = self.distance
        self.elapsedTime = homing_global.timer - self.startTime
        self.elapsedTimestep = homing_global.timestep - self.startTimestep

        return

    def score(self):
        """
            Score is the mean of the reward in the sliding window
        """
        score = self.brain.score()
        return score

    def collisionColor(self):
        """
            Change agent color during collision
        """
        self.color = Color.DeepSkyBlue
        self.raycastSideColor = Color.Magenta
        self.raycastFrontColor = Color.Magenta

    def endCollisionColor(self):
        """
            Reset agent color when end of collision
        """
        self.color = self.initial_color
        self.raycastFrontColor = self.initial_raycastFrontColor
        self.raycastSideColor = self.initial_raycastSideColor