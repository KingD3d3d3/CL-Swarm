import numpy as np
import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from enum import Enum
from pygame.locals import *

import res.colors as Color
from AI.DQN import Dqn
from AI.pytorch import Dqn
from Agent import Agent
from Setup import *
from Util import worldToPixels, pixelsToWorld, normalize, angle
from homing_task.RayCastCallback import RayCastCallback


# -------------------- Agent ----------------------

# Agent's possible actions
class Action(Enum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    NOTHING = 2


class AgentHomingPytorch(Agent):
    def __init__(self, screen=None, world=None, x=0, y=0, angle=0, radius=2, goal_threshold=100):
        super(AgentHomingPytorch, self).__init__(screen, world, x, y, angle, radius)
        self.sensor1 = 0  # left raycast
        self.sensor2 = 0  # middle raycast
        self.sensor3 = 0  # right raycast
        self.brain = Dqn(input_size=5, nb_action=len(list(Action)))

        self.goal = pixelsToWorld((goal_threshold, goal_threshold))

        self.last_reward = 0
        self.last_distance = 0

        self.raycastMaxLength = 2.0

        top_left = self.body.GetWorldVector((-0.70, 0.70))
        self.raycastLeft_point1 = self.body.worldCenter + top_left * self.radius
        self.raycastLeft_point2 = self.raycastLeft_point1 + top_left * 2
        #self.raycastLeft_point1 = vec2(0, 0)
        #self.raycastLeft_point2 = vec2(0, 0)

        forward_vec = self.body.GetWorldVector((0, 1))
        self.raycastFront_point1 = self.body.worldCenter + forward_vec * self.radius
        self.raycastFront_point2 = self.raycastFront_point1 + forward_vec * 2
        #self.raycastFront_point1 = vec2(0, 0)
        #self.raycastFront_point2 = vec2(0, 0)

        top_right = self.body.GetWorldVector((0.70, 0.70))  # sqr(2) / 2
        self.raycastRight_point1 = self.body.worldCenter + top_right * self.radius
        self.raycastRight_point2 = self.raycastRight_point1 + top_right * 2
        #self.raycastRight_point1 = vec2(0, 0)
        #self.raycastRight_point2 = vec2(0, 0)

    def draw(self):
        position = self.body.transform * self.fixture.shape.pos * PPM
        position = (position[0], SCREEN_HEIGHT - position[1])
        pygame.draw.circle(self.screen, self.color, [int(x) for x in position], int(self.radius * PPM))

        current_forward_normal = self.body.GetWorldVector((0, 1))
        pygame.draw.line(self.screen, Color.White, worldToPixels(self.body.worldCenter),
                         worldToPixels(self.body.worldCenter + current_forward_normal * self.radius))

        # Raycasts draw
        top_left = self.body.GetWorldVector((-0.70, 0.70))
        self.raycastLeft_point1 = self.body.worldCenter + top_left * self.radius
        self.raycastLeft_point2 = self.raycastLeft_point1 + top_left * 2
        pygame.draw.line(self.screen, Color.Yellow, worldToPixels(self.raycastLeft_point1),
                         worldToPixels(self.raycastLeft_point2))  # draw the raycast

        forward_vec = self.body.GetWorldVector((0, 1))
        self.raycastFront_point1 = self.body.worldCenter + forward_vec * self.radius
        self.raycastFront_point2 = self.raycastFront_point1 + forward_vec * 2
        pygame.draw.line(self.screen, Color.Red, worldToPixels(self.raycastFront_point1),
                         worldToPixels(self.raycastFront_point2))  # draw the raycast

        top_right = self.body.GetWorldVector((0.70, 0.70))  # sqr(2) / 2
        self.raycastRight_point1 = self.body.worldCenter + top_right * self.radius
        self.raycastRight_point2 = self.raycastRight_point1 + top_right * 2
        pygame.draw.line(self.screen, Color.Blue, worldToPixels(self.raycastRight_point1),
                         worldToPixels(self.raycastRight_point2))  # draw the raycast

    def drawEntity(self, entityPos):
        position = self.body.transform * entityPos * PPM
        position = (position[0], SCREEN_HEIGHT - position[1])
        pygame.draw.circle(self.screen, self.color, [int(x) for x in position], int(self.radius * PPM))

        # current_forward_normal = self.body.GetWorldVector((0, 1))
        # pygame.draw.line(self.screen, Color.White, worldToPixels(self.body.worldCenter),
        #                  worldToPixels(self.body.worldCenter + current_forward_normal * self.radius))
        #
        # # Raycasts draw
        # top_left = self.body.GetWorldVector((-0.70, 0.70))
        # self.raycastLeft_point1 = self.body.worldCenter + top_left * self.radius
        # self.raycastLeft_point2 = self.raycastLeft_point1 + top_left * 2
        # pygame.draw.line(self.screen, Color.Yellow, worldToPixels(self.raycastLeft_point1),
        #                  worldToPixels(self.raycastLeft_point2))  # draw the raycast
        #
        # forward_vec = self.body.GetWorldVector((0, 1))
        # self.raycastFront_point1 = self.body.worldCenter + forward_vec * self.radius
        # self.raycastFront_point2 = self.raycastFront_point1 + forward_vec * 2
        # pygame.draw.line(self.screen, Color.Red, worldToPixels(self.raycastFront_point1),
        #                  worldToPixels(self.raycastFront_point2))  # draw the raycast
        #
        # top_right = self.body.GetWorldVector((0.70, 0.70))  # sqr(2) / 2
        # self.raycastRight_point1 = self.body.worldCenter + top_right * self.radius
        # self.raycastRight_point2 = self.raycastRight_point1 + top_right * 2
        # pygame.draw.line(self.screen, Color.Blue, worldToPixels(self.raycastRight_point1),
        #                  worldToPixels(self.raycastRight_point2))  # draw the raycast


    def readSensors(self):
        # TODO refactor raycast method, multiple copies of the same code

        # Raycast Left
        rayCastLeft = RayCastCallback()
        # top_left = self.body.GetWorldVector((-0.70, 0.70))
        # self.raycastLeft_point1 = self.body.worldCenter + top_left * self.radius
        # self.raycastLeft_point2 = self.raycastLeft_point1 + top_left * 2
        self.world.RayCast(rayCastLeft, self.raycastLeft_point1, self.raycastLeft_point2)
        if rayCastLeft.hit:
            dist1 = (self.raycastLeft_point1 - rayCastLeft.point).length  # distance to the hit point
            self.sensor1 = round(dist1, 2)
        else:
            self.sensor1 = self.raycastMaxLength

        # Raycast Front
        rayCastFront = RayCastCallback()
        # forward_vec = self.body.GetWorldVector((0, 1))
        # self.raycastFront_point1 = self.body.worldCenter + forward_vec * self.radius
        # self.raycastFront_point2 = self.raycastFront_point1 + forward_vec * 2
        self.world.RayCast(rayCastFront, self.raycastFront_point1, self.raycastFront_point2)
        if rayCastFront.hit:
            dist2 = (self.raycastFront_point1 - rayCastFront.point).length  # distance to the hit point
            self.sensor2 = round(dist2, 2)
            #print('val sensor front', self.sensor2)
        else:
            self.sensor2 = self.raycastMaxLength

        # Raycast Right
        rayCastRight = RayCastCallback()
        # top_right = self.body.GetWorldVector((0.70, 0.70))  # sqr(2) / 2
        # self.raycastRight_point1 = self.body.worldCenter + top_right * self.radius
        # self.raycastRight_point2 = self.raycastRight_point1 + top_right * 2
        self.world.RayCast(rayCastRight, self.raycastRight_point1, self.raycastRight_point2)
        if rayCastRight.hit:
            dist3 = (self.raycastRight_point1 - rayCastRight.point).length  # distance to the hit point
            self.sensor3 = round(dist3, 2)
        else:
            self.sensor3 = self.raycastMaxLength

    def updateDrive(self, action):
        speed = 12

        if action == Action.TURN_LEFT:  # Turn Left
            self.body.angularVelocity = 10 # 5
            pass
        if action == Action.TURN_RIGHT:  # Turn Right
            self.body.angularVelocity = -10 # -5
            pass
        if action == Action.NOTHING:  # Don't turn
            pass

        forward_vec = self.body.GetWorldVector((0, 1))
        self.body.linearVelocity = forward_vec * speed

    def update(self):

        # Update sensors value
        self.readSensors()

        # Get orientation to the goal
        toGoal = normalize(self.goal - self.body.position)
        forward = normalize(self.body.GetWorldVector((0, 1)))
        orientation = angle(forward, toGoal) / 180.0
        orientation = round(orientation, 2)  # only 3 decimals

        # Select Action using AI
        #last_signal = np.asarray([0., 0., 0., orientation, -orientation])
        last_signal = np.asarray([self.sensor1, self.sensor2, self.sensor3, orientation, -orientation])
        action_num = self.brain.update(self.last_reward, last_signal)
        self.updateDrive(Action(action_num))
        #self.updateManualDrive()
        self.updateFriction()

        # Reward mechanism
        distance = np.sqrt((self.body.position.x - self.goal.x) ** 2 + (self.body.position.y - self.goal.y) ** 2)
        self.last_reward = -0.5
        if distance < self.last_distance:  # getting closer
            self.last_reward = 0.1

        if self.sensor1 < 0.1:
            self.last_reward = -1
        if self.sensor2 < 0.1:
            self.last_reward = -1
        if self.sensor3 < 0.1:
            self.last_reward = -1

        if distance < 2.5:
            self.goal.x = SCREEN_WIDTH / PPM - self.goal.x
            self.goal.y = SCREEN_HEIGHT / PPM - self.goal.y
            self.last_reward = 1

        self.last_distance = distance

        return

