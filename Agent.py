import random, sys

import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (vec2)
from enum import Enum
from pygame.locals import *

import res.colors as Color
from Setup import *
from Util import worldToPixels


# Agent's possible actions
class Action(Enum):
    TURN_LEFT = 1
    TURN_RIGHT = 2
    NOTHING = 3


moveTicker = 0


class Agent(object):
    def __init__(self, screen=None, world=None, x=0, y=0, angle=0, radius=2):
        self.screen = screen
        self.world = world

        self.id = random.randint(0, sys.maxint)
        self.radius = radius
        self.body = self.world.CreateDynamicBody(
            position=(x, y), userData=self, angle=angle)
        self.fixture = self.body.CreateCircleFixture(
            radius=radius, density=1, friction=0) # friction=0.3
        self.color = Color.Magenta
        self.action = Action.TURN_LEFT  # default action is turn LEFT

    def getLateralVelocity(self):
        currentRightNormal = self.body.GetWorldVector(vec2(1, 0))
        return currentRightNormal.dot(self.body.linearVelocity) * currentRightNormal

    def getForwardVelocity(self):
        currentForwardNormal = self.body.GetWorldVector(vec2(0, 1))
        return currentForwardNormal.dot(self.body.linearVelocity) * currentForwardNormal

    def updateFriction(self):
        impulse = self.body.mass * -self.getLateralVelocity()
        self.body.ApplyLinearImpulse(impulse, self.body.worldCenter, True)  # kill lateral velocity
        self.body.ApplyAngularImpulse(0.3 * self.body.inertia * - self.body.angularVelocity,
                                      True)  # kill angular velocity #0.1

        # Stop the forever roll
        currentForwardNormal = self.getForwardVelocity()
        currentForwardSpeed = currentForwardNormal.Normalize()
        dragForceMagnitude = -50 * currentForwardSpeed #-10
        self.body.ApplyForce(dragForceMagnitude * currentForwardNormal, self.body.worldCenter, True)

    def update(self):
        raise NotImplementedError("Update method not implemented")

    def updateManualDriveTestAngle(self, angle):
        speed = 12
        global moveTicker

        key = pygame.key.get_pressed()
        if key[K_LEFT]:  # Turn Left
            if moveTicker == 0:
                self.body.angularVelocity = angle
                print('left pressed')
            moveTicker += 1
            if moveTicker > 60:
                moveTicker = 0
            pass
        if key[K_RIGHT]:  # Turn Right
            if moveTicker == 0:
                self.body.angularVelocity = -angle
                print('right pressed')
            moveTicker += 1
            if moveTicker > 60:
                moveTicker = 0
            pass
        if key[K_SPACE]:  # Break
            speed = 0
            pass
        forward_vec = self.body.GetWorldVector((0, 1))
        self.body.linearVelocity = forward_vec * speed

    def updateManualDrive(self):
        speed = 12

        key = pygame.key.get_pressed()
        if key[K_LEFT]:  # Turn Left
            self.body.angularVelocity = 5
            pass
        if key[K_RIGHT]:  # Turn Right
            self.body.angularVelocity = -5
            pass
        if key[K_SPACE]:  # Break
            speed = 0
            pass
        forward_vec = self.body.GetWorldVector((0, 1))
        self.body.linearVelocity = forward_vec * speed

    def draw(self):
        position = self.body.transform * self.fixture.shape.pos * PPM
        position = (position[0], SCREEN_HEIGHT - position[1])
        pygame.draw.circle(self.screen, self.color, [int(x) for x in position], int(self.radius * PPM))

        current_forward_normal = self.body.GetWorldVector((0, 1))
        pygame.draw.line(self.screen, Color.White, worldToPixels(self.body.worldCenter),
                         worldToPixels(self.body.worldCenter + current_forward_normal * self.radius))

    def updateAutoDrive(self, change=False):
        speed = 12

        if change:
            self.action = random.choice(list(Action)) # select random action

        if self.action == Action.TURN_LEFT:  # Turn Left
            self.body.angularVelocity = 5
            pass
        if self.action == Action.TURN_RIGHT:  # Turn Right
            self.body.angularVelocity = -5
            pass
        if self.action == Action.NOTHING:  # Don't turn
            pass

        forward_vec = self.body.GetWorldVector((0, 1))
        self.body.linearVelocity = forward_vec * speed