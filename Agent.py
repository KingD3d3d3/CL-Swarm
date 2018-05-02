import random, sys

import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (vec2, pi)
from enum import Enum
from pygame.locals import *

try:
    # Running in PyCharm
    import res.colors as Color
    from Setup import *
    from Util import worldToPixels
    import Util
except:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running from command line -> Import libraries as package')
    from .res import colors as Color
    from .Setup import *
    from .Util import worldToPixels
    import Util


moveTicker = 0
prev_angle = 999
go_print_Turn = False
prev_turned_angle = 0

class Agent(object):
    def __init__(self, screen=None, world=None, x=0, y=0, angle=0, radius=2):
        self.screen = screen
        self.world = world

        self.id = random.randint(0, sys.maxint)
        self.radius = radius
        self.body = self.world.CreateDynamicBody(
            position=(x, y), userData=self, angle=angle)
        self.fixture = self.body.CreateCircleFixture(
            radius=radius, density=1, friction=0, restitution=0)  # friction=0.3
        self.initial_color = Color.Magenta
        self.color = Color.Magenta

        self.updateCalls = 0

    def getLateralVelocity(self):
        currentRightNormal = self.body.GetWorldVector(vec2(1, 0))
        return currentRightNormal.dot(self.body.linearVelocity) * currentRightNormal

    def getForwardVelocity(self):
        currentForwardNormal = self.body.GetWorldVector(vec2(0, 1))
        return currentForwardNormal.dot(self.body.linearVelocity) * currentForwardNormal

    def updateFriction(self):
        impulse = self.body.mass * -self.getLateralVelocity()
        self.body.ApplyLinearImpulse(impulse, self.body.worldCenter, True)  # kill lateral velocity
        self.body.ApplyAngularImpulse(0.8 * self.body.inertia * - self.body.angularVelocity,
                                      True)  # kill angular velocity #0.1 #0.3

        # Stop the forever roll
        currentForwardNormal = self.getForwardVelocity()
        currentForwardSpeed = currentForwardNormal.Normalize()
        dragForceMagnitude = -50 * currentForwardSpeed #-10
        self.body.ApplyForce(dragForceMagnitude * currentForwardNormal, self.body.worldCenter, True)

    def remainStatic(self):
        self.updateFriction()
        speed = 0

        forward_vec = self.body.GetWorldVector((0, 1))
        self.body.linearVelocity = forward_vec * speed

    def update(self):
        # raise NotImplementedError("Update method not implemented")
        self.updateCalls += 1

    def updateManualDriveTestAngle(self, angle):
        speed = 12
        global moveTicker
        global prev_angle
        global go_print_Turn
        global prev_turned_angle

        if go_print_Turn:
            myAngle = Util.radToDeg(self.body.angle % (2 * pi))
            turned_angle = myAngle - prev_angle
            # print('bodyAngle rad: {}, prev_angle rad: {}, angle turned : {}'.format(self.body.angle,
            #                                                                 prev_angle,
            #                                                                 Util.radToDeg(turned_angle % (2 * pi))))
            print('bodyAngle: {}, prev_angle: {}, angle turned : {}'.format(myAngle,
                                                                            prev_angle,
                                                                            turned_angle))
            #if turned_angle != 0 and prev_angle != 999:
            go_print_Turn = False

        myAngle = Util.radToDeg(self.body.angle % (2 * pi))
        turned_angle = myAngle - prev_angle
        if not (prev_turned_angle - 0.1 <= turned_angle <= prev_turned_angle \
                or prev_turned_angle <= turned_angle <= prev_turned_angle + 0.1):
            print('### bodyAngle: {}, prev_angle: {}, angle turned : {}'.format(myAngle,
                                                                        prev_angle,
                                                                        turned_angle))
        prev_turned_angle = turned_angle

        key = pygame.key.get_pressed()
        if key[K_LEFT]:  # Turn Left
            if moveTicker == 0:
                self.body.angularVelocity = angle
                prev_angle = Util.radToDeg(self.body.angle % (2 * pi))
                print('left pressed')
                go_print_Turn = True
            moveTicker += 1

            if moveTicker > 60:
                moveTicker = 0
            pass
        if key[K_RIGHT]:  # Turn Right
            if moveTicker == 0:
                self.body.angularVelocity = -angle
                prev_angle = Util.radToDeg(self.body.angle % (2 * pi))
                print('right pressed')
                go_print_Turn = True
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
