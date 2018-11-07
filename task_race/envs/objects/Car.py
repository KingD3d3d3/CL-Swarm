import pygame
from enum import Enum
from Box2D.b2 import (vec2)
from pygame.locals import *
import numpy as np
import res.colors as Color
from Box2D import b2RayCastCallback
from Util import world_to_pixels

class RayCastCallback(b2RayCastCallback):
    """
    This class captures the closest hit shape.
    """
    def __init__(self, **kwargs):
        b2RayCastCallback.__init__(self)
        self.fixture = None
        self.hit = False

    # Called for each fixture found in the query. You control how the ray proceeds
    # by returning a float that indicates the fractional length of the ray. By returning
    # 0, you set the ray length to zero. By returning the current fraction, you proceed
    # to find the closest point. By returning 1, you continue with the original ray
    # clipping.
    def ReportFixture(self, fixture, point, normal, fraction):
        self.fixture = fixture
        self.point = vec2(point)
        self.normal = vec2(normal)
        self.hit = True # flag to inform raycast hit an object
        self.fraction = fraction
        # You will get this error: "TypeError: Swig director type mismatch in output value of type 'float32'"
        # without returning a value
        return fraction

# Car actions
class Action(Enum):
    FORWARD_LEFT = 0
    FORWARD = 1
    FORWARD_RIGHT = 2
    BRAKE_LEFT = 3
    BRAKE = 4
    BRAKE_RIGHT = 5

class Car(object):
    def __init__(self, screen=None, world=None, x=0, y=0, angle=0, radius=1.25, screen_height=None, screen_width=None, ppm=20.0,
                 env_name='RaceCircleLeft'):
        self.screen = screen
        self.world = world
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.ppm = ppm
        self.env_name = env_name

        self.radius = radius
        self.body = self.world.CreateDynamicBody(position=(x, y), userData=self, angle=angle)
        self.fixture = self.body.CreateCircleFixture(radius=radius, density=1, friction=0, restitution=0)
        self.color = Color.Red

        # Speed specs
        self.max_forward_speed = 100 # 50
        self.max_drive_force = 300 # can reach maximum speed in 2.833 s (0-100) # 180
        self.turn_torque = 80  # achieved max_angular_velocity of 3.12 (~pi) rad/s in 2.5s
        self.speed = 0.

        # Proximity Sensors
        self.raycast_length = 3.0
        self.raycastSafeColor = Color.Green
        self.raycastDangerColor = Color.Red
        front_middle = vec2(0, 1)
        front_right = vec2(np.sqrt(2) / 2, np.sqrt(2) / 2)
        front_left = vec2(-np.sqrt(2) / 2, np.sqrt(2) / 2)
        self.raycast_vectors = (front_left, front_middle, front_right)  # Store raycast direction vector for each raycast
        self.num_sensors = len(self.raycast_vectors)
        self.sensors = np.ones(self.num_sensors) * self.raycast_length  # store the value of each sensors

        self.collision = False

    @property
    def lateral_velocity(self):
        right_normal = self.body.GetWorldVector(vec2(1, 0))
        return right_normal.dot(self.body.linearVelocity) * right_normal

    @property
    def forward_velocity(self):
        current_normal = self.body.GetWorldVector(vec2(0, 1))
        return current_normal.dot(self.body.linearVelocity) * current_normal

    def kill_motion(self):
        """
            Directly kill the velocity and angular-velocity
        """
        forward_vec = self.body.GetWorldVector((0, 1))
        self.body.linearVelocity = forward_vec * 0.
        self.body.angularVelocity = 0

    def kill_forward(self):
        """
            Annihilate the forward velocity
        """
        impulse = -self.forward_velocity * self.body.mass * (2. / 3.)
        self.body.ApplyLinearImpulse(impulse, self.body.worldCenter, True)  # kill forward

    def update_friction(self):
        impulse = -self.lateral_velocity * self.body.mass
        self.body.ApplyLinearImpulse(impulse, self.body.worldCenter, True)                          # kill lateral velocity
        self.body.ApplyAngularImpulse(0.1 * self.body.inertia * - self.body.angularVelocity, True)  # kill angular velocity

        current_forward_normal = self.forward_velocity
        current_forward_speed = current_forward_normal.Normalize()
        drag_force_magnitude = -2 * current_forward_speed
        self.body.ApplyForce(drag_force_magnitude * current_forward_normal, self.body.worldCenter, True) # stop the forever roll

    def update_drive_manual(self):
        key = pygame.key.get_pressed()
        if key[K_SPACE]:    # brake
            desired_speed = 0.
        else:               # forward
            desired_speed = self.max_forward_speed

        current_forward_normal = self.body.GetWorldVector((0, 1))
        current_speed = self.forward_velocity.dot(current_forward_normal) # current speed in the forward direction

        # apply necessary force
        if desired_speed > current_speed:
            force = self.max_drive_force
        elif desired_speed < current_speed:
            force = -self.max_drive_force
        else:
            return

        # handle negative speed case
        if desired_speed == 0. and current_speed <= 0.:
            self.kill_forward()
            return

        self.body.ApplyForce(force * current_forward_normal, self.body.worldCenter, True)

    def update_turn_manual(self):
        key = pygame.key.get_pressed()
        if key[K_LEFT]:
            desired_torque = self.turn_torque
        elif key[K_RIGHT]:
            desired_torque = -self.turn_torque
        else:
            return

        self.body.ApplyTorque(desired_torque, True)

    def update_drive(self, action):
        if action == Action.FORWARD or action == Action.FORWARD_LEFT or action == Action.FORWARD_RIGHT:     # forward
            desired_speed = self.max_forward_speed
        elif action == Action.BRAKE or action == Action.BRAKE_LEFT or action == Action.BRAKE_RIGHT:         # brake
            desired_speed = 0.
        else:
            return

        # current speed in the forward direction
        current_forward_normal = self.body.GetWorldVector((0, 1))
        current_speed = self.forward_velocity.dot(current_forward_normal)

        # apply necessary force
        if desired_speed > current_speed:
            force = self.max_drive_force
        elif desired_speed < current_speed:
            force = -self.max_drive_force
        else:
            return

        # handle negative speed case
        if desired_speed == 0. and current_speed <= 0.:
            self.kill_forward()
            return

        self.body.ApplyForce(force * current_forward_normal, self.body.worldCenter, True)

    def update_turn(self, action):
        if action == Action.BRAKE_LEFT or action == Action.FORWARD_LEFT:        # turn left
            desired_torque = self.turn_torque
        elif action == Action.BRAKE_RIGHT or action == Action.FORWARD_RIGHT:    # turn right
            desired_torque = -self.turn_torque
        else:
            return

        self.body.ApplyTorque(desired_torque, True)

    def read_sensors(self):
        for i in range(self.num_sensors):
            raycast = RayCastCallback()
            v = self.body.GetWorldVector(self.raycast_vectors[i])
            p1 = self.body.worldCenter + v * self.radius
            p2 = p1 + v * self.raycast_length
            self.world.RayCast(raycast, p1, p2)
            if raycast.hit:
                dist = (p1 - raycast.point).length  # distance to the hit point
                self.sensors[i] = dist
            else:
                self.sensors[i] = self.raycast_length

    def render(self):
        position = self.body.transform * self.fixture.shape.pos * self.ppm
        position = (position[0], self.screen_height - position[1])
        pygame.draw.circle(self.screen, self.color, [int(x) for x in position], int(self.radius * self.ppm))

        current_forward_normal = self.body.GetWorldVector((0, 1))
        pygame.draw.line(self.screen, Color.White, world_to_pixels(self.body.worldCenter, self.screen_height, self.ppm),
                         world_to_pixels(self.body.worldCenter + current_forward_normal * self.radius, self.screen_height, self.ppm))

        # Draw raycasts
        for i in range(self.num_sensors):
            v = self.body.GetWorldVector(self.raycast_vectors[i])
            p1 = self.body.worldCenter + v * self.radius
            p2 = p1 + v * self.sensors[i]
            if self.sensors[i] <= self.raycast_length / 2:
                ray_color = self.raycastDangerColor
            else:
                ray_color = self.raycastSafeColor
            pygame.draw.line(self.screen, ray_color, world_to_pixels(p1, self.screen_height, self.ppm), world_to_pixels(p2, self.screen_height, self.ppm))
