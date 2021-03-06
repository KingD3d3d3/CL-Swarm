
import pygame
from Box2D.b2 import (world, vec2, contactListener)
from task_race.envs.objects.geometric import *
import numpy as np
import res.Util as Util
from res.Util import world_to_pixels
from task_race.envs.objects.Car import Car, Action

SCREEN_WIDTH, SCREEN_HEIGHT = 720, 720
PPM = 20
TARGET_FPS = 60
PHYSICS_TIME_STEP = 1.0 / TARGET_FPS
VEL_ITERS, POS_ITERS = 10, 10

X_MAX = SCREEN_WIDTH / PPM                              # 36
Y_MAX = SCREEN_HEIGHT / PPM                             # 36
RADIUS_OUTTER = SCREEN_WIDTH / PPM / 2                  # 18
ROAD_WIDTH = 7                                          # 7
RADIUS_INNER = (SCREEN_WIDTH / PPM / 2) - ROAD_WIDTH    # 11
CENTER_POINT = vec2(X_MAX / 2, Y_MAX / 2)               # vec2(18, 18)
S_POINT = vec2(CENTER_POINT.x - RADIUS_INNER, CENTER_POINT.y)  # point of angle 0 in inner circle

# Goal Square
M1 = vec2(-np.cos(3 * (np.pi / 12)) * RADIUS_INNER + CENTER_POINT.x, CENTER_POINT.y)
M2 = vec2(0, CENTER_POINT.x)
M3 = vec2(0, np.sin(3 * (np.pi / 12)) * RADIUS_INNER + CENTER_POINT.y)
M4 = vec2(-np.cos(3 * (np.pi / 12)) * RADIUS_INNER + CENTER_POINT.x, np.sin(3 * (np.pi / 12)) * RADIUS_INNER + CENTER_POINT.y)
m1 = world_to_pixels(M1, SCREEN_HEIGHT, PPM)
m2 = world_to_pixels(M2, SCREEN_HEIGHT, PPM)
m3 = world_to_pixels(M3, SCREEN_HEIGHT, PPM)
m4 = world_to_pixels(M4, SCREEN_HEIGHT, PPM)

class RaceMap(object):
    def __init__(self, screen=None, world=None):
        self.screen = screen

        top_right_tri = [Triangle(screen=screen, world=world, screen_height=SCREEN_HEIGHT, screen_width=SCREEN_WIDTH, ppm=PPM,
                                  vertices=[
                                      (0, 0),
                                      (np.cos((i + 1) * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER - X_MAX,
                                       np.sin((i + 1) * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER - Y_MAX),
                                      (np.cos(i * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER - X_MAX,
                                       np.sin(i * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER - Y_MAX)
                                  ],
                                  x=X_MAX, y=Y_MAX)
                         for i in range(0, 6)]

        top_left_tri = [Triangle(screen=screen, world=world, screen_height=SCREEN_HEIGHT, screen_width=SCREEN_WIDTH, ppm=PPM,
                                 vertices=[
                                     (0, 0),
                                     (X_MAX - (np.cos((i - 1) * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER),
                                      np.sin((i - 1) * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER - Y_MAX),
                                     (X_MAX - (np.cos(i * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER),
                                      np.sin(i * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER - Y_MAX)
                                 ],
                                 x=0, y=Y_MAX)
                        for i in range(6, 0, -1)]

        bottom_left_tri = [Triangle(screen=screen, world=world, screen_height=SCREEN_HEIGHT, screen_width=SCREEN_WIDTH, ppm=PPM,
                                    vertices=[
                                        (0, 0),
                                        (X_MAX - (np.cos((i - 1) * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER),
                                         Y_MAX - (np.sin((i - 1) * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER)),
                                        (X_MAX - (np.cos(i * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER),
                                         Y_MAX - (np.sin(i * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER))
                                    ],
                                    x=0, y=0)
                           for i in range(6, 0, -1)]

        bottom_right_tri = [Triangle(screen=screen, world=world, screen_height=SCREEN_HEIGHT, screen_width=SCREEN_WIDTH, ppm=PPM,
                                     vertices=[
                                         (0, 0),
                                         (np.cos((i + 1) * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER - X_MAX,
                                          Y_MAX - (np.sin((i + 1) * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER)),
                                         (np.cos(i * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER - X_MAX,
                                          Y_MAX - (np.sin(i * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER))
                                     ],
                                     x=X_MAX, y=0)
                            for i in range(0, 6)]

        self.triangle_list = top_right_tri + top_left_tri + bottom_left_tri + bottom_right_tri
        self.inner_circle = Circle(screen=self.screen, world=world, screen_height=SCREEN_HEIGHT, screen_width=SCREEN_WIDTH, ppm=PPM,
                                   x=RADIUS_OUTTER, y=RADIUS_OUTTER,
                                   radius=RADIUS_INNER)
        self.wall = Quadrilateral(screen=screen, world=world, screen_height=SCREEN_HEIGHT, screen_width=SCREEN_WIDTH, ppm=PPM,
                                  color=Color.Gray,
                                  x=CENTER_POINT.x,
                                  y=CENTER_POINT.y,
                                  vertices=[
                                      (-np.cos(2 * np.pi / 12) * RADIUS_INNER, np.sin(2 * np.pi / 12) * RADIUS_INNER),
                                      (-np.cos(2 * np.pi / 12) * RADIUS_OUTTER, np.sin(2 * np.pi / 12) * RADIUS_OUTTER),
                                      (-np.cos(3 * np.pi / 12) * RADIUS_OUTTER, np.sin(3 * np.pi / 12) * RADIUS_OUTTER),
                                      (-np.cos(3 * np.pi / 12) * RADIUS_INNER, np.sin(3 * np.pi / 12) * RADIUS_INNER)
                                  ], )

    def render(self):
        # Goal square
        pygame.draw.rect(self.screen, Color.Orange, (m1[0], m1[1], m2[0] - m1[0], m3[1] - m1[1]), 0)
        pygame.draw.rect(self.screen, Color.Orange, (m3[0], m3[1], 100, -50), 0) # leftover

        for triangle in self.triangle_list:
            triangle.render()
        self.wall.render()
        self.inner_circle.render()

        # pygame.draw.line(self.screen, Color.Orange, world_to_pixels(M1, SCREEN_HEIGHT, PPM), world_to_pixels(M2, SCREEN_HEIGHT, PPM))
        # pygame.draw.line(self.screen, Color.Orange, world_to_pixels(M2, SCREEN_HEIGHT, PPM), world_to_pixels(M3, SCREEN_HEIGHT, PPM))
        # pygame.draw.line(self.screen, Color.Orange, world_to_pixels(M3, SCREEN_HEIGHT, PPM), world_to_pixels(M4, SCREEN_HEIGHT, PPM))
        # pygame.draw.line(self.screen, Color.Orange, world_to_pixels(M4, SCREEN_HEIGHT, PPM), world_to_pixels(M1, SCREEN_HEIGHT, PPM))

class RaceContactListener(contactListener):

    def __init__(self):
        contactListener.__init__(self)

    def BeginContact(self, contact):
        if isinstance(contact.fixtureA.body.userData, Car):
            car = contact.fixtureA.body.userData
            car.collision = True
            return

        if isinstance(contact.fixtureB.body.userData, Car):
            car = contact.fixtureB.body.userData
            car.collision = True
            return

class RaceCircleRight(object):

    def __init__(self, display=False, manual=False):

        # -------------------- Pygame Setup ----------------------

        self.display = display
        self.manual = manual
        self.screen = None
        if self.display:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
            pygame.display.set_caption('Environment Race Circle Right')
            self.clock = pygame.time.Clock()
            self.myfont = pygame.font.SysFont("monospace", 15)
            self.delta_time = 1.0 / TARGET_FPS
            self.fps = TARGET_FPS

        # -------------------- Environment and PyBox2d World Setup ----------------------

        # Create the world
        self.world = world(gravity=(0, 0), doSleep=True, contactListener=RaceContactListener())

        # Create physical objects
        self.car = Car(screen=self.screen, world=self.world, screen_height=SCREEN_HEIGHT, screen_width=SCREEN_WIDTH,
                       ppm=PPM, env_name='RaceCircleLeft')
        self.race = RaceMap(screen=self.screen, world=self.world)

        self.max_episode_steps = 1000
        self.prev_shaping = None
        self.timesteps = None
        self.orientation = None
        self.speed = None
        self.goal_reached = None
        self.d2g = None # distance to goal

        self.input_size = 7
        self.action_size = 6

    def _destroy(self):
        self.world.contactListener = None

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = RaceContactListener()
        self.world.contactListener = self.world.contactListener_keepref

        observation = []
        self.prev_shaping = None
        self.car.collision = False
        self.goal_reached = False
        self.car.speed = 0.
        self.timesteps = 0

        # Teleport car to initial position
        start_pos = vec2(CENTER_POINT.x, CENTER_POINT.y + RADIUS_INNER + (ROAD_WIDTH / 2))  # starting point
        self.car.body.position = start_pos
        self.car.body.angle = Util.deg_to_rad(-90)
        self.car.kill_motion() # reset car velocity

        # Distance to goal (debug purpose)
        self.d2g = self.distance_goal

        # Orientation in lane
        self.orientation = self.orientation_lane
        observation.append(self.orientation_lane)

        # Distance in lane
        distance_lane_norm = self.normalize_distance(self.distance_lane, _min=self.car.radius, _max=ROAD_WIDTH - self.car.radius)
        observation.append(distance_lane_norm)

        # Speed
        speed_norm = Util.min_max_normalization_m1_1(0., _min=0., _max=self.car.max_forward_speed)
        observation.append(speed_norm)

        # Angular velocity
        angular_vel_norm = Util.min_max_normalization_m1_1(0., _min=-3.5, _max=3.5)
        observation.append(angular_vel_norm)

        # Sensor's value
        self.car.read_sensors()
        for sensor in self.car.sensors:
            normed_sensor = self.normalize_sensors_value(sensor)
            observation.append(normed_sensor)

        # Go 1 step further to apply the environment reset
        self.world.Step(PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS)
        self.world.ClearForces()

        state = np.array(observation, dtype=np.float32)
        return state

    def close(self):
        pass

    def step(self, action):
        observation = []

        self.car.update_friction()
        if self.manual:
            self.car.update_drive_manual()
            self.car.update_turn_manual()
        else: # select action using AI
            self.car.update_drive(Action(action))
            self.car.update_turn(Action(action))

        self.world_step()
        self.timesteps += 1

        # Distance to goal (debug purpose)
        self.d2g = self.distance_goal

        # Orientation in lane
        self.orientation = self.orientation_lane
        observation.append(self.orientation_lane)

        # Distance in lane
        distance_lane_norm = self.normalize_distance(self.distance_lane, _min=self.car.radius, _max=ROAD_WIDTH - self.car.radius)
        observation.append(distance_lane_norm)

        # Speed
        current_forward_normal = self.car.body.GetWorldVector((0, 1))
        self.car.speed = self.car.forward_velocity.dot(current_forward_normal)
        speed_norm = Util.min_max_normalization_m1_1(self.car.speed, _min=0., _max=self.car.max_forward_speed)
        observation.append(speed_norm)

        # Angular velocity of the car
        angular_vel = self.car.body.angularVelocity
        max_angular_vel = 3.5
        angular_vel_norm = Util.min_max_normalization_m1_1(angular_vel, _min=-max_angular_vel, _max=max_angular_vel)
        observation.append(angular_vel_norm)

        # Sensor's value
        self.car.read_sensors()
        for sensor in self.car.sensors:
            normed_sensor = self.normalize_sensors_value(sensor)  # Normalize sensor's value
            observation.append(normed_sensor)

        # Terminal condition
        if self.inside_goal:
            self.goal_reached = True
            done = True
        elif self.timesteps >= self.max_episode_steps or self.car.collision:
            done = True
        else:
            done = False

        # Reward from the environment
        reward = self.reward_function()

        state = np.array(observation, dtype=np.float32)
        return state, reward, done, None

    def world_step(self):
        """
            Box2D world step
        """
        if self.display:
            self.delta_time = self.clock.tick(TARGET_FPS) / 1000.0
            self.fps = self.clock.get_fps()

            self.world.Step(PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS)
            self.world.ClearForces()
        else:
            # self.delta_time = self.clock.tick() / 1000.0
            self.world.Step(PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS)
            self.world.ClearForces()

    def render(self):
        self.screen.fill((0, 0, 0, 0))

        self.race.render()
        self.car.render()
        # self.gizmo()
        # Util.print_fps(self.screen, self.myfont, 'FPS : ' + str('{:3.2f}').format(self.fps))

        pygame.display.flip()

    def gizmo(self):
        pygame.draw.line(self.screen, Color.Magenta, world_to_pixels(CENTER_POINT, SCREEN_HEIGHT, PPM), world_to_pixels(S_POINT + vec2(-ROAD_WIDTH, 0), SCREEN_HEIGHT, PPM))

        theta = Util.angle_direct(Util.normalize(S_POINT - CENTER_POINT), Util.normalize(self.car.body.position - CENTER_POINT))
        theta = Util.deg_to_rad(theta)
        h = vec2(-RADIUS_INNER * np.cos(theta) + CENTER_POINT.x, -RADIUS_INNER * np.sin(theta) + CENTER_POINT.y) # orthogonal projection
        pygame.draw.line(self.screen, Color.Red, world_to_pixels(CENTER_POINT, SCREEN_HEIGHT, PPM), world_to_pixels(h, SCREEN_HEIGHT, PPM))

        tangent = Util.rotate(Util.normalize(CENTER_POINT - h), 90.0) # tangent to the circle
        pygame.draw.line(self.screen, Color.Yellow, world_to_pixels(h, SCREEN_HEIGHT, PPM), world_to_pixels(h + tangent, SCREEN_HEIGHT, PPM))

    @property
    def orientation_lane(self):
        """
            Get agent orientation in lane
        """
        theta = Util.angle_direct(Util.normalize(S_POINT - CENTER_POINT), Util.normalize(self.car.body.position - CENTER_POINT))
        theta = Util.deg_to_rad(theta)
        h = vec2(-RADIUS_INNER * np.cos(theta) + CENTER_POINT.x, -RADIUS_INNER * np.sin(theta) + CENTER_POINT.y) # orthogonal projection
        tangent = Util.rotate(Util.normalize(CENTER_POINT - h), 90.0) # tangent to the circle

        forward = Util.normalize(self.car.body.GetWorldVector((0, 1)))
        orientation = Util.angle_indirect(forward, tangent) / 180.0

        return orientation

    @property
    def distance_lane(self):
        d = RADIUS_OUTTER - Util.distance(self.car.body.position, CENTER_POINT)
        return d

    @staticmethod
    def normalize_distance(val, _min, _max):
        return Util.min_max_normalization_m1_1(val, _min=_min, _max=_max)

    def normalize_sensors_value(self, val):
        return Util.min_max_normalization_m1_1(val, _min=0.0, _max=self.car.raycast_length)

    @property
    def distance_goal(self):
        theta = Util.angle_direct(Util.normalize(self.car.body.position - CENTER_POINT), Util.normalize(S_POINT - CENTER_POINT))

        if theta < 0:
            theta = 180.0 + (180.0 + theta)
        theta = Util.deg_to_rad(theta)
        phi = 2 * np.pi - theta
        d2g = RADIUS_INNER * phi

        if self.inside_goal:
            return 0.

        d2g = np.abs(d2g)
        return d2g

    @property
    def inside_goal(self):
        pos = self.car.body.position
        if M2.x <= pos.x <= M1.x and M2.y <= pos.y <= M3.y:
            return True
        else:
            return False

    def reward_function(self):
        """
            Core function of the agent for training
        """
        # flag_col = False
        if self.car.collision:
            # flag_col = True
            return Reward.COLLISION

        # Process sensor's value
        r = np.fromiter((Reward.reward_sensor(s, max_length=self.car.raycast_length) for s in self.car.sensors),
                        self.car.sensors.dtype,
                        count=len(self.car.sensors))  # Apply sensor reward function to all sensors value
        reward_sensors = np.amin(r)  # take the min value

        # Flag that indicates the direction in the road
        if (0.5 < self.orientation <= 1) or (-1 <= self.orientation < -0.5):
            direction_road = -1. # punish speed when reverse facing the road
        else:
            direction_road = 1.

        # Reward shaping based on speed of agent
        potential_shaping = 0
        shaping = direction_road * 10. * self.car.speed # coeff of 10 to make it 10^0 (1 meter) order and not millimeter
        if self.prev_shaping is not None:
            """
                The faster the agent is -> higher the reward
                Get slower -> penalty
                Keep constant speed -> no reward
            """
            potential_shaping = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Overall reward equation
        # reward = flag_col * Reward.COLLISION + reward_sensors + potential_shaping
        reward = reward_sensors + potential_shaping

        return reward

class Reward:
    COLLISION = -100.

    @classmethod
    def reward_sensor(cls, val, max_length):
        """
            Shaping reward on sensors
        """
        ok = 0.                 # no punishment
        little = -0.1             # small punishment
        worst = cls.COLLISION   # higher punishment

        min_length = 0.0
        middle = (max_length + min_length) / 2

        if middle <= val <= max_length:
            # Linear
            b = (max_length, ok)
            a = (middle, little)
            m = (b[1] - a[1]) / (b[0] - a[0])
            p = a[1] - m * a[0]
            y = m * val + p

            return y
        else:
            # Linear
            b = (middle, little)
            a = (0, worst)
            m = (b[1] - a[1]) / (b[0] - a[0])
            p = a[1] - m * a[0]
            y = m * val + p

            return y