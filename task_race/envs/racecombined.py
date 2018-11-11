
import pygame
from Box2D.b2 import (world, vec2, contactListener)
from task_race.envs.objects.geometric import *
import numpy as np
import res.Util as Util
from res.Util import world_to_pixels
from task_race.envs.objects.Car import Car, Action

SCREEN_WIDTH, SCREEN_HEIGHT = 1300, 720
PPM = 20
TARGET_FPS = 60
PHYSICS_TIME_STEP = 1.0 / TARGET_FPS
VEL_ITERS, POS_ITERS = 10, 10

X_MAX = SCREEN_WIDTH / PPM                              # 65
Y_MAX = SCREEN_HEIGHT / PPM                             # 36
Y_MID = Y_MAX / 2                                       # 18
RADIUS_OUTTER = 18
RADIUS_INNER = 11
ROAD_WIDTH = RADIUS_OUTTER - RADIUS_INNER               # 7

# Goal Square
M1 = vec2(0, Y_MID)                                     # vec2(0, 18)
M2 = vec2(ROAD_WIDTH, Y_MID)                            # vec2(7, 18)
M3 = vec2(0, Y_MID + 7)                                 # vec2(0, 25)
M4 = vec2(ROAD_WIDTH, Y_MID + 7)                        # vec2(7, 25)

C2 = vec2(RADIUS_OUTTER, Y_MID)                         # vec2(18, 18)
S2_POINT = vec2(C2.x - RADIUS_INNER, C2.y)              # vec2(7, 18) # equal M2
C1 = vec2((RADIUS_OUTTER * 2) + RADIUS_INNER, Y_MID)    # vec2(47, 18)
S1_POINT = vec2(C1.x + RADIUS_INNER, C1.y)              # vec2(58, 18)
HALFWAY = vec2(RADIUS_OUTTER * 2, Y_MID)                # vec2(36, 18)

class RaceMap(object):
    def __init__(self, screen=None, world=None):
        self.screen = screen

        self.circle_c2 = Circle(screen=self.screen, world=world, screen_height=SCREEN_HEIGHT, screen_width=SCREEN_WIDTH, ppm=PPM,
                                   x=C2.x, y=C2.y,
                                   radius=RADIUS_INNER)

        self.circle_c1 = Circle(screen=self.screen, world=world, screen_height=SCREEN_HEIGHT, screen_width=SCREEN_WIDTH, ppm=PPM,
                                   x=C1.x, y=C1.y,
                                   radius=RADIUS_INNER)
        self.wall_2 = Quadrilateral(screen=self.screen, world=world, x=ROAD_WIDTH, y=Y_MID,
                                    vertices=[
                                        (0, 0),
                                        ((RADIUS_INNER * 2), 0),
                                        ((RADIUS_INNER * 2), Y_MID),
                                        (0, Y_MID)
                                    ],
                                    color=Color.Gray, screen_height=SCREEN_HEIGHT, screen_width=SCREEN_WIDTH, ppm=PPM)
        self.wall_1 = Quadrilateral(screen=self.screen, world=world, x=36, y=0,
                                    vertices=[
                                        (0, 0),
                                        ((RADIUS_INNER * 2), 0),
                                        ((RADIUS_INNER * 2), Y_MID),
                                        (0, Y_MID)
                                    ],
                                    color=Color.Gray, screen_height=SCREEN_HEIGHT, screen_width=SCREEN_WIDTH, ppm=PPM)

        top_right_tri = [
            Triangle(screen=screen, world=world, screen_height=SCREEN_HEIGHT, screen_width=SCREEN_WIDTH, ppm=PPM,
                     x=X_MAX, y=Y_MAX,
                     vertices=[
                         (0, 0),
                         (np.cos((i + 1) * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER - (ROAD_WIDTH/2 + X_MAX/2),
                          np.sin((i + 1) * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER - Y_MAX),
                         (np.cos(i * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER - (ROAD_WIDTH/2 + X_MAX/2),
                          np.sin(i * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER - Y_MAX)
                     ]
                     )
            for i in range(0, 6)]
        top_left_tri = [
            Triangle(screen=screen, world=world, screen_height=SCREEN_HEIGHT, screen_width=SCREEN_WIDTH, ppm=PPM,
                     x=(RADIUS_INNER * 2) + ROAD_WIDTH, y=Y_MAX,
                     vertices=[
                         (0, 0),
                         ((ROAD_WIDTH/2 + X_MAX/2) - (np.cos((i - 1) * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER),
                          np.sin((i - 1) * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER - Y_MAX),
                         ((ROAD_WIDTH/2 + X_MAX/2) - (np.cos(i * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER),
                          np.sin(i * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER - Y_MAX)
                     ])
            for i in range(6, 0, -1)]
        bottom_right_tri = [Triangle(screen=screen, world=world, screen_height=SCREEN_HEIGHT, screen_width=SCREEN_WIDTH,
                                     ppm=PPM, x=36, y=0,
                                     vertices=[
                                         (0, 0),
                                         (np.cos((i + 1) * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER - (ROAD_WIDTH/2 + X_MAX/2),
                                          Y_MAX - (np.sin((i + 1) * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER)),
                                         (np.cos(i * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER - (ROAD_WIDTH/2 + X_MAX/2),
                                          Y_MAX - (np.sin(i * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER))
                                     ])
                            for i in range(0, 6)]
        bottom_left_tri = [
            Triangle(screen=screen, world=world, screen_height=SCREEN_HEIGHT, screen_width=SCREEN_WIDTH, ppm=PPM,
                     x=0, y=0,
                     vertices=[
                         (0, 0),
                         ((ROAD_WIDTH/2 + X_MAX/2) - (np.cos((i - 1) * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER),
                          Y_MAX - (np.sin((i - 1) * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER)),
                         ((ROAD_WIDTH/2 + X_MAX/2) - (np.cos(i * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER),
                          Y_MAX - (np.sin(i * np.pi / 12) * RADIUS_OUTTER + RADIUS_OUTTER))
                     ])
            for i in range(6, 0, -1)]

        self.triangle_list = top_right_tri + top_left_tri + bottom_right_tri + bottom_left_tri

        # Boundary wall at race start
        self.boundary = []
        offset = 0
        boundary1 = Quadrilateral(screen=self.screen, world=world, x=58, y=0,
                                  screen_height=SCREEN_HEIGHT, screen_width=SCREEN_WIDTH, ppm=PPM,
                                    vertices=[
                                        (0, 0),
                                        (ROAD_WIDTH, 0),
                                        (ROAD_WIDTH, Y_MID - offset), # 18
                                        (0, Y_MID - offset) # 18
                                    ])
        self.boundary.append(boundary1)

        boundary3 = Quadrilateral(screen=self.screen, world=world, x=0, y=Y_MAX,
                                  screen_height=SCREEN_HEIGHT, screen_width=SCREEN_WIDTH, ppm=PPM,
                                  vertices=[
                                      (0, 0),
                                      (0, -(Y_MID - (M3.y - Y_MID))), # -11
                                      (ROAD_WIDTH, -(Y_MID - (M3.y - Y_MID))), # -11
                                      (ROAD_WIDTH, 0)
                                  ])
        self.boundary.append(boundary3)
        boundary4 = Quadrilateral(screen=self.screen, world=world, x=0.05, y=M3.y,
                                  screen_height=SCREEN_HEIGHT, screen_width=SCREEN_WIDTH, ppm=PPM, color=Color.Gray,
                                  vertices=[
                                      (0, 0),
                                      (0, -(M3.y - Y_MID)), # -7
                                      (-1, -(M3.y - Y_MID)), # -7
                                      (-1, 0)
                                  ])
        self.boundary.append(boundary4)

    def render(self):
        for triangle in self.triangle_list:
            triangle.render()
        self.circle_c2.render()
        self.circle_c1.render()
        self.wall_2.render()
        self.wall_1.render()
        for boundary in self.boundary:
            boundary.render()

        pygame.draw.line(self.screen, Color.Yellow, world_to_pixels(M1, SCREEN_HEIGHT, PPM), world_to_pixels(M2, SCREEN_HEIGHT, PPM))
        pygame.draw.line(self.screen, Color.Yellow, world_to_pixels(M2, SCREEN_HEIGHT, PPM), world_to_pixels(M4, SCREEN_HEIGHT, PPM))
        pygame.draw.line(self.screen, Color.Yellow, world_to_pixels(M4, SCREEN_HEIGHT, PPM), world_to_pixels(M3, SCREEN_HEIGHT, PPM))
        pygame.draw.line(self.screen, Color.Yellow, world_to_pixels(M3, SCREEN_HEIGHT, PPM), world_to_pixels(M1, SCREEN_HEIGHT, PPM))

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

class RaceCombined(object):

    def __init__(self, display=False, manual=False):

        # -------------------- Pygame Setup ----------------------

        self.display = display
        self.manual = manual
        self.screen = None
        if self.display:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
            pygame.display.set_caption('Environment Race Combined Right')
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
        start_pos = vec2(C1.x + RADIUS_INNER + (ROAD_WIDTH / 2), C1.y + (self.car.radius + 1))  # starting point
        self.car.body.position = start_pos
        self.car.body.angle = Util.deg_to_rad(22.5)
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
        max_angular_vel = 3.5
        angular_vel_norm = Util.min_max_normalization_m1_1(0., _min=-max_angular_vel, _max=max_angular_vel)
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
        self.gizmo()
        Util.print_fps(self.screen, self.myfont, 'FPS : ' + str('{:3.2f}').format(self.fps))

        pygame.display.flip()

    def gizmo(self):

        pygame.draw.line(self.screen, Color.Yellow, world_to_pixels(HALFWAY, SCREEN_HEIGHT, PPM),
                         world_to_pixels(HALFWAY - vec2(ROAD_WIDTH, 0), SCREEN_HEIGHT, PPM))

        if self.distance_goal < np.pi * RADIUS_INNER:

            # 2nd part : Turn right
            pygame.draw.line(self.screen, Color.Magenta, world_to_pixels(C2, SCREEN_HEIGHT, PPM),
                             world_to_pixels(S2_POINT, SCREEN_HEIGHT, PPM))

            theta = Util.angle_direct(Util.normalize(S2_POINT - C2),
                                      Util.normalize(self.car.body.position - C2))
            theta = Util.deg_to_rad(theta)
            h = vec2(-RADIUS_INNER * np.cos(theta) + C2.x,
                     -RADIUS_INNER * np.sin(theta) + C2.y)  # orthogonal projection
            pygame.draw.line(self.screen, Color.Red, world_to_pixels(C2, SCREEN_HEIGHT, PPM),
                             world_to_pixels(h, SCREEN_HEIGHT, PPM))

            tangent = Util.rotate(Util.normalize(C2 - h), 90.0)  # tangent to the circle
            pygame.draw.line(self.screen, Color.Yellow, world_to_pixels(h, SCREEN_HEIGHT, PPM),
                             world_to_pixels(h + tangent, SCREEN_HEIGHT, PPM))
        else:
            # 1st part : Turn Left
            pygame.draw.line(self.screen, Color.Magenta, world_to_pixels(C1, SCREEN_HEIGHT, PPM),
                             world_to_pixels(S1_POINT, SCREEN_HEIGHT, PPM))

            theta = Util.angle_direct(Util.normalize(S1_POINT - C1),
                                      Util.normalize(self.car.body.position - C1))
            theta = Util.deg_to_rad(theta)
            h = vec2(RADIUS_INNER * np.cos(theta) + C1.x,
                     RADIUS_INNER * np.sin(theta) + C1.y)  # orthogonal projection
            pygame.draw.line(self.screen, Color.Red, world_to_pixels(C1, SCREEN_HEIGHT, PPM),
                             world_to_pixels(h, SCREEN_HEIGHT, PPM))

            tangent = Util.rotate(Util.normalize(C1 - h), -90.0)  # tangent to the circle
            pygame.draw.line(self.screen, Color.Yellow, world_to_pixels(h, SCREEN_HEIGHT, PPM),
                             world_to_pixels(h + tangent, SCREEN_HEIGHT, PPM))

    @property
    def orientation_lane(self):
        """
            Get agent orientation in lane
        """
        pos = self.car.body.position
        if self.distance_goal < np.pi * RADIUS_INNER:

            # 2nd part : Turn right
            theta = Util.angle_direct(Util.normalize(S2_POINT - C2),
                                      Util.normalize(pos - C2))
            theta = Util.deg_to_rad(theta)
            h = vec2(-RADIUS_INNER * np.cos(theta) + C2.x,
                     -RADIUS_INNER * np.sin(theta) + C2.y)  # orthogonal projection
            tangent = Util.rotate(Util.normalize(C2 - h), 90.0)  # tangent to the circle
        else:
            # 1st part : Turn Left
            theta = Util.angle_direct(Util.normalize(S1_POINT - C1),
                                      Util.normalize(self.car.body.position - C1))
            theta = Util.deg_to_rad(theta)
            h = vec2(RADIUS_INNER * np.cos(theta) + C1.x,
                     RADIUS_INNER * np.sin(theta) + C1.y)  # orthogonal projection
            tangent = Util.rotate(Util.normalize(C1 - h), -90.0)  # tangent to the circle

        forward = Util.normalize(self.car.body.GetWorldVector((0, 1)))
        orientation = Util.angle_indirect(forward, tangent) / 180.0
        return orientation

    @property
    def distance_lane(self):
        pos = self.car.body.position
        if self.distance_goal < np.pi * RADIUS_INNER:
            # 2nd part : Turn right
            d = RADIUS_OUTTER - Util.distance(pos, C2)
            return d
        else:
            # 1st part : Turn Left
            d = Util.distance(pos, C1) - RADIUS_INNER
            return d

    @staticmethod
    def normalize_distance(val, _min, _max):
        return Util.min_max_normalization_m1_1(val, _min=_min, _max=_max)

    def normalize_sensors_value(self, val):
        return Util.min_max_normalization_m1_1(val, _min=0.0, _max=self.car.raycast_length)

    @property
    def distance_goal(self):
        pos = self.car.body.position

        if (pos.y < HALFWAY.y and pos.x < HALFWAY.x) or (pos.x < HALFWAY.x - ROAD_WIDTH):
            # 2nd part : Turn Right
            theta = Util.angle_direct(Util.normalize(pos - C2), Util.normalize(S2_POINT - C2))
        else:
            # 1st part : Turn Left
            theta = Util.angle_direct(Util.normalize(S1_POINT - C1), Util.normalize(pos - C1))

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
        if M1.x <= pos.x <= M2.x and M2.y <= pos.y <= M3.y:
            return True
        else:
            return False

    def reward_function(self):
        """
            Core function of the agent for training
        """
        flag_col = False
        if self.car.collision:
            flag_col = True

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
        reward = flag_col * Reward.COLLISION + reward_sensors + potential_shaping

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