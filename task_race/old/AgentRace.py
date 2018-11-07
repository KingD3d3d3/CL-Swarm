import pygame
from enum import Enum
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (vec2, pi)
from pygame.locals import *
from collections import deque
import numpy as np
import time
try:
    # Running in PyCharm
    import res.colors as Color
    from AI.DQN import DQN
    from Setup import *
    from Util import world_to_pixels
    import Util
    from res.print_colors import print_color
    import Global
    from task_race.RayCastCallback import RayCastCallback
except NameError as err:
    print(err, "--> our error message")
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running from command line -> Import libraries as package")
    from ..res import colors as Color
    from AI.DQN import DQN
    from ..Setup import *
    from ..Util import world_to_pixels
    from .. import Util
    from .. import Global
    from ..res.print_colors import print_color

# Agent brain config
race_hyperparams = {
    'layers': (32, 32),
    'mem_capacity': 100000,
    'batch_size': 32,
    'eps_start': 1.,
    'eps_end': 0.05,
    'eps_test': 0.,
    'exploration_steps': 10000,
    'gamma': 0.99,
    'lr': 0.001,
    'update_target_steps': 1000,
    'use_double_dqn': True,
    'use_prioritized_experience_replay': False
}

# Global variables
max_x = SCREEN_WIDTH / PPM  # 36
max_y = SCREEN_HEIGHT / PPM  # 36
r = (SCREEN_WIDTH / PPM / 2) - 7 # distance between the inner circle and outer circle
c = vec2(max_x / 2, max_y / 2)  # center of the race

# Agent's possible actions
class Action(Enum):
    FORWARD_LEFT = 0
    FORWARD = 1
    FORWARD_RIGHT = 2
    BRAKE_LEFT = 3
    BRAKE = 4
    BRAKE_RIGHT = 5

# Rewards Mechanism
class Reward:
    # GOAL_REACHED = 100.
    COLLISION = -100.

    @classmethod
    def reward_sensor(cls, x, max_length=3.):
        """
            Shaping reward on sensors
        """
        # reward
        ok = 0.
        little = -1 #-0.1 # tiny punishment
        worst = cls.COLLISION # -10. # higher punishment # 5

        # distance sensor
        min_length = 0.0
        middle = (max_length + min_length) / 2

        if middle <= x <= max_length:

            # Linear
            b = (max_length, ok)
            a = (middle, little)
            m = (b[1] - a[1]) / (b[0] - a[0])
            p = a[1] - m * a[0]
            y = m * x + p

            # y = round(y, 2)
            return y
        else:

            # Linear
            b = (middle, little)
            a = (0, worst)
            m = (b[1] - a[1]) / (b[0] - a[0])
            p = a[1] - m * a[0]
            y = m * x + p

            # y = round(y, 2)
            return y


class AgentRace(object):
    def __init__(self, screen=None, world=None, x=0, y=0, angle=0, radius=1.25, solved_score=100000, env='RaceCirce',
                 seed=None):
        self.screen = screen
        self.world = world
        self.id = 0

        self.car_radius = radius
        self.body = self.world.CreateDynamicBody(
            position=(x, y), userData=self, angle=angle)
        self.fixture = self.body.CreateCircleFixture(
            radius=radius, density=1, friction=0, restitution=0)
        self.color = Color.Red

        # Input size
        self.input_size = 7

        # Specifications
        self.rotation_speed = 1.0 * np.pi  # rad/s
        self.max_forward_speed = 100 # 50
        self.max_drive_force = 300 # can reach maximum speed in 2.833 s (0-100) # 180
        self.turn_torque = 80  # achieved max_angular_velocity of 3.12 rad/s in 2.5s

        # Proximity Sensors
        self.raycast_length = 3.0
        self.raycastSafeColor = Color.Green
        self.raycastDangerColor = Color.Red
        front_middle = vec2(0, 1)
        front_right = vec2(np.sqrt(2) / 2, np.sqrt(2) / 2)
        front_left = vec2(-np.sqrt(2) / 2, np.sqrt(2) / 2)
        self.raycast_vectors = (
            front_left, front_middle, front_right)  # Store raycast direction vector for each raycast
        self.num_sensors = len(self.raycast_vectors)
        self.sensors = np.ones(self.num_sensors) * self.raycast_length  # store the value of each sensors

        # ------------------ Variables to set at each simulation --------------------

        self.brain = None
        self.episodes = 0  # number of episodes during current simulation
        self.scores = deque(maxlen=100)  # keep total scores of last 100 episodes
        self.average_score = 0

        self.timesteps_list = deque(maxlen=20)  # keep timesteps of last 100 episodes
        self.average_timestep = 0

        self.score = 0  # keep score of 1 episode
        self.timesteps = 0  # count number of timesteps during 1 episode
        self.tot_timesteps = 0  # total number of timesteps of all episodes passed during 1 simulation

        # States
        self.done = False
        self.distance = None
        self.orientation = None
        self.distance_road = None
        self.action = None
        self.state = None
        self.goal_reached = None
        self.collision = None
        self.last_distance = None
        self.speed = None

        self.solved_score = solved_score  # average score agent needs to reach to consider the problem solved

        self.problem_done = False
        self.problem_solved = False

        self.env_name = env

        self.prev_shaping = None
        self.timeout = 999
        self.seed = seed

        self.best_average = self.timeout

    def setup(self, training):
        # Create agent's brain
        self.brain = DQN(input_size=self.input_size, action_size=len(list(Action)), id=self.id,
                         training=training, random_agent=False, **race_hyperparams, seed=self.seed)

        # Reset environment
        self.reset()

    def reset(self):
        """
            Reset agent's states at the beginning or when episode is done
        """
        self.done = False
        self.prev_shaping = None

        # Initial position
        start_pos = vec2(c.x, c.y + r + (7 / 2))  # starting point
        self.body.position = start_pos

        # Initial orientation
        if self.env_name == 'RaceCircle':
            self.body.angle = Util.deg_to_rad(90)
        elif self.env_name == 'RaceCircle_v2':
            self.body.angle = Util.deg_to_rad(-90)

        # Flags
        self.goal_reached = False
        self.collision = False

        # Initial Observation
        observation = []

        # self.distance = self.distance_goal()  # Distance to goal
        # self.last_distance = self.distance
        # max_dist = 11 * (2 * np.pi)
        # distance_norm = self.normalize_distance(val=self.distance, min=0.0, max=max_dist)
        # observation.append(distance_norm)

        # Orientation in lane
        self.orientation = self.orientation_lane()
        observation.append(self.orientation)

        # Distance in lane
        self.distance_road = self.distance_lane()
        distance_road_norm = self.normalize_distance(val=self.distance_road, min=0.0, max=7 - (self.car_radius * 2))
        observation.append(distance_road_norm)

        # Speed of the car
        self.speed = 0.
        speed_norm = Util.min_max_normalization_m1_1(self.speed, _min=0., _max=self.max_forward_speed)
        observation.append(speed_norm)

        # Angular velocity of the car
        angular_vel = 0.
        angular_vel_norm = Util.min_max_normalization_m1_1(angular_vel, _min=-3.5, _max=3.5)
        observation.append(angular_vel_norm)

        # Sensor's value
        self.read_sensors()
        for sensor in self.sensors:
            normed_sensor = self.normalize_sensors_value(sensor)  # Normalize sensor's value
            observation.append(normed_sensor)

        # Kill velocity
        forward_vec = self.body.GetWorldVector((0, 1))
        self.body.linearVelocity = forward_vec * 0.
        self.body.angularVelocity = 0

        # Initial State
        self.state = self.brain.preprocess(observation)

        # Go 1 step further to apply the environment reset
        self.world.Step(PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS)
        self.world.ClearForces()

    @property
    def lateral_velocity(self):
        right_normal = self.body.GetWorldVector(vec2(1, 0))
        return right_normal.dot(self.body.linearVelocity) * right_normal

    @property
    def forward_velocity(self):
        current_normal = self.body.GetWorldVector(vec2(0, 1))
        return current_normal.dot(self.body.linearVelocity) * current_normal

    def kill_forward(self):
        """
            Annihilate the forward velocity
        """
        impulse = -self.forward_velocity * self.body.mass * (2. / 3.)
        self.body.ApplyLinearImpulse(impulse, self.body.worldCenter, True)  # kill forward

    def update_friction(self):
        impulse = -self.lateral_velocity * self.body.mass

        self.body.ApplyLinearImpulse(impulse, self.body.worldCenter, True)  # kill lateral velocity
        self.body.ApplyAngularImpulse(0.1 * self.body.inertia * - self.body.angularVelocity,
                                      True)  # kill angular velocity (old: 0.8)

        # Stop the forever roll
        current_forward_normal = self.forward_velocity
        current_forward_speed = current_forward_normal.Normalize()
        drag_force_magnitude = -2 * current_forward_speed  # -10
        self.body.ApplyForce(drag_force_magnitude * current_forward_normal, self.body.worldCenter, True)

    def remain_static(self):
        self.update_friction()
        speed = 0

        forward_vec = self.body.GetWorldVector((0, 1))
        self.body.linearVelocity = forward_vec * speed

    def update_drive_manual(self):

        key = pygame.key.get_pressed()
        if key[K_SPACE]:  # Brake
            desired_speed = 0.
        else:  # Forward
            desired_speed = self.max_forward_speed

        # find the current speed in the forward direction
        current_forward_normal = self.body.GetWorldVector((0, 1))
        current_speed = self.forward_velocity.dot(current_forward_normal)
        # print(current_speed)

        # apply necessary force
        if desired_speed > current_speed:
            force = self.max_drive_force
        elif desired_speed < current_speed:
            force = -self.max_drive_force
        else:
            return

        # Handle negative speed case
        if desired_speed == 0. and current_speed <= 0.:
            self.kill_forward()
            return

        self.body.ApplyForce(force * current_forward_normal, self.body.worldCenter, True)

    def update_turn_manual(self):
        key = pygame.key.get_pressed()
        if key[K_LEFT]:
            # self.body.angularVelocity =  self.rotation_speed
            desired_torque = self.turn_torque
        elif key[K_RIGHT]:
            # self.body.angularVelocity = -self.rotation_speed
            desired_torque = -self.turn_torque
        else:
            return

        self.body.ApplyTorque(desired_torque, True)
        # print(self.body.angularVelocity)

    def update_drive(self, action):

        if action == Action.FORWARD or action == Action.FORWARD_LEFT or action == Action.FORWARD_RIGHT:  # Forward
            desired_speed = self.max_forward_speed
        elif action == Action.BRAKE or action == Action.BRAKE_LEFT or action == Action.BRAKE_RIGHT:  # Brake
            desired_speed = 0.
        else:
            return

        # find the current speed in the forward direction
        current_forward_normal = self.body.GetWorldVector((0, 1))
        current_speed = self.forward_velocity.dot(current_forward_normal)

        # apply necessary force
        if desired_speed > current_speed:
            force = self.max_drive_force
        elif desired_speed < current_speed:
            force = -self.max_drive_force
        else:
            return

        # Handle negative speed case
        if desired_speed == 0. and current_speed <= 0.:
            self.kill_forward()
            return

        self.body.ApplyForce(force * current_forward_normal, self.body.worldCenter, True)

    def update_turn(self, action):
        if action == Action.BRAKE_LEFT or action == Action.FORWARD_LEFT:  # Turn left
            # self.body.angularVelocity = self.rotation_speed
            desired_torque = self.turn_torque
        elif action == Action.BRAKE_RIGHT or action == Action.FORWARD_RIGHT:  # Turn right
            # self.body.angularVelocity = -self.rotation_speed
            desired_torque = -self.turn_torque
        else:
            return

        self.body.ApplyTorque(desired_torque, True)
        # print(self.body.angularVelocity)

    def read_sensors(self):

        # Read raycasts value
        for i in range(self.num_sensors):
            raycast = RayCastCallback()
            v = self.body.GetWorldVector(self.raycast_vectors[i])
            p1 = self.body.worldCenter + v * self.car_radius
            p2 = p1 + v * self.raycast_length
            self.world.RayCast(raycast, p1, p2)
            if raycast.hit:
                dist = (p1 - raycast.point).length  # distance to the hit point
                self.sensors[i] = dist
            else:
                self.sensors[i] = self.raycast_length  # default value is raycastLength
                # print('sensors {}: {}'.format(i, self.sensors[i]))

    def draw(self):
        position = self.body.transform * self.fixture.shape.pos * PPM
        position = (position[0], SCREEN_HEIGHT - position[1])
        pygame.draw.circle(self.screen, self.color, [int(x) for x in position], int(self.car_radius * PPM))

        current_forward_normal = self.body.GetWorldVector((0, 1))
        pygame.draw.line(self.screen, Color.White, world_to_pixels(self.body.worldCenter),
                         world_to_pixels(self.body.worldCenter + current_forward_normal * self.car_radius))

        # Draw raycasts
        for i in range(self.num_sensors):
            v = self.body.GetWorldVector(self.raycast_vectors[i])
            p1 = self.body.worldCenter + v * self.car_radius
            p2 = p1 + v * self.sensors[i]  # self.raycast_length
            if self.sensors[i] <= self.raycast_length / 2:
                ray_color = self.raycastDangerColor
            else:
                ray_color = self.raycastSafeColor
            pygame.draw.line(self.screen, ray_color, world_to_pixels(p1), world_to_pixels(p2))

        # DEBUG gizmo --------------------------------------------------------------------------------------------------

        # Starting point
        s = None
        if self.env_name == 'RaceCircle':
            s = vec2(c.x + r, c.y)
            pygame.draw.line(self.screen, Color.Magenta, world_to_pixels(c), world_to_pixels(s + vec2(7, 0)))
        elif self.env_name == 'RaceCircle_v2':
            s = vec2(c.x - r, c.y)
            pygame.draw.line(self.screen, Color.Magenta, world_to_pixels(c), world_to_pixels(s + vec2(-7, 0)))

        # Orthogonal projection to the circle
        ph = None
        if self.env_name == 'RaceCircle':
            theta = Util.angle_direct(Util.normalize(s - c), Util.normalize(self.body.position - c))
            theta = Util.deg_to_rad(theta)
            ph = vec2(r * np.cos(theta) + c.x, r * np.sin(theta) + c.y)
        elif self.env_name == 'RaceCircle_v2':
            theta = Util.angle_direct(Util.normalize(s - c), Util.normalize(self.body.position - c))
            theta = Util.deg_to_rad(theta)
            ph = vec2(-r * np.cos(theta) + c.x, -r * np.sin(theta) + c.y)
        pygame.draw.line(self.screen, Color.Red, world_to_pixels(c), world_to_pixels(ph))

        # Tangent to the circle
        tangent = None
        if self.env_name == 'RaceCircle':
            tangent = Util.rotate(Util.normalize(c - ph), -90.0)
        elif self.env_name == 'RaceCircle_v2':
            tangent = Util.rotate(Util.normalize(c - ph), 90.0)
        pygame.draw.line(self.screen, Color.Yellow, world_to_pixels(ph), world_to_pixels(ph + tangent))
        # --------------------------------------------------------------------------------------------------------------

    def check_goal_reached(self):
        if self.distance_goal() == 0. or self.inside_goal():
            self.goal_reached = True
            self.done = True
        else:
            pass

    def distance_goal(self):
        # Starting point
        s = None
        if self.env_name == 'RaceCircle':
            s = vec2(c.x + r, c.y)
        elif self.env_name == 'RaceCircle_v2':
            s = vec2(c.x - r, c.y)

        # Orthogonal projection to the circle
        theta = None
        if self.env_name == 'RaceCircle':
            theta = Util.angle_direct(Util.normalize(s - c), Util.normalize(self.body.position - c))
        elif self.env_name == 'RaceCircle_v2':
            theta = Util.angle_indirect(Util.normalize(s - c), Util.normalize(self.body.position - c))

        if theta < 0:
            theta = 180.0 + (180.0 + theta)

        theta = Util.deg_to_rad(theta)
        phi = 2 * np.pi - theta
        d2g = r * phi

        # Boundary reaching goal
        if self.distance and np.abs(d2g - self.distance) >= 10.0:
            return 0.

        # res = round(np.abs(d2g), 2)
        res = np.abs(d2g)

        return res

    def inside_goal(self):

        if self.env_name == 'RaceCircle':

            M1 = vec2(np.cos(3 * (np.pi / 12)) * 11 + 18, 18)
            M2 = vec2(36, 18)
            M3 = vec2(36, np.sin(3 * (np.pi / 12)) * 11 + 18)
            M4 = vec2(np.cos(3 * (np.pi / 12)) * 11 + 18, np.sin(3 * (np.pi / 12)) * 11 + 18)

            pos = self.body.position
            if M1.x <= pos.x <= M2.x and M2.y <= pos.y <= M3.y:
                return True
            else:
                return False

        elif self.env_name == 'RaceCircle_v2':
            M1 = vec2(-np.cos(3 * (np.pi / 12)) * 11 + 18, 18)
            M2 = vec2(0, 18)
            M3 = vec2(0, np.sin(3 * (np.pi / 12)) * 11 + 18)
            M4 = vec2(-np.cos(3 * (np.pi / 12)) * 11 + 18, np.sin(3 * (np.pi / 12)) * 11 + 18)

            pos = self.body.position
            if M2.x <= pos.x <= M1.x and M2.y <= pos.y <= M3.y:
                return True
            else:
                return False

        else:
            return None

    def orientation_lane(self):
        """
            Get agent orientation in lane
        """
        # Starting point
        s = None
        if self.env_name == 'RaceCircle':
            s = vec2(c.x + r, c.y)
        elif self.env_name == 'RaceCircle_v2':
            s = vec2(c.x - r, c.y)

        # Orthogonal projection to the circle
        ph = None
        if self.env_name == 'RaceCircle':
            theta = Util.angle_direct(Util.normalize(s - c), Util.normalize(self.body.position - c))
            theta = Util.deg_to_rad(theta)
            ph = vec2(r * np.cos(theta) + c.x, r * np.sin(theta) + c.y)
        elif self.env_name == 'RaceCircle_v2':
            theta = Util.angle_direct(Util.normalize(s - c), Util.normalize(self.body.position - c))
            theta = Util.deg_to_rad(theta)
            ph = vec2(-r * np.cos(theta) + c.x, -r * np.sin(theta) + c.y)

        # Tangent to the circle
        tangent = None
        if self.env_name == 'RaceCircle':
            tangent = Util.rotate(Util.normalize(c - ph), -90.0)
        elif self.env_name == 'RaceCircle_v2':
            tangent = Util.rotate(Util.normalize(c - ph), 90.0)

        forward = Util.normalize(self.body.GetWorldVector((0, 1)))
        orientation = Util.angle_indirect(forward, tangent) / 180.0
        # orientation = round(orientation, 2)  # only 3 decimals

        return orientation

    def distance_lane(self):
        # d = round(np.abs(Util.distance(self.body.position, c) - (r + self.car_radius)), 2)
        d = np.abs(Util.distance(self.body.position, c) - (r + self.car_radius))
        return d

    def normalize_sensors_value(self, val):
        # return round(Util.min_max_normalization_m1_1(val, _min=0.0, _max=self.raycast_length), 2)
        return Util.min_max_normalization_m1_1(val, _min=0.0, _max=self.raycast_length)

    @staticmethod
    def normalize_distance(val, min, max):
        # return round(Util.min_max_normalization_m1_1(val, _min=min, _max=max), 2)
        return Util.min_max_normalization_m1_1(val, _min=min, _max=max)

    def reward_function(self):

        flag_col = False
        if self.collision:
            flag_col = True

        # # Goal Reached
        # flag_gr = False
        # if self.goal_reached:
        #     flag_gr = True

        # Process sensor's value
        r = np.fromiter((Reward.reward_sensor(s, max_length=self.raycast_length) for s in self.sensors),
                        self.sensors.dtype,
                        count=len(self.sensors))  # Apply sensor reward function to all sensors value
        reward_sensors = np.amin(r)  # take the min value

        # Flag that indicates the direction in the road
        if (0.5 < self.orientation <= 1) or (-1 <= self.orientation < -0.5):
            direction_road = -1. # punish speed when reverse facing the road
        else:
            direction_road = 1.

        # Reward shaping based on speed of agent
        potential_shaping = 0
        shaping = direction_road * 10. * self.speed # coeff of 10 to make it 10^0 (1 meter) order and not millimeter
        if self.prev_shaping is not None:
            """
                The faster the agent is -> higher the reward
                Get slower -> penalty
                Keep constant speed -> no reward
            """
            potential_shaping = shaping - self.prev_shaping
            # print(potential_shaping)
        self.prev_shaping = shaping

        # Overall reward
        reward = flag_col * Reward.COLLISION + \
                 reward_sensors + potential_shaping

        return reward

    def before_step(self):
        """
            Processing before the environment takes next step
        """
        # Select action
        self.action = self.brain.select_action(self.state)
        self.update_friction()
        self.update_drive(Action(self.action))
        self.update_turn(Action(self.action))
        # self.update_drive_manual()
        # self.update_turn_manual() # 2.25s to achieve max rotation speed of 3.129 rad/s

    def after_step(self):
        """
            Processing after the environment took step
        """
        # ------------------ Observation from the environment ----------------------------

        observation = []

        # # Distance to goal
        self.distance = self.distance_goal()
        self.check_goal_reached()
        # max_dist = 11 * (2 * np.pi)
        # distance_norm = self.normalize_distance(val=self.distance, min=0.0, max=max_dist)
        # observation.append(distance_norm)

        # Orientation in lane
        self.orientation = self.orientation_lane()
        observation.append(self.orientation)

        # Distance in lane
        self.distance_road = self.distance_lane()
        distance_road_norm = self.normalize_distance(val=self.distance_road, min=0.0, max=7 - (self.car_radius * 2))
        observation.append(distance_road_norm)

        # Speed of the car
        current_forward_normal = self.body.GetWorldVector((0, 1))
        self.speed = self.forward_velocity.dot(current_forward_normal)
        # print(speed)
        speed_norm = Util.min_max_normalization_m1_1(self.speed, _min=0., _max=self.max_forward_speed)
        observation.append(speed_norm)

        # Angular velocity of the car
        angular_vel = self.body.angularVelocity
        angular_vel_norm = Util.min_max_normalization_m1_1(angular_vel, _min=-3.5, _max=3.5)
        observation.append(angular_vel_norm)

        # Sensor's value
        self.read_sensors()
        for sensor in self.sensors:
            normed_sensor = self.normalize_sensors_value(sensor)  # Normalize sensor's value
            observation.append(normed_sensor)

        # State
        observation = np.asarray(observation)

        # Time limits terminal condition
        if self.timesteps >= self.timeout:
            self.done = True
        # --------------------------------------------------------------------------------

        # Reward from the environment
        reward = self.reward_function()

        # Record experience
        next_state = self.brain.preprocess(observation)
        self.brain.record((self.state, self.action, reward, next_state, self.done))
        self.state = next_state
        self.brain.train()

        # Update variables old states
        self.last_distance = self.distance

        self.score += reward
        self.timesteps += 1



        # --------------------------- END OF EPISODE -------------------------------------------------------------------

        if self.done:  # start new episode
            self.episodes += 1

            if not self.goal_reached:
                self.timesteps = self.timeout + 1

            self.scores.append(self.score)
            self.timesteps_list.append(self.timesteps)

            # Calculate average over the last episodes
            self.average_score = sum(self.scores) / len(self.scores)
            self.average_timestep = sum(self.timesteps_list) / len(self.timesteps_list)
            self.tot_timesteps += self.timesteps  # increment total number of timesteps of all episodes

            # Calculate best average
            if self.average_timestep < self.best_average and len(self.timesteps_list) >= 20:
                self.best_average = self.average_timestep

            # Periodically print current average of reward
            if self.episodes % 1 == 0:
                print(
                    "episode: {:5.0f}, timesteps: {:3.0f}, tot_timestep: {:8.0f}, score: {:3.0f}, "
                    "average_score: {:3.2f}, average_tmstp: {:3.2f}, best_avg: {:3.2f}"
                    .format(self.episodes, self.timesteps, self.tot_timesteps,
                            self.score, self.average_score, self.average_timestep, self.best_average))

            # reset for next episode
            self.score = 0
            self.timesteps = 0
            self.reset()


            # Problem solved
            if self.average_timestep <= self.solved_score and len(self.timesteps_list) >= 20: # last 20 run
                print(
                    "agent: {:4.0f}, *** Solved after {} episodes *** reached solved score timestep: {}".format(self.id,
                                                                                                                self.episodes,
                                                                                                                self.solved_score))
                self.problem_done = True
                self.problem_solved = True
                return
