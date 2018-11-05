import random
import pygame
from enum import Enum
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (vec2, pi)
from pygame.locals import *
import os
from collections import deque
import errno
import numpy as np

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
    'exploration_steps': 1000,
    'gamma': 0.99,
    'lr': 0.001,
    'update_target_steps': 1000,
    'use_double_dqn': True,
    'use_prioritized_experience_replay': False
}

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
    LIVING_PENALTY = -0.1
    GOAL_REACHED = 100.
    COLLISION = -100.
    GETTING_CLOSER = 0.01

    @classmethod
    def reward_sensor(cls, x):
        """
            Shaping reward on sensors
        """
        if 2. <= x <= 4.:
            return 0.
        else:
            # Linear
            y = 5. * x - 10.
            return y


class AgentRace(object):
    def __init__(self, screen=None, world=None, x=0, y=0, angle=0, radius=1.25, solved_score=100000, env='RaceCirce'):
        self.screen = screen
        self.world = world

        self.id = 0
        self.car_radius = radius
        self.body = self.world.CreateDynamicBody(
            position=(x, y), userData=self, angle=angle)

        self.fixture = self.body.CreateCircleFixture(
            radius=radius, density=1, friction=0, restitution=0)

        self.initial_color = Color.Magenta
        self.color = Color.Red

        # Specifications
        # self.speed = 5  # m/s
        self.rotation_speed = 1.0 * np.pi  # rad/s

        self.max_forward_speed = 30
        self.max_backward_speed = -5
        self.max_drive_force = 100

        # Distance travelled by agent at each timestep
        # self.delta_dist = self.speed * (1. / TARGET_FPS)

        # Proximity Sensors
        self.raycast_length = 4.0
        self.raycastSafeColor = Color.Green
        self.raycastDangerColor = Color.Red
        front_middle = vec2(0, 1)
        front_right = vec2(np.sqrt(2) / 2, np.sqrt(2) / 2)
        front_left = vec2(-np.sqrt(2) / 2, np.sqrt(2) / 2)
        self.raycast_vectors = (
            front_left, front_middle, front_right)  # Store raycast direction vector for each raycast
        self.num_sensors = len(self.raycast_vectors)
        self.sensors = np.ones(self.num_sensors) * self.raycast_length  # store the value of each sensors
        print("num sensors : {}".format(self.num_sensors))

        # ------------------ Variables to set at each simulation --------------------

        self.brain = None
        self.episodes = 0  # number of episodes during current simulation
        self.scores = deque(maxlen=100)  # keep total scores of last 100 episodes
        self.average_score = 0
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

        # Input size
        self.input_size = 7

        self.solved_score = solved_score # average score agent needs to reach to consider the problem solved

        self.problem_done = False
        self.problem_solved = False

        self.env = env


    def setup(self, training):
        # Create agent's brain
        self.brain = DQN(input_size=self.input_size, action_size=len(list(Action)), id=self.id,
                         training=training, random_agent=False, **race_hyperparams)

        self.reset()
        pass


    def reset(self):
        """
            Reset agent's states when episode is done
        """
        self.done = False

        # Initial position
        max_x = SCREEN_WIDTH / PPM  # 36
        max_y = SCREEN_HEIGHT / PPM  # 36
        r = (SCREEN_WIDTH / PPM / 2) - 7
        c = vec2(max_x / 2, max_y / 2)  # centre
        start_pos = vec2(c.x, c.y + r + (7 / 2))  # starting point
        self.body.position = start_pos

        # Initial orientation
        if self.env == 'RaceCircle':
            self.body.angle = Util.deg_to_rad(90)
        elif self.env == 'RaceCircle_v2':
            self.body.angle = Util.deg_to_rad(-90)

        # Flags
        self.goal_reached = False
        self.collision = False

        # Initial Observation
        observation = []

        self.distance = self.distance_goal()  # Distance to goal
        self.last_distance = self.distance
        max_dist = 11 * (2 * np.pi)
        distance_norm = self.normalize_distance(val=self.distance, min=0.0, max=max_dist)
        observation.append(distance_norm)

        self.orientation = self.orientation_lane()  # Orientation in lane
        observation.append(self.orientation)

        self.distance_road = self.distance_lane()  # Distance in lane
        distance_road_norm = self.normalize_distance(val=self.distance_road, min=0.0, max=7 - (self.car_radius * 2))
        observation.append(distance_road_norm)

        # Sensor's value
        self.read_sensors()
        for sensor in self.sensors:
            normed_sensor = self.normalize_sensors_value(sensor)  # Normalize sensor's value
            observation.append(normed_sensor)

        # Speed of the car
        speed = 0.
        speed_norm = Util.min_max_normalization_m1_1(speed, _min=0., _max=self.max_forward_speed)
        observation.append(speed_norm)

        # Kill velocity
        forward_vec = self.body.GetWorldVector((0, 1))
        self.body.linearVelocity = forward_vec * 0.
        self.body.angularVelocity = 0

        # Initial State
        self.state = self.brain.preprocess(observation)

    @property
    def lateral_velocity(self):
        right_normal = self.body.GetWorldVector(vec2(1, 0))
        return right_normal.dot(self.body.linearVelocity) * right_normal

    @property
    def forward_velocity(self):
        current_normal = self.body.GetWorldVector(vec2(0, 1))
        return current_normal.dot(self.body.linearVelocity) * current_normal

    # def update_friction(self):
    #     impulse = self.body.mass * -self.lateral_velocity
    #     self.body.ApplyLinearImpulse(impulse, self.body.worldCenter, True)  # kill lateral velocity
    #     self.body.ApplyAngularImpulse(0.8 * self.body.inertia * - self.body.angularVelocity,
    #                                   True)  # kill angular velocity #0.1 #0.3
    #
    #     # Stop the forever roll
    #     current_forward_normal = self.forward_velocity
    #     current_forward_speed = current_forward_normal.Normalize()
    #     drag_force_magnitude = -50 * current_forward_speed  # -10
    #     self.body.ApplyForce(drag_force_magnitude * current_forward_normal, self.body.worldCenter, True)

    def update_friction(self):
        impulse = -self.lateral_velocity * self.body.mass

        self.body.ApplyLinearImpulse(impulse, self.body.worldCenter, True)  # kill lateral velocity
        self.body.ApplyAngularImpulse(0.1 * self.body.inertia * - self.body.angularVelocity, True)  # kill angular velocity (old: 0.8)

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

    def update_manual_drive_force(self):

        key = pygame.key.get_pressed()
        if key[K_UP]:  # Forward
            desired_speed = self.max_forward_speed
        elif key[K_SPACE]: # Break
            desired_speed = 0.
        else:
            return

        # find the current speed in the forward direction
        current_forward_normal = self.body.GetWorldVector((0, 1))
        current_speed = self.forward_velocity.dot(current_forward_normal)
        print('current_speed', current_speed)

        # apply necessary force
        force = 0.0
        if desired_speed > current_speed:
            force = self.max_drive_force
        elif desired_speed < current_speed:
            force = -self.max_drive_force
        else:
            return

        self.body.ApplyForce(force * current_forward_normal, self.body.worldCenter, True)

    def update_turn_manual(self):
        key = pygame.key.get_pressed()
        if key[K_LEFT]:
            self.body.angularVelocity = self.rotation_speed

        elif key[K_RIGHT]:
            self.body.angularVelocity = -self.rotation_speed

        else:
            return

    def update_drive_force(self, action):

        if action == Action.FORWARD or action == Action.FORWARD_LEFT or action == Action.FORWARD_RIGHT: # Forward
            desired_speed = self.max_forward_speed
        elif action == Action.BRAKE or action == Action.BRAKE_LEFT or action == Action.BRAKE_RIGHT: # Brake
            desired_speed = 0.
        else:
            return

        # find the current speed in the forward direction
        current_forward_normal = self.body.GetWorldVector((0, 1))
        current_speed = self.forward_velocity.dot(current_forward_normal)
        # print('current_speed', current_speed)

        # apply necessary force
        if desired_speed > current_speed:
            force = self.max_drive_force
        elif desired_speed < current_speed:
            force = -self.max_drive_force
        else:
            return

        self.body.ApplyForce(force * current_forward_normal, self.body.worldCenter, True)

    def update_turn(self, action):
        if action == Action.BRAKE_LEFT or action == Action.FORWARD_LEFT: # Turn left
            self.body.angularVelocity = self.rotation_speed
        elif action == Action.BRAKE_RIGHT or action == Action.FORWARD_RIGHT: # Turn right
            self.body.angularVelocity = -self.rotation_speed
        else:
            return

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
        max_x = SCREEN_WIDTH / PPM  # 36
        max_y = SCREEN_HEIGHT / PPM  # 36
        r = (SCREEN_WIDTH / PPM / 2) - 7
        c = vec2(max_x / 2, max_y / 2)  # centre

        # Starting point
        s = vec2(0, 0)
        if self.env == 'RaceCircle':
            s = vec2(c.x + r, c.y)
            pygame.draw.line(self.screen, Color.Magenta, world_to_pixels(c), world_to_pixels(s + vec2(7, 0)))
        elif self.env == 'RaceCircle_v2':
            s = vec2(c.x - r, c.y)
            pygame.draw.line(self.screen, Color.Magenta, world_to_pixels(c), world_to_pixels(s + vec2(-7, 0)))

        # Orthogonal projection to the circle
        ph = vec2(0, 0)
        if self.env == 'RaceCircle':
            theta = Util.angle_v2(Util.normalize(s - c), Util.normalize(self.body.position - c))
            theta = Util.deg_to_rad(theta)
            ph = vec2(r * np.cos(theta) + c.x, r * np.sin(theta) + c.y)
        elif self.env == 'RaceCircle_v2':
            theta = Util.angle_v2(Util.normalize(s - c), Util.normalize(self.body.position - c))
            theta = Util.deg_to_rad(theta)
            ph = vec2(-r * np.cos(theta) + c.x, -r * np.sin(theta) + c.y)
        pygame.draw.line(self.screen, Color.Red, world_to_pixels(c), world_to_pixels(ph))

        # Tangent to the circle
        tangent = Util.rotate(Util.normalize(c - ph), -90.0)
        if self.env == 'RaceCircle':
            tangent = Util.rotate(Util.normalize(c - ph), -90.0)
        elif self.env == 'RaceCircle_v2':
            tangent = Util.rotate(Util.normalize(c - ph), 90.0)
        pygame.draw.line(self.screen, Color.Yellow, world_to_pixels(ph), world_to_pixels(ph + tangent))
        # --------------------------------------------------------------------------------------------------------------

    def check_goal_reached(self):
        if self.distance_goal() == 0. or self.inside_goal():
            self.goal_reached = True
            self.done = True
            print('reached goal')
        else:
            pass

    def distance_goal(self):
        max_x = SCREEN_WIDTH / PPM  # 36
        max_y = SCREEN_HEIGHT / PPM  # 36
        r = (SCREEN_WIDTH / PPM / 2) - 7  #
        c = vec2(max_x / 2, max_y / 2)  # centre

        # Starting point
        s = vec2(0, 0)
        if self.env == 'RaceCircle':
            s = vec2(c.x + r, c.y)
        elif self.env == 'RaceCircle_v2':
            s = vec2(c.x - r, c.y)

        # Orthogonal projection to the circle
        theta = 0
        if self.env == 'RaceCircle':
            theta = Util.angle_v2(Util.normalize(s - c), Util.normalize(self.body.position - c))
        elif self.env == 'RaceCircle_v2':
            theta = Util.angle(Util.normalize(s - c), Util.normalize(self.body.position - c))

        if theta < 0:
            theta = 180.0 + (180.0 + theta)

        theta = Util.deg_to_rad(theta)
        phi = 2 * np.pi - theta
        d2g = r * phi

        # Boundary reaching goal
        if self.distance and np.abs(d2g - self.distance) >= 10.0:
            return 0.

        return round(np.abs(d2g), 2)

    def inside_goal(self):

        M1 = 0
        M2 = 0
        M3 = 0
        if self.env == 'RaceCircle':

            M1 = vec2(np.cos(3 * (np.pi / 12)) * 11 + 18, 18)
            M2 = vec2(36, 18)
            M3 = vec2(36, np.sin(3 * (np.pi / 12)) * 11 + 18)
            M4 = vec2(np.cos(3 * (np.pi / 12)) * 11 + 18, np.sin(3 * (np.pi / 12)) * 11 + 18)

            pos = self.body.position
            if M1.x <= pos.x <= M2.x and M2.y <= pos.y <= M3.y:
                return True
            else:
                return False

        elif self.env == 'RaceCircle_v2':
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
        max_x = SCREEN_WIDTH / PPM  # 36
        max_y = SCREEN_HEIGHT / PPM  # 36
        r = (SCREEN_WIDTH / PPM / 2) - 7
        c = vec2(max_x / 2, max_y / 2)  # centre
        s = vec2(c.x + r, c.y)  # starting point

        # Orthogonal projection to the circle
        theta = Util.angle_v2(Util.normalize(s - c), Util.normalize(self.body.position - c))
        theta = Util.deg_to_rad(theta)
        p_h = vec2(r * np.cos(theta) + c.x, r * np.sin(theta) + c.y)

        # Tangent to the circle
        tangent = Util.rotate(Util.normalize(c - p_h), -90.0)

        forward = Util.normalize(self.body.GetWorldVector((0, 1)))
        orientation = Util.angle(forward, tangent) / 180.0
        orientation = round(orientation, 2)  # only 3 decimals

        return orientation

    def distance_lane(self):
        max_x = SCREEN_WIDTH / PPM  # 36
        max_y = SCREEN_HEIGHT / PPM  # 36
        r = (SCREEN_WIDTH / PPM / 2) - 7  #
        c = vec2(max_x / 2, max_y / 2)  # center

        d = round(np.abs(Util.distance(self.body.position, c) - (r + self.car_radius)), 2)
        return d

    def normalize_sensors_value(self, val):
        return round(Util.min_max_normalization_m1_1(val, _min=0.0, _max=self.raycast_length), 2)

    def normalize_distance(self, val, min, max):
        return Util.min_max_normalization_m1_1(val, _min=min, _max=max)

    def reward_function(self):

        flag_col = False
        if self.collision:
            flag_col = True

        # Goal Reached
        flag_gr = False
        if self.goal_reached:
            flag_gr = True

        # TODO reward shaping : current_distance - previous_distance
        # Process getting_closer reward
        flag_gc = False
        coeff = 1.
        if self.distance < self.last_distance:
            flag_gc = True
            coeff = 1.
        elif self.distance > self.last_distance:
            flag_gc = True
            coeff = -1.

        # Process sensor's value
        r = np.fromiter((Reward.reward_sensor(s) for s in self.sensors), self.sensors.dtype,
                        count=len(self.sensors))  # Apply sensor reward function to all sensors value
        reward_sensors = np.amin(r)  # take the min value

        # Overall reward
        reward = flag_gr * Reward.GOAL_REACHED + flag_col * Reward.COLLISION + \
                 reward_sensors + coeff * flag_gc * Reward.GETTING_CLOSER + Reward.LIVING_PENALTY

        return reward

    def before_step(self):
        """
            Processing before the environment takes next step
        """
        # Select action
        self.action = self.brain.select_action(self.state)
        self.update_friction()
        self.update_drive_force(Action(self.action))
        self.update_turn(Action(self.action))
        # self.update_manual_drive_force()
        # self.update_turn_manual()

    def after_step(self):
        """
            Processing after the environment took step
        """
        # ------------------ Observation from the environment ----------------------------

        observation = []

        # Distance to goal
        self.distance = self.distance_goal()
        self.check_goal_reached()
        max_dist = 11 * (2 * np.pi)
        distance_norm = self.normalize_distance(val=self.distance, min=0.0, max=max_dist)
        observation.append(distance_norm)

        # Orientation in lane
        self.orientation = self.orientation_lane()
        observation.append(self.orientation)

        # Distance in lane
        self.distance_road = self.distance_lane()
        # print('self.distance_road', self.distance_road)
        distance_road_norm = self.normalize_distance(val=self.distance_road, min=0.0, max=7 - (self.car_radius * 2))
        observation.append(distance_road_norm)

        # Sensor's value
        self.read_sensors()
        for sensor in self.sensors:
            normed_sensor = self.normalize_sensors_value(sensor)  # Normalize sensor's value
            observation.append(normed_sensor)

        # Speed of the car
        current_forward_normal = self.body.GetWorldVector((0, 1))
        speed = self.forward_velocity.dot(current_forward_normal)
        speed_norm = Util.min_max_normalization_m1_1(speed, _min=0., _max=self.max_forward_speed)
        observation.append(speed_norm)

        # State
        observation = np.asarray(observation)

        # Time limits terminal condition
        if self.timesteps >= 999:
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

        if self.done:  # start new episode
            self.episodes += 1
            self.scores.append(self.score)

            # Calculate average over the last episodes
            self.average_score = sum(self.scores) / len(self.scores)
            self.tot_timesteps += self.timesteps  # increment total number of timesteps of all episodes

            # Periodically print current average of reward
            if self.episodes % 1 == 0:
                print("episode: {:5.0f}, timesteps: {:3.0f}, tot_timestep: {:8.0f}, score: {:3.0f}, average: {:3.2f}"
                      .format(self.episodes, self.timesteps, self.tot_timesteps, self.score, self.average_score))

            # reset for next episode
            self.score = 0
            self.timesteps = 0
            self.reset()

            # Problem solved
            if self.average_score >= self.solved_score:  # need reach solved score and at least 100 episodes to terminate
                print("agent: {:4.0f}, *** Solved after {} episodes *** reached solved score: {}".format(self.id,
                                                                                                         self.episodes,
                                                                                                         self.solved_score))
                self.problem_done = True
                self.problem_solved = True
                return



# OLD code


    def update_manual_drive_old(self):
        speed = self.speed
        move = True
        action = 0

        key = pygame.key.get_pressed()
        if key[K_LEFT] and key[K_SPACE]:  # Turn Left && Stop
            move = False
            self.body.angularVelocity = self.rotation_speed
            action = Action.STOP_TURN_LEFT
        elif key[K_RIGHT] and key[K_SPACE]:  # Turn Right && Stop
            move = False
            self.body.angularVelocity = -self.rotation_speed
            action = Action.STOP_TURN_RIGHT
        elif key[K_SPACE]:  # Keep Orientation && Stop
            move = False
            speed = 0
            action = Action.STOP_KEEP_ORIENTATION
        elif key[K_LEFT]:  # Turn Left && Forward
            self.body.angularVelocity = self.rotation_speed
            action = Action.TURN_LEFT
        elif key[K_RIGHT]:  # Turn Right && Forward
            self.body.angularVelocity = -self.rotation_speed
            action = Action.TURN_RIGHT
        else:  # Keep Orientation && Forward
            action = Action.KEEP_ORIENTATION
            pass

        if move:
            forward_vec = self.body.GetWorldVector((0, 1))
            self.body.linearVelocity = forward_vec * speed
        else:
            # Kill velocity
            impulse = -self.forward_velocity * self.body.mass * (2. / 3.)
            self.body.ApplyLinearImpulse(impulse, self.body.worldCenter, True)  # kill forward

        return action

    def update_drive_old(self, action):
        """
            Perform agent's movement based on the input action
        """
        speed = self.speed
        move = True

        if action == Action.TURN_LEFT:  # Turn Left
            self.body.angularVelocity = self.rotation_speed
        elif action == Action.KEEP_ORIENTATION:  # Keep orientation
            pass
        elif action == Action.TURN_RIGHT:  # Turn Right
            self.body.angularVelocity = -self.rotation_speed
        elif action == Action.STOP_KEEP_ORIENTATION:  # Stop moving
            move = False
            speed = 0.
        elif action == Action.STOP_TURN_LEFT:  # Stop and turn left
            move = False
            speed = 0.
            self.body.angularVelocity = self.rotation_speed
        elif action == Action.STOP_TURN_RIGHT:  # Stop and turn right
            move = False
            speed = 0.
            self.body.angularVelocity = -self.rotation_speed

        # Move Agent
        if move:
            forward_vec = self.body.GetWorldVector((0, 1))
            self.body.linearVelocity = forward_vec * speed
        else:
            # Kill velocity
            impulse = -self.forward_velocity * self.body.mass * (2. / 3.)
            self.body.ApplyLinearImpulse(impulse, self.body.worldCenter, True)  # kill forward