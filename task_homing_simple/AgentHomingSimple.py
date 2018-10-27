import numpy as np
import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (vec2)
from enum import Enum
from keras.layers import Dense
from keras.layers import Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from pygame.locals import *
import sys
from collections import deque
import time
import os
import errno
import csv
import pandas as pd
import keras.optimizers

try:
    # Running in PyCharm
    import res.colors as Color
    # from ..res import colors as Color
    from AI.DQN import DQN
    from objects.Agent import Agent
    from Setup import *
    from Util import worldToPixels, pixelsToWorld
    import Util
    import res.print_colors as PrintColor
    import task_homing_simple.debug_homing_simple as debug_homing_simple
    import task_homing_simple.global_homing_simple as global_homing_simple
    import Global
except NameError as err:
    print(err, "--> our error message")
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running from command line -> Import libraries as package")
    from ..res import colors as Color
    from ..AI.DQN import DQN
    from ..objects.Agent import Agent
    from ..Setup import *
    from ..Util import worldToPixels, pixelsToWorld
    from .. import Util
    from ..res import print_colors as PrintColor
    import task_homing_simple.debug_homing_simple as debug_homing_simple
    import task_homing_simple.global_homing_simple as global_homing_simple
    from .. import Global

# Agent brain config
homing_simple_hyperparams = {
    'layers': (32, 32),
    'mem_capacity': 100000,
    'batch_size': 32,
    'eps_start': 1.,
    'eps_end': 0.05,
    'eps_test': 0.01,
    'exploration_steps': 1000,
    'gamma': 0.95,
    'lr': 0.001,
    'update_target_steps': 1000,
    'use_double_dqn': True,
    'use_prioritized_experience_replay': False
}


# -------------------- Agent ----------------------

# Agent's possible actions
class Action(Enum):
    TURN_LEFT = 0
    KEEP_ORIENTATION = 1
    TURN_RIGHT = 2
    STOP = 3
    STOP_TURN_LEFT = 4
    STOP_TURN_RIGHT = 5

# Rewards Mechanism
class Reward:
    LIVING_PENALTY = 0. # -0.1
    GOAL_REACHED = 100.

    @classmethod
    def reward_distance(cls, d):
        return 1 - ((d/Util.MAX_DIST) ** 0.4)

    @classmethod
    def reward_orientation(cls, o):
        return 1 - (np.abs(o) ** 0.4)

    @classmethod
    def reward_global(cls, d, o):
        return (cls.reward_distance(d) + cls.reward_orientation(o)) / 2


class AgentHomingSimple(Agent):
    def __init__(self, screen=None, world=None, x=0, y=0, angle=0, radius=1.5, id=-1, goals=vec2(0, 0), num_agents=1):
        super(AgentHomingSimple, self).__init__(screen, world, x, y, angle, radius)

        # Agent's ID
        self.id = id

        # Input size
        self.input_size = 2

        # Goal
        self.current_goal_index = 0
        self.goals = goals
        self.goal_reached_threshold = 2.4  # If agent-to-goal distance is less than this value then agent reached the goal
        self.num_goals = len(self.goals)  # number of goals

        # Event features
        self.goal_reached_count = 0
        self.start_time = 0.0
        self.start_timestep = 0
        self.elapsed_time = 0.00
        self.elapsed_timestep = 0

        self.last_distance = 0.0  # last agent's distance to the goal
        self.distance = self.distance_goal()  # current distance to the goal
        self.last_position = vec2(self.body.position.x, self.body.position.y)  # keep track of previous position
        self.last_orientation = self.orientation_goal()
        self.ready_observe = False  # flag that tells if agent receive reward and new state at the beginning
        self.reward = None  # last agent's reward
        self.action = None
        self.state = None
        self.orientation = self.orientation_goal()

        # Number of agents
        self.num_agents = num_agents

        # List of agents
        self.agents_list = []

        # Meeting with Agents
        self.elapsed_timestep_meetings = np.zeros(num_agents)  # timestep passed between meetings
        self.start_timestep_meetings = np.zeros(num_agents)  # start timestep since a meeting

        # Terminal state : reaching goal
        self.reached_goal = False

        self.prev_shaping = None
        self.done = False

    def setup(self, training=True, random_agent=False):

        # Update the agent's flag
        self.training = training

        if random_agent:  # can't train when random
            self.training = False
            self.random_agent = random_agent

        # Create agent's brain
        self.brain = DQN(input_size=self.input_size, action_size=len(list(Action)), id=self.id,
                         training=self.training, random_agent=self.random_agent, **homing_simple_hyperparams)

        # Initial position : start from goal 2
        start_pos = self.goals[1]
        self.body.position = start_pos
        self.distance = self.distance_goal()

        # Initial orientation : start by looking at goal 1
        toGoal = Util.normalize(pixelsToWorld(self.goals[0]) - start_pos)
        forward = vec2(0, 1)
        angleDeg = Util.angle(forward, toGoal)
        angle = Util.degToRad(angleDeg)
        angle = -angle
        self.body.angle = angle
        self.orientation = self.orientation_goal()

        # Initial state
        observation = np.asarray([self.distance, self.orientation])
        # observation = np.asarray([self.orientation])
        self.state = self.brain.preprocess(observation)

        # Goals
        self.current_goal_index = 0

        # Meeting with Agents
        self.elapsed_timestep_meetings = np.zeros(self.num_agents)
        self.start_timestep_meetings = np.zeros(self.num_agents)

        # Event features
        self.goal_reached_count = 0
        self.start_time = 0.0
        self.start_timestep = 0
        self.elapsed_time = 0.00
        self.elapsed_timestep = 0

        self.prev_shaping = None
        self.done = False

        debug_homing_simple.printEvent(color=PrintColor.PRINT_CYAN, agent=self,
                                       event_message="agent is ready")

    def draw(self):

        # Collision's circle
        position = self.body.transform * self.fixture.shape.pos * PPM
        position = (position[0], SCREEN_HEIGHT - position[1])
        pygame.draw.circle(self.screen, Color.Blue, [int(x) for x in position], int(self.radius * PPM))

        # Visual triangle
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

        # # Communication's ellipse
        # position = self.body.transform * self.fixture.shape.pos * PPM
        # position = (position[0], SCREEN_HEIGHT - position[1])
        # pygame.draw.circle(self.screen, Color.Red, [int(x) for x in position], int(self.communication_range * PPM), 1)

    def update_drive(self, action):
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
        elif action == Action.STOP:  # Stop moving
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
            impulse = -self.getForwardVelocity() * self.body.mass * (2. / 3.)
            self.body.ApplyLinearImpulse(impulse, self.body.worldCenter, True)  # kill forward

    def orientation_goal(self):
        """
            Get agent orientation to the goal
            Angle in degrees, normalized between [-1,1] and counter-clockwise
            0 degree -> 0, perfectly aligned in direction to the goal
            90 deg -> 0.5, goal is on the right
            180 deg -> 1 , reverse facing
            -90 deg -> -0.5, goal is on the left
        """
        toGoal = Util.normalize(self.goals[self.current_goal_index] - self.body.position)
        forward = Util.normalize(self.body.GetWorldVector((0, 1)))
        orientation = Util.angle(forward, toGoal) / 180.0
        orientation = round(orientation, 2)  # only 3 decimals
        return orientation

    def distance_goal(self):
        distance = np.sqrt((self.body.position.x - self.goals[self.current_goal_index].x) ** 2 +
                           (self.body.position.y - self.goals[self.current_goal_index].y) ** 2)
        return round(distance, 2)

    def compute_goal_reached(self):
        self.goal_reached_count += 1
        self.elapsed_time = global_homing_simple.timer - self.start_time
        self.elapsed_timestep = Global.sim_timesteps - self.start_timestep

        # Goal reached event
        debug_homing_simple.printEvent(color=PrintColor.PRINT_RED, agent=self,
                                       event_message="reached goal: {}".format(self.current_goal_index + 1))

        # Reset, Update
        self.start_time = global_homing_simple.timer
        self.start_timestep = Global.sim_timesteps
        self.current_goal_index = (self.current_goal_index + 1) % self.num_goals  # change goal

        self.done = True

    def reward_function(self, distance, orientation):

        # Reward shaping
        potential_shaping = 0
        # coeff of 100 to make it 10^0 (1 meter) order and not millimeter or milliangle
        shaping = - (10/2) * (distance + np.abs(orientation))

        if self.prev_shaping is not None:
            potential_shaping = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Goal Reached
        if self.distance < self.goal_reached_threshold:
            return Reward.GOAL_REACHED

        # reward = Reward.reward_global(distance, orientation) + Reward.LIVING_PENALTY

        # Overall reward
        reward = potential_shaping # + Reward.LIVING_PENALTY

        return reward

    def before_step(self):
        """
            Processing before the environment takes next step
        """
        # Select action
        self.action = self.brain.select_action(self.state)
        self.updateFriction()
        self.update_drive(Action(self.action))

    def after_step(self):
        """
            Processing after the environment took step
        """

        # ------------------ Observation from the environment ----------------------------

        observation = []

        # Distance to goal
        distance = self.distance_goal()
        if distance < self.goal_reached_threshold:  # reached Goal
            self.compute_goal_reached()
        observation.append(distance)

        # Orientation to goal
        orientation = self.orientation_goal()  # orientation to the goal
        observation.append(orientation)

        observation = np.asarray(observation)

        # Time limits terminal condition
        if self.elapsed_timestep > 3000:
            self.done = True

        # --------------------------------------------------------------------------------

        # Reward from the environment
        reward = self.reward_function(distance, orientation)

        # Record experience
        next_state = self.brain.preprocess(observation)
        self.brain.record((self.state, self.action, reward, next_state, self.done))
        self.state = next_state
        self.brain.train()

        # Update variables old states
        self.distance = distance
        self.orientation = orientation
        self.elapsed_time = global_homing_simple.timer - self.start_time
        self.elapsed_timestep = Global.sim_timesteps - self.start_timestep

        if self.done: # start new episode
            self.done = False
            self.prev_shaping = None

    # Place holder code


    # TODO : Check distance to others agents
    # for agent in self.agents_list:
    #     if self != agent:
    #
    #         # In communication range
    #         if Util.sqr_distance(self.body.position, agent.body.position) \
    #                 <= (self.communication_range + agent.radius) ** 2:
    #             pass
    #             # # Frequency of communication between both agents
    #             # if self.is_timestep_meeting_allowed(agent):
    #             #     print("Myself agent: {} just met agent: {}".format(self.id, agent.id))

    def is_timestep_meeting_allowed(self, other_agent):
        """
            Check if tmstp between successive meetings between the same 2 agent was too short
        """
        MEETING_TIMESTEP = 60  # 1s -> 60 timesteps

        agentA = self
        agentB = other_agent

        idA = agentA.id
        idB = agentB.id

        agentA.elapsed_timestep_meetings[idB] = Global.sim_timesteps - agentA.start_timestep_meetings[idB]
        agentB.elapsed_timestep_meetings[idA] = Global.sim_timesteps - agentB.start_timestep_meetings[idA]

        if agentA.elapsed_timestep_meetings[idB] <= MEETING_TIMESTEP \
                and agentB.elapsed_timestep_meetings[idA] <= MEETING_TIMESTEP:  # Check elapsed timestep

            print("too short")
            agentA.start_timestep_meetings[idB] = Global.sim_timesteps  # update based on elapsed time
            agentB.start_timestep_meetings[idA] = Global.sim_timesteps  # update based on elapsed time

            return False  # time was too short since the previous meeting -> return False

        return True

        # Old code
        #     def update(self):
        #         """
        #             Main function of the agent
        #         """
        #         super(AgentHomingSimple, self).update()
        #         # TODO refactor agent homing task to work with updated DQN code
        #
        #         # Orientation to the goal
        #         if self.reached_goal:
        #             orientation = 0.0
        #             self.reached_goal = False
        #         else:
        #             orientation = self.orientation_goal()
        #
        #         # Agent's observation
        #         observation = np.asarray([orientation])
        #
        #         # Select action using AI
        #         action = self.update_brain(self.reward, observation)
        #         self.updateFriction()
        #         self.update_drive(Action(action))
        #         # self.updateManualDrive()
        #         # self.remainStatic()
        #
        #         # Calculate agent's distance to the goal
        #         self.distance = self.distance_goal()
        #
        #         # Agent's Reward
        #         self.reward = self.reward_function()
        #
        #         # Reached Goal
        #         if self.distance < self.goal_reached_threshold:
        #             self.reached_goal = True
        #             self.compute_goal_reached()
        #
        #         # Update variables
        #         self.last_distance = self.distance
        #         self.elapsed_time = global_homing_simple.timer - self.start_time
        #         self.elapsed_timestep = Global.sim_timesteps - self.start_timestep
        #         self.last_position = vec2(self.body.position.x, self.body.position.y)
        #         self.last_orientation = self.body.angle
        #         self.action = action
        #         self.ready_observe = True

        # def update_brain(self, reward, observation, done=False):
        #     """
        #         Main function of the agent's brain
        #         Return the action to be performed
        #     """
        #     if self.ready_observe:
        #         new_state = self.brain.preprocess(observation)
        #         experience = (self.state, self.action, self.reward, new_state, done)
        #
        #         # Add new experience to memory
        #         self.brain.record(experience)
        #
        #         # Update variable's states
        #         self.state = new_state
        #         self.reward = reward
        #
        #         # Training
        #         self.brain.train()
        #
        #     # Select action
        #     action = self.brain.select_action(self.state)
        #
        #     # Append reward
        #     self.brain.reward_window.append(reward)
        #
        #     return action
