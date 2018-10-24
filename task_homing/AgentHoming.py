
import numpy as np
import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (vec2)
from enum import Enum
from pygame.locals import *

try:
    # Running in PyCharm
    import res.colors as Color
    # from ..res import colors as Color
    from AI.DQN import DQN
    from objects.Agent import Agent
    from Setup import *
    from Util import worldToPixels, pixelsToWorld
    import Util
    from task_homing.RayCastCallback import RayCastCallback
    import res.print_colors as PrintColor
    import task_homing.debug_homing as debug_homing
    import task_homing.global_homing as global_homing
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
    from task_homing.RayCastCallback import RayCastCallback
    from ..res import print_colors as PrintColor
    import task_homing.debug_homing
    import task_homing.global_homing
    from .. import Global

hyperparams = {
    'layers': (64, 64),
    'mem_capacity': 100000,
    'batch_size': 32,
    'eps_start': 1.,
    'eps_end': 0.1,
    'eps_test': 0.05,
    'exploration_steps': 1000,
    'gamma': 0.99,
    'lr': 0.001,
    'update_target_steps': 1000,
    'use_double_dqn': True,
    'use_prioritized_experience_replay': False
}

# -------------------- Agent ----------------------

# Agent's possible actions
class Action(Enum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    KEEP_ORIENTATION = 2
    STOP = 3
    STOP_TURN_LEFT = 4
    STOP_TURN_RIGHT = 5

# Rewards Mechanism
class Reward:
    GETTING_CLOSER = 0.1
    LIVING_PENALTY = -1.
    GOAL_REACHED = 1000.

    @classmethod
    def sensor_reward(cls, x):
        """
            Return -100 if proximity sensor is equal to 0 (that means collision occurs) else 0
        """
        if not x:
            return -100.
        else:
            return 0.


class AgentHoming(Agent):
    def __init__(self, screen=None, world=None, x=0, y=0, angle=0, radius=1.5, id=-1, goals=vec2(0, 0), numAgents=0):
        super(AgentHoming, self).__init__(screen, world, x, y, angle, radius)

        # Agent's ID
        self.id = id

        # Proximity Sensors
        self.raycast_length = 2.0
        self.initial_raycastDiagonalColor = Color.Yellow
        self.raycastDiagonalColor = Color.Yellow
        self.initial_raycastStraightColor = Color.Red
        self.raycastStraightColor = Color.Red
        forward_vec = vec2(0, 1)  # Raycast Front Middle
        top_right = vec2(np.sqrt(2)/2, np.sqrt(2)/2)  # Raycast Front Right : pi/4 <-> [ cos(pi/4) , sin(pi/4) ]
        top_left = vec2(-np.sqrt(2)/2, np.sqrt(2)/2)  # Raycast Front Left : 3*pi/4 <-> [ -cos(pi/4) , sin(pi/4) ]
        right = vec2(1, 0)  # Raycast Right Side
        left = vec2(-1, 0)  # Raycast Left Side

        # bottom_left = vec2(-0.70, -0.70)  # Raycast Left Back
        # backward = vec2(0, -1)  # Raycast Middle Back
        # bottom_right = vec2(0.70, -0.70)  # Raycast Right Back

        self.raycast_vectors = (top_left, forward_vec, top_right, left, right) # bottom_right, backward, bottom_left)  # Store raycast direction vector for each raycast
        self.num_sensors = len(self.raycast_vectors)
        self.sensors = np.ones(self.num_sensors) * self.raycast_length # store the value of each sensors
        print("num sensors : {}".format(self.num_sensors))

        # Input size
        self.input_size = self.num_sensors + 1  # 8 sensors + 1 orientation

        # Number of agents
        self.numAgents = numAgents

        # Goal
        self.currentGoalIndex = 0
        self.goals = goals
        self.goalReachedThreshold = 2.4  # If agent-to-goal distance is less than this value then agent reached the goal
        self.num_goals = len(self.goals)  # number of goals

        # Collision with obstacles (walls, obstacles in path)
        self.t2GCollisionCount = 0
        self.collisionCount = 0
        self.elapsedTimestepObstacleCollision = 0.00  # timestep passed between collision (for same objects collision)
        self.startTimestepObstacleCollision = 0.00  # start timestep since a collision (for same objects collision)
        self.lastObstacleCollide = None

        # Collision with Agents
        self.elapsedTimestepAgentCollision = np.zeros(numAgents)  # timestep passed between collision (for same objects collision)
        self.startTimestepAgentCollision = np.zeros(numAgents)  # start timestep since a collision (for same objects collision)
        self.t2GAgentCollisionCount = 0
        self.agentCollisionCount = 0

        self.currentCollisionCount = 0  # number of objects the agent is colliding at the current timestep

        # Event features
        self.goalReachedCount = 0
        self.startTime = 0.0
        self.startTimestep = 0
        self.elapsedTime = 0.00
        self.elapsedTimestep = 0

        self.last_reward = 0.0  # last agent's reward
        self.last_distance = 0.0  # last agent's distance to the goal
        self.distance = self.distance_goal()  # 0.0  # current distance to the goal
        self.last_position = vec2(self.body.position.x, self.body.position.y)  # keep track of previous position
        self.last_orientation = self.orientation_goal()

        self.inited = False
        self.state = None
        self.action = None

    def setup(self, training=True, random_agent=False):

        # Update the agent's flag
        self.training = training

        if random_agent:  # can't train when random
            self.training = False
            self.random_agent = random_agent

        # Create agent's brain
        self.brain = DQN(input_size=self.input_size, action_size=len(list(Action)), id=self.id,
                         training=self.training, random_agent=self.random_agent, **hyperparams)

        # Set agent's position : Start from goal 2
        start_pos = self.goals[1]
        self.body.position = start_pos

        # Set agent's orientation : Start by looking at goal 1
        toGoal = Util.normalize(pixelsToWorld(self.goals[0]) - start_pos)
        forward = vec2(0, 1)
        angleDeg = Util.angle(forward, toGoal)
        angle = Util.degToRad(angleDeg)
        angle = -angle
        self.body.angle = angle

        # -------------------- Reset agent's variables --------------------

        # Goals
        self.currentGoalIndex = 0

        # Collision with obstacles (walls, obstacles in path, agents)
        self.t2GCollisionCount = 0
        self.collisionCount = 0
        self.elapsedTimestepObstacleCollision = 0.00  # timestep passed between collision (for same objects collision)
        self.startTimestepObstacleCollision = 0.00  # start timestep since a collision (for same objects collision)
        self.lastObstacleCollide = None

        # Collision with Agents
        self.elapsedTimestepAgentCollision = np.zeros(
            self.numAgents)  # timestep passed between collision (for same objects collision)
        self.startTimestepAgentCollision = np.zeros(
            self.numAgents)  # start timestep since a collision (for same objects collision)
        self.t2GAgentCollisionCount = 0
        self.agentCollisionCount = 0

        self.currentCollisionCount = 0  # number of objects the agent is colliding at the current timestep

        # Event features
        self.goalReachedCount = 0
        self.startTime = 0.0
        self.startTimestep = 0
        self.elapsedTime = 0.00
        self.elapsedTimestep = 0

        self.last_reward = 0.0
        self.last_distance = 0.0
        self.distance = self.distance_goal()  # current distance to the goal
        self.last_position = vec2(self.body.position.x, self.body.position.y)
        self.last_orientation = self.orientation_goal()

        debug_homing.print_event(color=PrintColor.PRINT_CYAN, agent=self,
                                 event_message="agent is ready")

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

        # Draw raycasts
        for i in range(self.num_sensors):
            v = self.body.GetWorldVector(self.raycast_vectors[i])
            p1 = self.body.worldCenter + v * self.radius
            p2 = p1 + v * self.raycast_length
            if i == 0 or i == 2: #i % 2 == 0:
                ray_color = self.raycastDiagonalColor
            else:
                ray_color = self.raycastStraightColor
            pygame.draw.line(self.screen, ray_color, worldToPixels(p1), worldToPixels(p2))

    def read_sensors(self):

        # Read raycasts value
        for i in range(self.num_sensors):
            raycast = RayCastCallback()
            v = self.body.GetWorldVector(self.raycast_vectors[i])
            p1 = self.body.worldCenter + v * self.radius
            p2 = p1 + v * self.raycast_length
            self.world.RayCast(raycast, p1, p2)
            if raycast.hit:
                dist = (p1 - raycast.point).length  # distance to the hit point
                self.sensors[i] = round(dist, 2)
            else:
                self.sensors[i] = self.raycast_length  # default value is raycastLength

    def normalize_sensors_value(self, val):
        return Util.minMaxNormalization_m1_1(val, _min=0.0, _max=self.raycast_length)

    def update_drive(self, action):
        """
            Perform agent's movement based on the input action
        """
        speed = self.speed
        move = True

        if action == Action.TURN_LEFT:  # Turn Left
            self.body.angularVelocity = self.rotation_speed
        elif action == Action.TURN_RIGHT:  # Turn Right
            self.body.angularVelocity = -self.rotation_speed
        elif action == Action.KEEP_ORIENTATION:  # Don't turn
            pass
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
        toGoal = Util.normalize(self.goals[self.currentGoalIndex] - self.body.position)
        forward = Util.normalize(self.body.GetWorldVector((0, 1)))
        orientation = Util.angle(forward, toGoal) / 180.0
        orientation = round(orientation, 2)  # only 3 decimals
        return orientation

    def distance_goal(self):
        distance = np.sqrt((self.body.position.x - self.goals[self.currentGoalIndex].x) ** 2 +
                           (self.body.position.y - self.goals[self.currentGoalIndex].y) ** 2)
        return round(distance, 2)

    def compute_goal_reached(self):
        self.goalReachedCount += 1
        self.elapsedTime = global_homing.timer - self.startTime
        self.elapsedTimestep = Global.sim_timesteps - self.startTimestep

        # Goal reached event
        debug_homing.print_event(color=PrintColor.PRINT_RED, agent=self,
                                 event_message="reached goal: {}".format(self.currentGoalIndex + 1))

        # Reset, Update
        self.startTime = global_homing.timer
        self.startTimestep = Global.sim_timesteps
        self.currentGoalIndex = (self.currentGoalIndex + 1) % self.num_goals  # change goal
        self.distance = self.distance_goal()
        self.t2GCollisionCount = 0
        self.t2GAgentCollisionCount = 0

    def reward_function(self):

        # Process getting_closer reward
        flagGC = False
        getting_closer = 0.
        if self.distance < self.last_distance:
            flagGC = True  # getting closer
            getting_closer = Reward.GETTING_CLOSER

        # Check Goal Reached
        flagGR = False
        if self.distance < self.goalReachedThreshold:
            flagGR = True

        # Process sensor's value
        r = np.fromiter((Reward.sensor_reward(s) for s in self.sensors), self.sensors.dtype,
                        count=len(self.sensors))  # Apply sensor reward function to all sensors value
        sensor_reward = np.amin(r)  # take the min value

        # Overall reward
        reward = flagGC * getting_closer + sensor_reward + flagGR * Reward.GOAL_REACHED + Reward.LIVING_PENALTY

        return reward

    def update(self):
        """
            Main function of the agent
        """
        super(AgentHoming, self).update()

        done = False # done is always False in non-episodic task

        if self.inited:

            # agent's distance to the goal
            self.distance = self.distance_goal()
            if self.distance < self.goalReachedThreshold: # reached Goal
                self.compute_goal_reached()

            # Observe from the environment
            observation = []
            orientation = self.orientation_goal() # orientation to the goal
            observation.append(orientation)
            self.read_sensors() # sensor's value
            for i in range(self.num_sensors):
                normed_sensor = self.normalize_sensors_value(self.sensors[i]) # Normalize sensor's value
                observation.append(normed_sensor)
            observation = np.asarray(observation)

            # Reward from the environment
            reward = self.reward_function()

            # Record experience
            next_state = self.brain.preprocess(observation)
            self.brain.record((self.state, self.action, reward, next_state, done))
            self.state = next_state

            self.brain.train()

            # Update variables old states
            self.last_distance = self.distance
            self.elapsedTime = global_homing.timer - self.startTime
            self.elapsedTimestep = Global.sim_timesteps - self.startTimestep
            self.last_position = vec2(self.body.position.x, self.body.position.y)
            self.last_orientation = self.body.angle

        # Act in the environment
        self.action = self.brain.select_action(self.state)
        self.updateFriction()
        self.update_drive(Action(self.action))

        # Initialization done after agents perform its first action
        if not self.inited:
            # Observe from the environment
            observation = []
            orientation = self.orientation_goal()  # orientation to the goal
            observation.append(orientation)
            self.read_sensors()  # sensor's value
            for i in range(self.num_sensors):
                normed_sensor = self.normalize_sensors_value(self.sensors[i])  # Normalize sensor's value
                observation.append(normed_sensor)
            observation = np.asarray(observation)
            self.state = self.brain.preprocess(observation)

            self.inited = True

    def collision_color(self):
        """
            Change agent color during collision
        """
        self.currentCollisionCount += 1

        self.color = Color.DeepSkyBlue
        self.raycastDiagonalColor = Color.Magenta
        self.raycastStraightColor = Color.Magenta

    def end_collision_color(self):
        """
            Reset agent color when end of collision
        """
        self.currentCollisionCount -= 1

        if self.currentCollisionCount <= 0:
            self.currentCollisionCount = 0
            self.color = self.initial_color
            self.raycastStraightColor = self.initial_raycastStraightColor
            self.raycastDiagonalColor = self.initial_raycastDiagonalColor
