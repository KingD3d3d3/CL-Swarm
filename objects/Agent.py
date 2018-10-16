from __future__ import division
import random
import sys

import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (vec2, pi)
from enum import Enum
from pygame.locals import *
import time
import os
import errno
import csv
import pandas as pd
import numpy as np
try:
    # Running in PyCharm
    import res.colors as Color
    from Setup import *
    from Util import worldToPixels
    import Util
    from res.print_colors import printColor
    import Global
except:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running from command line -> Import libraries as package')
    from ..res import colors as Color
    from ..Setup import *
    from ..Util import worldToPixels
    from .. import Util
    from .. import Global
    from ..res.print_colors import printColor


class Agent(object):
    def __init__(self, screen=None, world=None, x=0, y=0, angle=0, radius=1.5):
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

        # Specifications
        self.speed = 10 # m/s
        self.rotation_speed = 2.0 * np.pi # rad/s
        self.communication_range = 4 # m

        self.brain = None

        # Distance travelled by agent at each timestep
        self.delta_dist = self.speed * (1. / TARGET_FPS)

        # Initial training and random_agent flag
        self.training = True
        self.random_agent = False

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
        dragForceMagnitude = -50 * currentForwardSpeed  # -10
        self.body.ApplyForce(dragForceMagnitude * currentForwardNormal, self.body.worldCenter, True)

    def remainStatic(self):
        self.updateFriction()
        speed = 0

        forward_vec = self.body.GetWorldVector((0, 1))
        self.body.linearVelocity = forward_vec * speed

    def update(self):

        # Consistency: Both agent's and brain should have the same flag value
        if self.random_agent != self.brain.random_agent:
            print('random_agent FLAG wasnt the same')
            self.go_random_agent()
        if self.training != self.brain.training:
            print('training FLAG wasnt the same')
            self.stop_training()

    def updateManualDrive(self):
        speed = self.speed
        move = True

        key = pygame.key.get_pressed()
        if key[K_LEFT]:  # Turn Left
            self.body.angularVelocity = self.rotation_speed #10.5 #5
            pass
        if key[K_RIGHT]:  # Turn Right
            self.body.angularVelocity = -self.rotation_speed #10.5 #-5
            pass
        if key[K_SPACE]:  # Break
            move = False
            speed = 0
            pass
        forward_vec = self.body.GetWorldVector((0, 1))

        if move:
            self.body.linearVelocity = forward_vec * speed
        else:
            impulse = -self.getForwardVelocity() * self.body.mass * (2. / 3.)
            self.body.ApplyLinearImpulse(impulse, self.body.worldCenter, True)  # kill forward

    def draw(self):
        position = self.body.transform * self.fixture.shape.pos * PPM
        position = (position[0], SCREEN_HEIGHT - position[1])
        pygame.draw.circle(self.screen, self.color, [int(x) for x in position], int(self.radius * PPM))

        current_forward_normal = self.body.GetWorldVector((0, 1))
        pygame.draw.line(self.screen, Color.White, worldToPixels(self.body.worldCenter),
                         worldToPixels(self.body.worldCenter + current_forward_normal * self.radius))

    def learning_score(self):
        """
            Score is the mean of the reward in the sliding window
        """
        learning_score = self.brain.learning_score()
        return learning_score

    def save_brain(self, dir, suffix=""):
        """
            Save agent's brain (neural network model and memory) in file
            Call save_model() and save_memory()
        """
        self.save_model(dir, suffix)
        self.save_memory(dir, suffix)

    def save_model(self, dir, suffix=""):
        """
            Save agent's model (neural network, optimizer, loss, etc) in file
            Also create the /brain_files/ directory if it doesn't exist
        """

        timestr = Util.getTimeString()
        directory = dir
        timestep = "_" + str(Global.timestep) + "tmstp"
        suffix = "_" + suffix
        network_model = directory + timestr + timestep + suffix + "_model.h5"  # neural network model file

        if not os.path.exists(os.path.dirname(directory)):
            try:
                os.makedirs(os.path.dirname(directory))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        self.brain.save_model(network_model)

    def save_memory(self, dir, suffix=""):
        """
            Save Agent's memory (experiences) in csv file
            Also create the /brain_files/ directory if it doesn't exist
        """
        timestr = Util.getTimeString()
        directory = dir
        timestep = "_" + str(Global.timestep) + "tmstp"
        suffix = "_" + suffix
        memory_file = directory + timestr + timestep + suffix + "_memory.csv"  # neural network model file

        if not os.path.exists(os.path.dirname(directory)):
            try:
                os.makedirs(os.path.dirname(directory))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        self.brain.save_memory(memory_file)

    def load_model(self, file=""):
        """
            Load Agent's model config from file
            Everything : NN architecture, optimizer, weights, ...
        """
        self.random_agent = False
        if file != "":
            model_file = file
        else:
            # Default file to load
            directory = "./brain_files/"
            model_file = directory + "brain" + "_model.h5"  # neural network model file

        self.brain.load_model(model_file)

    def load_full_weights(self, file=""):
        """
            Load Agent's weights from file
        """
        self.random_agent = False
        if file != "":
            model_file = file
        else:
            # Default file to load
            directory = "./brain_files/"
            model_file = directory + "brain" + "_model.h5"  # neural network model file

        self.brain.load_full_weights(model_file)

    def load_h1_weights(self, file=""):
        """
            Load Agent's 1st hidden layer weights from file
        """
        self.random_agent = False
        if file != "":
            model_file = file
        else:
            # Default file to load
            directory = "./brain_files/"
            model_file = directory + "brain" + "_model.h5"  # neural network model file

        self.brain.load_h1_weights(model_file)

    def load_h2_weights(self, file=""):
        """
            Load Agent's 2nd hidden layer weights from file
        """
        self.random_agent = False
        if file != "":
            model_file = file
        else:
            # Default file to load
            directory = "./brain_files/"
            model_file = directory + "brain" + "_model.h5"  # neural network model file

        self.brain.load_h2_weights(model_file)

    def load_out_weights(self, file=""):
        """
            Load Agent's output layer weights from file
        """
        self.random_agent = False
        if file != "":
            model_file = file
        else:
            # Default file to load
            directory = "./brain_files/"
            model_file = directory + "brain" + "_model.h5"  # neural network model file

        self.brain.load_out_weights(model_file)

    def load_h1h2_weights(self, file):
        """
            Load Agent's 1st and 2nd hidden layer weights from file
        """
        self.random_agent = False
        if file != "":
            model_file = file
        else:
            # Default file to load
            directory = "./brain_files/"
            model_file = directory + "brain" + "_model.h5"  # neural network model file

        self.brain.load_h1h2_weights(model_file)

    def load_h2out_weights(self, file):
        """
            Load Agent's 2nd hidden layer and output layer weights from file
        """
        self.random_agent = False
        if file != "":
            model_file = file
        else:
            # Default file to load
            directory = "./brain_files/"
            model_file = directory + "brain" + "_model.h5"  # neural network model file

        self.brain.load_h2out_weights(model_file)


    def load_h1out_weights(self, file):
        """
            Load Agent's 1st hidden layer and output layer weights from file
        """
        self.random_agent = False
        if file != "":
            model_file = file
        else:
            # Default file to load
            directory = "./brain_files/"
            model_file = directory + "brain" + "_model.h5"  # neural network model file

        self.brain.load_h1out_weights(model_file)

    def load_memory(self, file="", size=-1):
        """
            Load memory from file
        """

        self.random_agent = False

        if file != "":
            memory_file = file
        else:
            # Default file to load
            directory = "./brain_files/"
            memory_file = directory + "brain" + "_memory.csv"  # neural network model file

        self.brain.load_memory(memory_file, size)

    def stop_training(self):
        """
            Stop training the brain (Neural Network)
            Stop exploration -> only exploitation
        """
        self.training = False
        self.brain.stop_training()

    def stop_exploring(self):
        """
            Stop exploration -> only exploitation
        """
        self.brain.stop_exploring()

    def stop_collect_experiences(self):
        """
            Agent doesn't append new experience to memory
        """
        self.brain.stop_collect_experiences()

    def training_it(self):
        """
            Return the agent's number of training iterations
        """
        return self.brain.training_it

    def go_random_agent(self):
        """
            Agent takes random action
        """
        self.random_agent = True
        self.brain.go_random_agent()

    def reset_brain(self):
        self.brain.reset_brain()












# -------------------------------------- Old dirty code ------------------------------------------------------------

    def updateManualDriveTestAngle(self, angle):
        speed = self.speed
        global moveTicker
        global prev_angle
        global go_print_Turn
        global prev_turned_angle

        if go_print_Turn:
            myAngle = Util.radToDeg(self.body.angle % (2 * pi))
            turned_angle = myAngle - prev_angle
            print('bodyAngle: {}, prev_angle: {}, angle turned : {}'.format(myAngle,
                                                                            prev_angle,
                                                                            turned_angle))
            go_print_Turn = False

        myAngle = Util.radToDeg(self.body.angle % (2 * pi))
        turned_angle = myAngle - prev_angle
        if not (prev_turned_angle - 0.1 <= turned_angle <= prev_turned_angle or
                prev_turned_angle <= turned_angle <= prev_turned_angle + 0.1):
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

moveTicker = 0
prev_angle = 999
go_print_Turn = False
prev_turned_angle = 0