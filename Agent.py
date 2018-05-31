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
    from .res import colors as Color
    from .Setup import *
    from .Util import worldToPixels
    import Util
    import Global
    from .res.print_colors import printColor

moveTicker = 0
prev_angle = 999
go_print_Turn = False
prev_turned_angle = 0


class Agent(object):
    def __init__(self, screen=None, world=None, x=0, y=0, angle=0, radius=2, training=True):
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

        self.initialSpeed = 12
        self.updateCalls = 0

        self.brain = None
        # Training flag
        self.training = training

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
        # raise NotImplementedError("Update method not implemented")
        self.updateCalls += 1

    def updateManualDriveTestAngle(self, angle):
        speed = self.initialSpeed
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
            # if turned_angle != 0 and prev_angle != 999:
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

    def updateManualDrive(self):
        speed = self.initialSpeed

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

    def learning_score(self):
        """
            Score is the mean of the reward in the sliding window
        """
        learning_score = self.brain.learning_score()
        return learning_score

    def save_brain(self, dir):
        """
            Save agent's brain (neural network model and memory) in file
            Call save_model() and save_memory()
        """
        self.save_model(dir)
        self.save_memory(dir)

    def save_model(self, dir):
        """
            Save agent's model (neural network, optimizer, loss, etc) in file
            Also create the /brain_files/ directory if it doesn't exist
        """
        timestr = time.strftime("%Y_%m_%d_%H%M%S")
        directory = dir
        network_model = directory + timestr + "_model.h5"  # neural network model file

        if not os.path.exists(os.path.dirname(directory)):
            try:
                os.makedirs(os.path.dirname(directory))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        self.brain.save_model(network_model)
        printColor(msg="Agent: {:3.0f}, ".format(self.id) +
                       "{:>25s}".format("Save Agent's model") +
                       ", file: {}".format(network_model) +
                       ", tmstp: {:10.0f}".format(Global.timestep) +
                       ", t: {}".format(Global.get_time()))

    def save_memory(self, dir):
        """
            Save Agent's memory (experiences) in csv file
            Also create the /brain_files/ directory if it doesn't exist
        """
        timestr = time.strftime("%Y_%m_%d_%H%M%S")
        directory = dir
        memory_file = directory + timestr + "_memory.csv"  # neural network model file

        if not os.path.exists(os.path.dirname(directory)):
            try:
                os.makedirs(os.path.dirname(directory))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        self.brain.save_memory(memory_file)
        printColor(msg="Agent: {:3.0f}, ".format(self.id) +
                       "{:>25s}".format("Save Agent's memory") +
                       ", file: {}".format(memory_file) +
                       ", tmstp: {:10.0f}".format(Global.timestep) +
                       ", t: {}".format(Global.get_time()))

    def load_model(self):
        """
            Load Agent's model config from file
            Everything : NN architecture, optimizer, weights, ...
        """
        directory = "./brain_files/"
        model_file = directory + "brain" + "_model.h5"  # neural network model file

        self.brain.load_model(model_file)
        printColor(msg="Agent: {:3.0f}, ".format(self.id) +
                       "{:>25s}".format("Load full model") +
                       ", file: {}".format(model_file) +
                       ", tmstp: {:10.0f}".format(Global.timestep) +
                       ", t: {}".format(Global.get_time()))

    def load_weights(self):
        """
            Load Agent's weights from file
        """
        directory = "./brain_files/"
        model_file = directory + "brain" + "_model.h5"  # neural network model file

        self.brain.load_weights(model_file)

        printColor(msg="Agent: {:3.0f}, ".format(self.id) +
                       "{:>25s}".format("Load model full weights") +
                       ", file: {}".format(model_file) +
                       ", tmstp: {:10.0f}".format(Global.timestep) +
                       ", t: {}".format(Global.get_time()))

    def load_h1_weights(self):
        """
            Load Agent's 1st hidden layer weights from file
        """
        directory = "./brain_files/"
        model_file = directory + "brain" + "_model.h5"  # neural network model file

        self.brain.load_h1_weights(model_file)

        printColor(msg="Agent: {:3.0f}, ".format(self.id) +
                       "{:>25s}".format("Load 1st hidden layer weights") +
                       ", file: {}".format(model_file) +
                       ", tmstp: {:10.0f}".format(Global.timestep) +
                       ", t: {}".format(Global.get_time()))

    def load_h1h2_weights(self):
        """
            Load Agent's 1st and 2nd hidden layer weights from file
        """
        directory = "./brain_files/"
        model_file = directory + "brain" + "_model.h5"  # neural network model file

        self.brain.load_h1h2_weights(model_file)

        printColor(msg="Agent: {:3.0f}, ".format(self.id) +
                       "{:>25s}".format("Load 1st and 2nd hidden layer weights") +
                       ", file: {}".format(model_file) +
                       ", tmstp: {:10.0f}".format(Global.timestep) +
                       ", t: {}".format(Global.get_time()))

    def load_memory(self):
        """
            Load memory from file
        """
        directory = "./brain_files/"
        memory_file = directory + "brain" + "_memory.csv"  # neural network model file

        memory_list = []
        data = pd.read_csv(memory_file)

        remove_bracket = lambda x: x.replace('[', '').replace(']', '')
        string_to_array = lambda x: np.expand_dims(np.fromstring(x, sep=' '), axis=0)

        data['state'] = data['state'].map(remove_bracket).map(string_to_array)
        data['next_state'] = data['next_state'].map(remove_bracket).map(string_to_array)

        for i, row in data.iterrows():
            exp = (row['state'], row['action'], row['reward'], row['next_state'])
            memory_list.append(exp)

        printColor(msg="Agent: {:3.0f}, ".format(self.id) +
                       "{:>25s}".format("Load full memory") +
                       ", file: {}".format(memory_file) +
                       ", tmstp: {:10.0f}".format(Global.timestep) +
                       ", t: {}".format(Global.get_time()))

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
