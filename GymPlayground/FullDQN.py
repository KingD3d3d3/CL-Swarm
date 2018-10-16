# -*- coding: utf-8 -*-

import numpy as np
import random
from collections import deque
from keras.models import load_model

#-------------------- NETWORK -------------------------

class Network(object):

    def __init__(self, inputCnt, actionCnt):
        self.inputCnt = inputCnt
        self.actionCnt = actionCnt

        # Q-Network
        self.q_network = None

        # Target Network
        self.target_network = None

    def train(self, x, y, nb_epochs=1, verbose=0, batch_len=1):
        '''
            Fit the model
        '''
        self.q_network.fit(x, y, epochs=nb_epochs, verbose=verbose, batch_size=batch_len)

    def predict(self, state, batch_len=1, target=False):
        '''
            Predict q-values given an input state
        '''
        if target:
            # Predict from Target-network
            return self.target_network.predict(state, batch_size=batch_len)
        else:
            # Predict from Q-network
            return self.q_network.predict(state, batch_size=batch_len)

    def updateTargetNetwork(self):
        '''
            Update Target-Network : copy Q-Network weights into Target-Network
        '''
        self.target_network.set_weights(self.q_network.get_weights())

#-------------------- MEMORY --------------------------

class Memory(object):  # sample stored as (s, a, r, s_, done)

    def __init__(self, capacity):
        self.capacity = capacity
        self.samples = deque(maxlen=capacity)

    def push(self, sample):
        '''
            Append a new event to the memory
        '''
        self.samples.append(sample)

    def get(self, n):
        '''
            Get n samples randomly
        '''
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def isFull(self):
        return len(self.samples) >= self.capacity

#-------------------- DQN AGENT -----------------------

# DEFAULT HYPERPARAMETERS

#BATCH_SIZE = 32
#MEMORY_CAPACITY = 2000
#GAMMA = 0.9 # Discount Factor
#LEARNING_RATE = 0.001
#
#EPSILON_MAX = 1 # Exploration Rate
#EPSILON_MIN = 0.1
#EPSILON_DECAY = 0.995
# EXPLORATION_STEPS = 10000  # Number of steps over which initial value of epsilon is reduced to its final value for training
# EPSILON_STEPS = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
# UPDATE_TARGET_STEPS = 1000 # 500 # 1 / TAU # every 1000 now # before was 100, too fast

class FullDqnAgent(object):

    def __init__(self, inputCnt, actionCnt, batch_size=32, mem_capacity=2000, gamma=0.9,
                 lr=0.001, epsilon_max=1, epsilon_min=0.1, exploration_steps=10000,
                 update_target_steps=1000, brain_file=""):

        # Hyperparameters
        self.batch_size = batch_size
        self.mem_capacity = mem_capacity
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.exploration_steps = exploration_steps
        self.epsilon_steps = (epsilon_max - epsilon_min) / exploration_steps
        self.update_target_steps = update_target_steps

        self.inputCnt = inputCnt
        self.actionCnt = actionCnt

        # Create Experience Replay
        self.memory = Memory(mem_capacity)

        # Instantiate the model
        self.model = Network(inputCnt, actionCnt)

        # Build Q-network and Target-network
        self.model.q_network = self.build_model()
        self.model.target_network = self.build_model()


        self.brain_file = brain_file
        self.printStopExploration = False

        self.training_iterations = 0

    def build_model(self):
        raise NotImplementedError("Build model method not implemented")

    # Epsilon greedy action-selection policy from now
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.actionCnt)

        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    def record(self, sample):
        '''
            Add sample to memory
        '''
        self.memory.push(sample)

    def preprocess(self, state):
        # Input shape in Keras : (batch_size, input_dim)
        return np.reshape(state, [1, self.inputCnt])  # need to add 1 dimension for batch

    def replay(self):
        # If not enough sample in memory
        if len(self.memory.samples) < self.batch_size:
            return

        # if len(self.memory.samples) <= 100:
        #     return

        # Old technique with for loop -> slow
        # batch = self.memory.sample(self.batch_size)
        #
        # x_batch, y_batch = [], []
        #
        # for state, action, reward, next_state, done in batch:
        #     target = reward
        #     if not done:
        #         q_values = self.model.predict(next_state)[0]
        #         target = reward + self.gamma * np.amax(q_values)
        #
        #     labels = self.model.predict(state)
        #     labels[0][action] = target  # only the one action passed in the sample will have the actual target
        #
        #     x_batch.append(state[0])  # remove fake dimension
        #     y_batch.append(labels[0])
        #
        # x_batch = np.array(x_batch)  # convert list to array -> automatically add dimension
        # y_batch = np.array(y_batch)
        # self.model.train(x_batch, y_batch, batch_len=self.batch_size)


        #  ----------------- Optimized -----------------------------------------

        batch = self.memory.get(self.batch_size)
        batch = zip(*batch)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = batch

        batch_state = np.array(batch_state).squeeze(axis=1)
        q_values = self.model.predict(batch_state, batch_len=self.batch_size)

        batch_next_state = np.array(batch_next_state).squeeze(axis=1)
        q_values_target = self.model.predict(batch_next_state, batch_len=self.batch_size, target=True)

        batch_reward = np.array(batch_reward)

        # Optimization -> Matrix form
        X = batch_state
        labels = batch_reward + self.gamma * np.amax(q_values_target, axis=1)

        for i in xrange(self.batch_size):
            a = batch_action[i]

            if batch_done[i]: # 'Done' state
                q_values[i][a] = batch_reward[i]
            else:
                q_values[i][a] = labels[i]

        Y = q_values
        self.model.train(X, Y, batch_len=self.batch_size)

        # ----------------------------------------------------------------------


        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_steps
        else:
            if not self.printStopExploration:
                print("finished exploration")
                self.printStopExploration = True

        # Increment training iterations counter
        self.training_iterations += 1

        # Update target network
        if self.training_iterations != 0 and self.training_iterations % self.update_target_steps == 0:
            # print("update target network at it: {}".format(self.training_it))
            self.model.updateTargetNetwork()

        return

    def save_model(self, brainfile):
        self.model.q_network.save(brainfile)

    def load_model(self, model_file):
        """
            Load model from file and set Q-Network, Target-Network
        """
        print("loading model file: {}".format(model_file))
        self.model.q_network = load_model(model_file)
        self.model.target_network_network = load_model(model_file)

    def load_full_weights(self, model_file):
        """
            Load weights from file and set Q-Network, Target-Network
            Default: Stop training
        """
        print("loading full-weights from model file: {}".format(model_file))
        self.model.q_network.load_weights(model_file)
        self.model.target_network.load_weights(model_file)
