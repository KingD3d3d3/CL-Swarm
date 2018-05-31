from __future__ import division
import numpy as np
import random
from collections import deque


# -------------------- MODEL -------------------------

class Model(object):
    def __init__(self, inputCnt, actionCnt):
        self.inputCnt = inputCnt
        self.actionCnt = actionCnt

        # Q-Network
        self.q_network = None

    def train(self, x, y, nb_epochs=1, verbose=0, batch_len=1):
        """
            Fit the model
        """
        #self.model.fit(x, y, epochs=nb_epochs, verbose=verbose, batch_size=batch_len)
        self.q_network.fit(x, y, batch_size=batch_len, nb_epoch=nb_epochs, verbose=verbose) # keras 1.2.2

    def predict(self, state, batch_len=1):
        """
            Predict q-values given an input state
        """
        return self.q_network.predict(state, batch_size=batch_len)


# -------------------- MEMORY --------------------------

class Memory(object):  # pop stored as (s, a, r, s_, done)

    def __init__(self, capacity):
        self.capacity = capacity
        self.samples = deque(maxlen=capacity)

    def push(self, sample):
        """
            Append a new event to the memory
        """
        self.samples.append(sample)

    def sample(self, n):
        """
            Get n samples randomly
        """
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)


# -------------------- DQN AGENT -----------------------

# DEFAULT HYPERPARAMETERS
BATCH_SIZE = 32
MEMORY_CAPACITY = 2000
GAMMA = 0.9 # Discount Factor
LEARNING_RATE = 0.001

INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
EXPLORATION_STEPS = 10000  #1000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value

printTimeToLearn = False

class SimpleDQN(object):
    def __init__(self, inputCnt, actionCnt, batch_size=BATCH_SIZE, mem_capacity=MEMORY_CAPACITY, gamma=GAMMA,
                 lr=LEARNING_RATE, brain_file=""):

        # Hyperparameters
        self.batch_size = batch_size
        self.mem_capacity = mem_capacity
        self.gamma = gamma
        self.lr = lr

        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS

        self.inputCnt = inputCnt
        self.actionCnt = actionCnt

        # Instantiate the model
        self.model = Model(inputCnt, actionCnt)

        # Create Experience Replay
        self.memory = Memory(mem_capacity)

        # Save file of the model
        self.brain_file = brain_file

        # Build Q-network
        self.model.q_network = self.build_model()

        self.last_state = self.preprocess(np.zeros(self.inputCnt))
        self.last_action = 0
        self.last_reward = 0
        self.reward_window = deque(maxlen=1000)

        # Dummy variables
        self.zeros_state = np.zeros([1, self.inputCnt])
        self.zeros_x = np.zeros((1, self.inputCnt))
        self.zeros_y = np.zeros((1, self.actionCnt))

        # Dummy Neural Network Processing, to avoid the freeze at the beginning of training
        dummy = self.model.predict(self.zeros_state)
        self.model.train(self.zeros_x, self.zeros_y)

        # Count the number of iterations
        self.steps = 0

    def build_model(self):
        raise NotImplementedError("Build model method not implemented")

    def update(self, reward, signal):
        """
            Core function of the agent's brain
            Return the action to be performed
        """
        global printTimeToLearn

        new_state = signal
        new_state = self.preprocess(new_state)
        sample = (self.last_state, self.last_action, self.last_reward, new_state)
        self.record(sample)
        action = self.select_action(self.last_state)
        # Training each update
        if len(self.memory.samples) > 100:
            if not isPrinted:
                print('time to learn')
                isPrinted = True
            self.replay()

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)

        self.steps += 1

        return action

    # Epsilon greedy action-selection policy from now
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.actionCnt)

        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    def record(self, sample):
        """
            Add pop to memory
        """
        self.memory.push(sample)

    def replay(self):
        # If not enough pop in memory
        if len(self.memory.samples) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        batch = zip(*batch)
        batch_state, batch_action, batch_reward, batch_next_state = batch

        batch_state = np.array(batch_state).squeeze(axis=1)
        labels = self.model.predict(batch_state, batch_len=self.batch_size)

        batch_next_state = np.array(batch_next_state).squeeze(axis=1)
        q_values = self.model.predict(batch_next_state, batch_len=self.batch_size)

        batch_action = np.array(batch_action)
        batch_reward = np.array(batch_reward)

        x = np.zeros((self.batch_size, self.inputCnt))
        y = np.zeros((self.batch_size, self.actionCnt))
        for i in xrange(self.batch_size):
            s = batch_state[i]
            a = batch_action[i]
            r = batch_reward[i]

            target = r + self.gamma * np.amax(q_values[i])
            labels[i][a] = target

            x[i] = s
            y[i] = labels[i]

        self.model.train(x, y, batch_len=self.batch_size)

        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= self.epsilon_step

    def preprocess(self, state):
        # Input shape in Keras : (batch_size, input_dim)
        return np.reshape(state, [1, self.inputCnt])  # need to add 1 dimension for batch

    def save_model(self, brainfile):
        self.model.q_network.save(brainfile)

    def score(self):
        """
            Score is the mean of the reward in the sliding window
        """
        score = sum(self.reward_window) / (len(self.reward_window) + 1.)  # +1 to avoid division by zero
        return score