import numpy as np
import random
from collections import deque
from keras.models import load_model


# -------------------- NETWORK -------------------------

class Network(object):
    def __init__(self, inputCnt, actionCnt):
        self.inputCnt = inputCnt
        self.actionCnt = actionCnt

        self.model = None

    def train(self, x, y, nb_epochs=1, verbose=0, batch_len=1):
        """
            Fit the model
        """
        #self.model.fit(x, y, epochs=nb_epochs, verbose=verbose, batch_size=batch_len)
        self.model.fit(x, y, batch_size=batch_len, nb_epoch=nb_epochs, verbose=verbose) # keras 1.2.2

    def predict(self, state, batch_len=1):
        """
            Predict q-values given an input state
        """
        return self.model.predict(state, batch_size=batch_len)


# -------------------- MEMORY --------------------------

class Memory(object):  # sample stored as (s, a, r, s_, done)

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
# BATCH_SIZE = 32
# MEMORY_CAPACITY = 2000
# GAMMA = 0.9 # Discount Factor
# LEARNING_RATE = 0.001
# EPSILON_MAX = 1 # Exploration Rate
# EPSILON_MIN = 0.01
# EPSILON_DECAY = 0.995

INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
EXPLORATION_STEPS = 1000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value

isPrinted = False

class Dqn(object):
    def __init__(self, inputCnt, actionCnt, batch_size=32, mem_capacity=2000, gamma=0.9,
                 lr=0.001, epsilon_max=1, epsilon_min=0.001, epsilon_decay=0.995,
                 brain_file=""):

        # Hyperparameters
        self.batch_size = batch_size
        self.mem_capacity = mem_capacity
        self.gamma = gamma
        self.lr = lr
        # self.epsilon = epsilon_max
        # self.epsilon_min = epsilon_min
        # self.epsilon_decay = epsilon_decay

        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS

        self.inputCnt = inputCnt
        self.actionCnt = actionCnt

        self.model = Network(inputCnt, actionCnt)
        self.memory = Memory(mem_capacity)
        self.brain_file = brain_file

        self.model.model = self.build_model()

        self.last_state = self.preprocess(np.zeros(self.inputCnt))
        self.last_action = 0
        self.last_reward = 0

        # Dummy variables
        self.zeros_state = np.zeros([1, self.inputCnt])
        self.zeros_x = np.zeros((1, self.inputCnt))
        self.zeros_y = np.zeros((1, self.actionCnt))

        # Dummy Neural Network Processing, to avoid the freeze at the beginning of training
        dummy = self.model.predict(self.zeros_state)
        self.model.train(self.zeros_x, self.zeros_y)
        
    def build_model(self):
        raise NotImplementedError("Build model method not implemented")

    def update(self, reward, signal):
        """
            Core function of the agent's brain
            Return the action to be performed
        """
        global isPrinted

        new_state = signal
        new_state = self.preprocess(new_state)
        sample = (self.last_state, self.last_action, self.last_reward, new_state)
        self.record(sample)
        action = self.select_action(self.last_state)
        # if len(self.memory.samples) > 100:
        #     if not isPrinted:
        #         print('time to learn')
        #         isPrinted = True
        #     self.replay()
        # else:
        #     #a = self.model.model.get_weights()
        #     # Dummy Neural Network Processing, to avoid the freeze at the beginning of training
        #     dummy = self.model.predict(self.zeros_state)
        #     self.model.train(self.zeros_x, self.zeros_y)

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        return action

    # Epsilon greedy action-selection policy from now
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.actionCnt)

        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    def record(self, sample):
        """
            Add sample to memory
        """
        self.memory.push(sample)

    def replay(self):
        # If not enough sample in memory
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


    # def replay(self):
    #     # If not enough sample in memory
    #     if len(self.memory.samples) < self.batch_size:
    #         return
    #
    #     # batch = self.memory.sample(self.batch_size)
    #
    #     batchLen = self.batch_size
    #
    #     states = np.array([o[0] for o in batch]).squeeze(axis=1)
    #     labels = self.model.predict(states, batch_len=batchLen)
    #
    #     # next_states = np.array([o[3] for o in batch]).squeeze(axis=1)
    #     # q_values = self.model.predict(next_states, batch_len=batchLen)
    #     #
    #     # x = np.zeros((batchLen, self.inputCnt))
    #     # y = np.zeros((batchLen, self.actionCnt))
    #     # for i in xrange(batchLen):
    #     #     o = batch[i]
    #     #     s = o[0]
    #     #     a = o[1]
    #     #     r = o[2]
    #     #
    #     #     target = r + self.gamma * np.amax(q_values[i])
    #     #     labels[i][a] = target
    #     #
    #     #     x[i] = s
    #     #     y[i] = labels[i]
    #     #
    #     # self.model.train(x, y, batch_len=batchLen)
    #
    #     if self.epsilon > FINAL_EPSILON:
    #         self.epsilon -= self.epsilon_step
    #
    #         # if self.epsilon > self.epsilon_min:
    #         #     self.epsilon *= self.epsilon_decay

    def preprocess(self, state):
        # Input shape in Keras : (batch_size, input_dim)
        return np.reshape(state, [1, self.inputCnt])  # need to add 1 dimension for batch

    def save_model(self, brainfile):
        self.model.model.save(brainfile)
