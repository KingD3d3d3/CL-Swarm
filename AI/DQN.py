import numpy as np
import random
from collections import deque
import sys
from keras.models import load_model, clone_model

try:
    from res.print_colors import *
except:
    from ..res.print_colors import *


# -------------------- MODEL -------------------------

class Model(object):
    def __init__(self, inputCnt, actionCnt):
        self.inputCnt = inputCnt
        self.actionCnt = actionCnt

        # Q-Network
        self.q_network = None

        # Target Network
        self.target_network = None

    def train(self, x, y, nb_epochs=1, verbose=0, batch_len=1):
        """
            Fit the model
        """
        #self.q_network.fit(x, y, batch_size=batch_len, nb_epoch=nb_epochs, verbose=verbose)  # keras 1.2.2
        self.q_network.fit(x, y, batch_size=batch_len, epochs=nb_epochs, verbose=verbose)  # keras 2

    def predict(self, state, batch_len=1, target=False):
        """
            Predict q-values given an input state
        """
        if target:
            return self.q_network.predict(state, batch_size=batch_len)
        else:
            return self.target_network.predict(state, batch_size=batch_len)

    def updateTargetNetwork(self):
        """
            Update Target-Network : copy Q-Network weights into Target-Network
        """
        self.target_network.set_weights(self.q_network.get_weights())

    def get_lower_layers_weights(self):
        """
            Get lower layers weights of Q-Network
        """
        return self.q_network.layers[0].get_weights()

    def set_lower_layers_weights(self, weights):
        """
            Set lower layers weights of Q-Network and Target-Network
        """
        self.q_network.layers[0].set_weights(weights)
        self.target_network.layers[0].set_weights(weights)


# -------------------- MEMORY --------------------------

class Memory(object):  # sample stored as (s, a, r, s_, done)

    def __init__(self, capacity):
        self.capacity = capacity
        self.samples = deque(maxlen=capacity)

    def push(self, sample):
        """
            Append a new sample to the memory
        """
        self.samples.append(sample)  # add only one event

    def pop(self, n):
        """
            Get n samples randomly
        """
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def receive(self, samples):
        """
            Samples : a list of samples
            Receive samples and append them to the memory
        """
        for sp in samples:
            self.push(sp)
            # self.samples.extend(sample)
            #
            # def save(self):
            #     """
            #         Get n samples randomly
            #     """
            #     n = min(n, len(self.samples))
            #     return random.sample(self.samples, n)


# -------------------- DQN AGENT -----------------------

# DEFAULT HYPERPARAMETERS
BATCH_SIZE = 32
MEMORY_CAPACITY = 2000
GAMMA = 0.9  # Discount Factor
LEARNING_RATE = 0.001
TAU = 0.01  # 0.001 # update target network rate
UPDATE_TARGET_TIMESTEP = 1 / TAU
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
EXPLORATION_STEPS = 10000  # 1000  # Number of steps over which initial value of epsilon is reduced to its final value

printTimeToLearn = False


class DQN(object):
    def __init__(self, inputCnt, actionCnt, brain_file="", id=-1, training=True):

        # Agent's ID
        self.id = id

        # Training flag
        self.training = training

        # Hyperparameters
        self.batch_size = BATCH_SIZE
        self.mem_capacity = MEMORY_CAPACITY
        self.gamma = GAMMA
        self.lr = LEARNING_RATE
        if self.training:
            self.epsilon = INITIAL_EPSILON
        else:
            self.epsilon = FINAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS

        # Input - Output
        self.inputCnt = inputCnt
        self.actionCnt = actionCnt

        # Instantiate the model
        self.model = Model(inputCnt, actionCnt)

        # Create Experience Replay
        self.memory = Memory(self.mem_capacity)

        # Save file of the model
        self.model_file = brain_file

        # Build Q-network and Target-network
        self.model.q_network = self.build_model()
        self.model.target_network = self.build_model()

        self.last_state = self.preprocess(np.zeros(self.inputCnt))
        self.last_action = 0
        self.last_reward = 0.0
        self.reward_window = deque(maxlen=1000)

        # Dummy Neural Network Processing, to avoid the freeze at the beginning of training
        self.zeros_state = np.zeros([1, self.inputCnt])
        self.zeros_x = np.zeros((1, self.inputCnt))
        self.zeros_y = np.zeros((1, self.actionCnt))
        dummy = self.model.predict(self.zeros_state)
        dummy2 = self.model.predict(self.zeros_state, target=True)
        self.model.train(self.zeros_x, self.zeros_y)

        # Count the number of iterations
        self.steps = 0

    def build_model(self):
        raise NotImplementedError("Build model method not implemented")

    def update(self, reward, observation):
        """
            Main function of the agent's brain
            Return the action to be performed
        """
        global printTimeToLearn

        new_state = self.preprocess(observation)
        experience = (self.last_state, self.last_action, self.last_reward, new_state)
        self.record(experience)

        # Select action
        action = self.select_action(self.last_state)

        # Training each update
        if self.training and len(self.memory.samples) > 100:
            if not printTimeToLearn:
                printColor(msg="Agent: {:3.0f}".format(self.id) + "{:>28s}".format("time to learn"))
                printTimeToLearn = True
            self.replay()

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)

        self.steps += 1

        # Update target network
        if self.steps % UPDATE_TARGET_TIMESTEP == 0:
            self.model.updateTargetNetwork()

        return action

    # Epsilon greedy action-selection policy
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
        # If not training then return
        if not self.training:
            return

        # If not enough samples in memory
        if len(self.memory.samples) < self.batch_size:
            return

        batch = self.memory.pop(self.batch_size)
        batch = zip(*batch)
        batch_state, batch_action, batch_reward, batch_next_state = batch

        batch_state = np.array(batch_state).squeeze(axis=1)
        labels = self.model.predict(batch_state, batch_len=self.batch_size)

        batch_next_state = np.array(batch_next_state).squeeze(axis=1)
        q_values_t = self.model.predict(batch_next_state, batch_len=self.batch_size, target=True)

        batch_action = np.array(batch_action)
        batch_reward = np.array(batch_reward)

        x = np.zeros((self.batch_size, self.inputCnt))
        y = np.zeros((self.batch_size, self.actionCnt))
        for i in xrange(self.batch_size):
            s = batch_state[i]
            a = batch_action[i]
            r = batch_reward[i]

            target = r + self.gamma * np.amax(q_values_t[i])
            labels[i][a] = target

            x[i] = s
            y[i] = labels[i]

        self.model.train(x, y, batch_len=self.batch_size)

        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= self.epsilon_step

    def preprocess(self, state):
        # Input shape in Keras : (batch_size, input_dim)
        return np.reshape(state, [1, self.inputCnt])  # need to add 1 dimension for batch

    def learning_score(self):
        """
            Score is the mean of the reward in the sliding window
        """
        learning_score = sum(self.reward_window) / (len(self.reward_window) + 1.)  # +1 to avoid division by zero
        return learning_score

    def save_model(self, model_file):
        """
            Save model : neural network, optimizer, loss, etc in 'model_file'
        """
        self.model.q_network.save(model_file)

        #
        # def save_memory(self, memory_file):
        #     self.memory.save(memory_file)

    def load_model(self, model_file):
        """
            Load model from 'model_file' and set Q-Network, Target-Network
            Default: Stop training
        """
        self.model.q_network = load_model(model_file)
        self.model.target_network_network = load_model(model_file)

    def load_weights(self, model_file):
        """
            Load weights from 'model_file' and set Q-Network, Target-Network
            Default: Stop training
        """
        self.model.q_network.load_weights(model_file)
        self.model.target_network.load_weights(model_file)

    def load_lower_layers_weights(self, model_file):
        """
            Load lower layers' weights from 'model_file' and set Q-Network, Target-Network
            Default: Stop training
        """

        model_copy = clone_model(self.model.q_network)
        model_copy.load_weights(model_file)

        weights = model_copy.layers[0].get_weights()

        self.model.set_lower_layers_weights(weights)

        if np.array_equal(self.model.q_network.layers[0].get_weights(), weights):
            sys.exit('Error! Q-Network lower layer is not equal to the lower layer weights from file')

    def stop_training(self):
        """
            Stop training the Neural Network
            Stop exploration -> only exploitation
        """
        self.training = False
        self.epsilon = FINAL_EPSILON
