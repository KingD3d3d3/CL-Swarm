from __future__ import division
import numpy as np
import random
from collections import deque
import sys
from keras.models import load_model, clone_model
import csv
import pandas as pd
import sklearn.utils
from keras.optimizers import Adam
try:
    from res.print_colors import *
    import Global
except:
    from ..res.print_colors import *
    from .. import Global

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
        # self.q_network.fit(x, y, batch_size=batch_len, nb_epoch=nb_epochs, verbose=verbose)  # keras 1.2.2
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

    def set_h1_weights(self, weights):
        """
            Set lower layers weights of Q-Network and Target-Network
        """
        self.q_network.layers[0].set_weights(weights)
        self.target_network.layers[0].set_weights(weights)

    def set_h1h2_weights(self, weights_h1, weights_h2):
        """
            Set lower layers weights of Q-Network and Target-Network
        """
        self.q_network.layers[0].set_weights(weights_h1)
        self.q_network.layers[1].set_weights(weights_h2)
        self.target_network.layers[0].set_weights(weights_h1)
        self.target_network.layers[1].set_weights(weights_h2)

    def shuffle_weights(self):
        """
            Randomly permute the weights in `model`, or the given `weights`.
            This is a fast approximation of re-initializing the weights of a model.
            Assumes weights are distributed independently of the dimensions of the weight tensors
            (i.e., the weights have the same distribution along each dimension).
            Credit: https://gist.github.com/jkleint/eb6dc49c861a1c21b612b568dd188668
        """
        # Get weights
        q_weights = self.q_network.get_weights()
        t_weights = self.target_network.get_weights()

        # Randomize
        q_weights = [np.random.permutation(w.flat).reshape(w.shape) for w in q_weights]
        t_weights = [np.random.permutation(w.flat).reshape(w.shape) for w in t_weights]

        # Apply to networks
        self.q_network.set_weights(q_weights)
        self.target_network.set_weights(t_weights)

# -------------------- MEMORY --------------------------

class Memory(object):  # sample stored as (s, a, r, s_, done)

    def __init__(self, capacity):
        self.capacity = capacity # max capacity of container
        self.samples = deque(maxlen=capacity) # container of experiences (queue)

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

    def is_full(self):
        """
            Return True if the container is full
            Else False
        """
        return len(self.samples) >= self.capacity

    def size(self):
        return len(self.samples)

# -------------------- DQN AGENT -----------------------

# DEFAULT HYPERPARAMETERS
BATCH_SIZE = 32
# B is typically chosen between 1 and a few hundreds, e.g. B = 32 is a good default value, with values above 10 taking advantage of the speed-up of matrix-matrix products over matrix-vector products. (from Bengio's 2012 paper)
MEMORY_CAPACITY = 10000  # 2000
GAMMA = 0.9  # Discount Factor
LEARNING_RATE = 0.001 # 0.01
TAU = 0.01  # 0.001 # update target network rate
UPDATE_TARGET_TIMESTEP = 1 / TAU # every 100
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
EXPLORATION_STEPS = 10000 # 10000  # 1000  # Number of steps over which initial value of epsilon is reduced to its final value

EPSILON_STEPS = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS

class DQN(object):
    def __init__(self, inputCnt, actionCnt, brain_file="", id=-1, ratio_update=1, training=True, random_agent=False):

        # Agent's ID
        self.id = id

        # Hyperparameters
        self.batch_size = BATCH_SIZE
        self.mem_capacity = MEMORY_CAPACITY
        self.gamma = GAMMA
        self.lr = LEARNING_RATE

        self.epsilon = INITIAL_EPSILON

        # Input - Output
        self.inputCnt = inputCnt
        self.actionCnt = actionCnt

        # Instantiate the model
        self.model = Model(inputCnt, actionCnt)

        # Create Experience Replay
        self.memory = Memory(MEMORY_CAPACITY)

        # Save file of the model
        self.model_file = brain_file

        # Build Q-network and Target-network
        self.model.q_network = self.build_model()
        self.model.target_network = self.build_model()

        self.last_state = self.preprocess(np.zeros(inputCnt))
        self.last_action = 0
        self.last_reward = 0.0
        self.reward_window = deque(maxlen=1000)

        # Dummy Neural Network Processing, to avoid the freeze at the beginning of training
        self.zeros_state = np.zeros([1, inputCnt])
        self.zeros_x = np.zeros((1, inputCnt))
        self.zeros_y = np.zeros((1, actionCnt))
        self.model.predict(self.zeros_state)
        self.model.predict(self.zeros_state, target=True)
        self.model.train(self.zeros_x, self.zeros_y)

        # Count the number of training iterations
        self.training_iterations = 0

        # Number of iterations for learning update
        self.ratio_update = ratio_update  # Default 1 means the agent learns every timestep
        self.update_counter = 0

        self.printTimeToLearn = False
        self.printStopExploration = False

        # Record new experience every timestep
        self.collect_experiences = True

        self.training = True # init flag
        if not training:
            self.stop_training()

        self.random_agent = False  # init flag
        if random_agent:
            self.go_random_agent()

    def build_model(self):
        raise NotImplementedError("Build model method not implemented")

    def update(self, reward, observation):
        """
            Main function of the agent's brain
            Return the action to be performed
        """
        new_state = self.preprocess(observation)
        experience = (self.last_state, self.last_action, self.last_reward, new_state)

        if self.collect_experiences:
            self.record(experience)

        # Select action
        action = self.select_action(self.last_state)

        # Training each update
        if self.training and len(self.memory.samples) >= 100 and self.update_counter % self.ratio_update == 0:
            self.replay()
            self.update_epsilon()
            self.reward_window.append(reward)

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward

        self.update_counter += 1

        # Update target network
        if self.training and self.training_iterations != 0 and self.training_iterations % UPDATE_TARGET_TIMESTEP == 0:
            self.model.updateTargetNetwork()

        return action

    # Epsilon greedy action-selection policy
    def select_action(self, state):

        # Random agent
        if self.random_agent:
            random_action = np.random.randint(0, self.actionCnt)
            return random_action

        # Epsilon greedy
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

        if not self.printTimeToLearn:
            printColor(color=PRINT_CYAN,
                       msg="Agent: {:3.0f}, ".format(self.id) +
                           "{:>25s}".format("time to learn") +
                           ", tmstp: {:10.0f}".format(Global.timestep) +
                           ", training_it: {:10.0f}".format(self.training_iterations) +
                           ", t: {}".format(Global.get_time()))
            self.printTimeToLearn = True

        batch = self.memory.pop(self.batch_size)
        batch = zip(*batch)
        batch_state, batch_action, batch_reward, batch_next_state = batch

        batch_state = np.array(batch_state).squeeze(axis=1)
        labels = self.model.predict(batch_state, batch_len=self.batch_size)

        batch_next_state = np.array(batch_next_state).squeeze(axis=1)
        q_values_t = self.model.predict(batch_next_state, batch_len=self.batch_size, target=True)

        batch_reward = np.array(batch_reward)

        # Optimization code - Matrix form
        X = batch_state
        target = batch_reward + self.gamma * np.amax(q_values_t, axis=1)
        for i in xrange(self.batch_size):
            a = batch_action[i]
            labels[i][a] = target[i]
        Y = labels

        self.model.train(X, Y, batch_len=self.batch_size)

        # Increment training iterations counter
        self.training_iterations += 1

    def update_epsilon(self):

        # If not training then return
        if not self.training:
            return

        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= EPSILON_STEPS

            # Reached final epsilon
            if self.epsilon <= FINAL_EPSILON:
                if not self.printStopExploration:
                    printColor(color=PRINT_CYAN,
                               msg="Agent: {:3.0f}, ".format(self.id) +
                                   "{:>25s}".format("finished exploration") +
                                   ", tmstp: {:10.0f}".format(Global.timestep) +
                                   ", training_it: {:10.0f}".format(self.training_iterations) +
                                   ", t: {}".format(Global.get_time()))
                    self.printStopExploration = True


    def preprocess(self, state):
        # Input shape in Keras : (batch_size, input_dim)
        return np.reshape(state, [1, self.inputCnt])  # need to add 1 dimension for batch

    def learning_score(self):
        """
            Score is the mean of the reward in the sliding window
        """
        learning_score = sum(self.reward_window) / (len(self.reward_window) + 1.)  # +1 to avoid division by zero
        return learning_score

    def stop_training(self):
        """
            Stop training the Neural Network
            Stop exploration -> only exploitation
        """
        self.training = False
        self.epsilon = FINAL_EPSILON

        printColor(color=PRINT_CYAN,
                   msg="Agent: {:3.0f}, ".format(self.id) +
                       "{:>25s}".format("Stop training, Stop exploring") +
                       ", tmstp: {:10.0f}".format(Global.timestep) +
                       ", training_it: {:10.0f}".format(self.training_iterations) +
                       ", t: {}".format(Global.get_time()))

    def stop_exploring(self):
        """
            Stop exploration -> only exploitation
        """
        self.epsilon = FINAL_EPSILON

        printColor(color=PRINT_CYAN,
                   msg="Agent: {:3.0f}, ".format(self.id) +
                       "{:>25s}".format("Stop exploring") +
                       ", tmstp: {:10.0f}".format(Global.timestep) +
                       ", training_it: {:10.0f}".format(self.training_iterations) +
                       ", t: {}".format(Global.get_time()))

    def go_random_agent(self):
        # Random action
        self.random_agent = True
        if self.random_agent:
            printColor(color=PRINT_CYAN,
                       msg="Agent: {:3.0f}, ".format(self.id) +
                           "{:>25s}".format("Random agent") +
                           ", tmstp: {:10.0f}".format(Global.timestep) +
                           ", training_it: {:10.0f}".format(self.training_iterations) +
                           ", t: {}".format(Global.get_time()))
            if self.training:
                self.stop_training()

    def stop_collect_experiences(self):
        """
            Stop appending new experience to memory
        """
        self.collect_experiences = False

        printColor(color=PRINT_CYAN,
                   msg="Agent: {:3.0f}, ".format(self.id) +
                       "{:>25s}".format("Stop collecting experiences") +
                       ", tmstp: {:10.0f}".format(Global.timestep) +
                       ", training_it: {:10.0f}".format(self.training_iterations) +
                       ", t: {}".format(Global.get_time()))

    def save_model(self, model_file):
        """
            Save model (neural network, optimizer, loss, etc ..) in file
        """
        self.model.q_network.save(model_file)

        printColor(color=PRINT_CYAN,
                   msg="Agent: {:3.0f}, ".format(self.id) +
                       "{:>25s}".format("Save Agent's model") +
                       ", file: {}".format(model_file) +
                       ", tmstp: {:10.0f}".format(Global.timestep) +
                       ", training_it: {:10.0f}".format(self.training_iterations) +
                       ", t: {}".format(Global.get_time()))

    def save_memory(self, memory_file):
        """
            Save agent's experiences in file
        """
        header = ("state", "action", "reward", "next_state")

        with open(memory_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.memory.samples)

        printColor(color=PRINT_CYAN,
                   msg="Agent: {:3.0f}, ".format(self.id) +
                       "{:>25s}".format("Save Agent's memory") +
                       ", file: {}".format(memory_file) +
                       ", tmstp: {:10.0f}".format(Global.timestep) +
                       ", training_it: {:10.0f}".format(self.training_iterations) +
                       ", t: {}".format(Global.get_time()))

    def load_model(self, model_file):
        """
            Load model from file and set Q-Network, Target-Network
            Default: Stop training
        """
        self.model.q_network = load_model(model_file)
        self.model.target_network_network = load_model(model_file)

        printColor(color=PRINT_CYAN,
                   msg="Agent: {:3.0f}, ".format(self.id) +
                       "{:>25s}".format("Load full model") +
                       ", file: {}".format(model_file) +
                       ", tmstp: {:10.0f}".format(Global.timestep) +
                       ", training_it: {:10.0f}".format(self.training_iterations) +
                       ", t: {}".format(Global.get_time()))

    def load_full_weights(self, model_file):
        """
            Load weights from file and set Q-Network, Target-Network
            Default: Stop training
        """
        self.model.q_network.load_weights(model_file)
        self.model.target_network.load_weights(model_file)

        printColor(color=PRINT_CYAN,
                   msg="Agent: {:3.0f}, ".format(self.id) +
                       "{:>25s}".format("Load model full weights") +
                       ", file: {}".format(model_file) +
                       ", tmstp: {:10.0f}".format(Global.timestep) +
                       ", training_it: {:10.0f}".format(self.training_iterations) +
                       ", t: {}".format(Global.get_time()))

    def load_h1_weights(self, model_file):
        """
            Load lower layers' weights from file and set Q-Network, Target-Network
            Default: Stop training
        """
        model_copy = clone_model(self.model.q_network)
        model_copy.load_weights(model_file)

        weights = model_copy.layers[0].get_weights()

        self.model.set_h1_weights(weights)

        if np.array_equal(self.model.q_network.layers[0].get_weights(), weights):
            sys.exit('Error! Q-Network lower layer is not equal to the lower layer weights from file')

        printColor(color=PRINT_CYAN,
                   msg="Agent: {:3.0f}, ".format(self.id) +
                       "{:>25s}".format("Load 1st hidden layer weights") +
                       ", file: {}".format(model_file) +
                       ", tmstp: {:10.0f}".format(Global.timestep) +
                       ", training_it: {:10.0f}".format(self.training_iterations) +
                       ", t: {}".format(Global.get_time()))

    def load_h1h2_weights(self, model_file):
        """
            Load lower layers' weights from file and set Q-Network, Target-Network
            Default: Stop training
        """
        model_copy = clone_model(self.model.q_network)
        model_copy.load_weights(model_file)

        weights_h1 = model_copy.layers[0].get_weights()
        weights_h2 = model_copy.layers[1].get_weights()

        self.model.set_h1h2_weights(weights_h1, weights_h2)

        # if np.array_equal(self.model.q_network.layers[0].get_weights(), weights):
        #     sys.exit('Error! Q-Network lower layer is not equal to the lower layer weights from file')

        printColor(color=PRINT_CYAN,
                   msg="Agent: {:3.0f}, ".format(self.id) +
                       "{:>25s}".format("Load 1st and 2nd hidden layer weights") +
                       ", file: {}".format(model_file) +
                       ", tmstp: {:10.0f}".format(Global.timestep) +
                       ", training_it: {:10.0f}".format(self.training_iterations) +
                       ", t: {}".format(Global.get_time()))

    def load_memory(self, memory_file, size=MEMORY_CAPACITY):
        """
            Load memory from file
            Default size = -1 means load full memory from the file
        """
        experiences = []
        data = pd.read_csv(memory_file)

        remove_bracket = lambda x: x.replace('[', '').replace(']', '')
        string_to_array = lambda x: np.expand_dims(np.fromstring(x, sep=' '), axis=0)

        data['state'] = data['state'].map(remove_bracket).map(string_to_array)
        data['next_state'] = data['next_state'].map(remove_bracket).map(string_to_array)

        if size == MEMORY_CAPACITY:
            """
                Load full memory
            """
            for i, row in data.iterrows():
                exp = (row['state'], row['action'], row['reward'], row['next_state'])
                experiences.append(exp)
        else:
            """
                Load specified number of experiences
            """
            shuffled_data = sklearn.utils.shuffle(data)
            #shuffled_data = shuffled_data[0:size]
            for i, row in shuffled_data[0:size].iterrows():
                exp = (row['state'], row['action'], row['reward'], row['next_state'])
                experiences.append(exp)
        self.memory.receive(experiences)

        printColor(color=PRINT_CYAN,
                   msg="Agent: {:3.0f}, ".format(self.id) +
                       "{:>25s}".format("Load memory: {} exp".format(size)) +
                       ", file: {}".format(memory_file) +
                       ", tmstp: {:10.0f}".format(Global.timestep) +
                       ", training_it: {:10.0f}".format(self.training_iterations) +
                       ", t: {}".format(Global.get_time()))