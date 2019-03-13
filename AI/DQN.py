
import numpy as np
import random
from collections import deque
from keras.models import load_model, clone_model
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import pandas as pd
import sklearn.utils
from keras.optimizers import Adam
import os
import errno
import Global
import res.Util as Util
from res.print_colors import *

# DEFAULT HYPERPARAMETERS
H1 = 64  # number of neurons in the 1st hidden layer
H2 = 64  # number of neurons in the 2nd hidden layer
BATCH_SIZE = 32  # Typically chosen between 1 and a few hundreds, e.g. B = 32 is a good default value, with values above 10 taking advantage of the speed-up of matrix-matrix products over matrix-vector products. (from Bengio's 2012 paper)
MEMORY_CAPACITY = 100000  # 2000
GAMMA = 0.99  # Discount Factor
LEARNING_RATE = 0.001  # default Adam optimizer learning rate value
UPDATE_TARGET_STEPS = 1000  # update target network rate every given timesteps
EPSILON_START = 1.  # Initial value of epsilon in epsilon-greedy during training
EPSILON_END = 0.01  # Final value of epsilon in epsilon-greedy during training
EXPLORATION_STEPS = 10000  # 10000  # 1000  # Number of steps over which initial value of epsilon is reduced to its final value for training
EPSILON_TEST = 0  # epsilon value during testing after training is done

# class LossHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs=None):
#         self.losses = []
#
#     def on_batch_end(self, batch, logs=None):
#         self.losses.append(logs.get('loss'))

# -------------------- MODEL -------------------------

class Model(object):
    def __init__(self, input_size, output_size, layers=(H1, H2), lr=LEARNING_RATE):
        """
            :param input_size: size of the input layer
            :param output_size: size of the output layer
            :param layers: list of ints defining the size of each layer used in the model
            :param lr: learning rate
        """
        self.q_network = self.build_model(input_size=input_size, output_size=output_size, layers=layers, lr=lr)
        self.target_network = self.build_model(input_size=input_size, output_size=output_size, layers=layers, lr=lr)

        self.dummy_processing(input_size, output_size)  # dummy processing at init to prevent training freeze in runtime

        self.num_layers = len(layers)

    @staticmethod
    def build_model(input_size, output_size, layers, lr):
        """
            Build the neural network (NN) model
            :param input_size: size of the input layer
            :param output_size: size of the output layer
            :param layers: list of ints defining the size of each layer used in the model
            :param lr: learning rate
            :return: the NN model
        """
        model = Sequential()  # Sequential() creates the foundation of the layers.

        model.add(Dense(layers[0], activation='relu', input_dim=input_size))  # add input to 1st hidden layer

        for layer in layers[1:]:
            model.add(Dense(layer, activation='relu'))  # add successive hidden layers

        model.add(Dense(output_size, activation='linear'))  # Output layer
        model.compile(loss='mse', optimizer=Adam(lr=lr))  # Adam optimizer for stochastic gradient descent

        return model

    def train(self, x, y, nb_epochs=1, verbose=0, batch_len=BATCH_SIZE):
        """
            Fit the model using batch update
            :param x: array of training data
            :param y: array of target data
            :param nb_epochs: number of epochs to train the model
            :param verbose: verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            :param batch_len: number of samples per gradient update
        """
        # self.q_network.fit(x, y, batch_size=batch_len, nb_epoch=nb_epochs, verbose=verbose)  # keras 1.2.2
        self.q_network.fit(x, y, batch_size=batch_len, epochs=nb_epochs, verbose=verbose) # keras 2

    def predict(self, state, batch_len=1, target=False):
        """
            Predict q-values from NN given input state
            :param state: array of input state
            :param batch_len: number of samples to predict
            :param target: if yes predict from target-network, otherwise Q-network
            :return: Q-values
        """
        if target:
            # Predict from Target-network
            return self.target_network.predict(state, batch_size=batch_len)
        else:
            # Predict from Q-network
            return self.q_network.predict(state, batch_size=batch_len)

    def dummy_processing(self, input_size, output_size):
        """
            Dummy neural network processing to avoid the freeze when the training starts
            :param input_size: size of the input layer
            :param output_size: size of the output layer
        """
        zeros_state = np.zeros([1, input_size])
        zeros_x = np.zeros((1, input_size))
        zeros_y = np.zeros((1, output_size))

        self.predict(zeros_state)
        self.predict(zeros_state, target=True)
        self.train(zeros_x, zeros_y)

    def update_target_network(self):
        """
            Copy the Q-Network weights into the target-Network
        """
        self.target_network.set_weights(self.q_network.get_weights())

    def set_weights(self, weights):
        """
            Set given weights into the Q-network and target-network
            :param weights: weights use for replace
        """
        self.q_network.set_weights(weights)
        self.target_network.set_weights(weights)

    def set_weights_by_layer(self, weights, layer_num):
        """
            Set given weights into the specified layer index of Q-network and target-network
            :param weights: weights use for replace
            :param layer_num: layer index
        """
        self.q_network.layers[layer_num].set_weights(weights)
        self.target_network.layers[layer_num].set_weights(weights)

    def set_h1_weights(self, weights):
        """
            Set first hidden layer weights of Q-Network and target-Network
            :param weights: weights use for replace
        """
        self.set_weights_by_layer(weights, 0)

    def set_h2_weights(self, weights):
        """
            Set second hidden layer weights of Q-Network and target-Network
            :param weights: weights use for replace
        """
        self.set_weights_by_layer(weights, 1)

    def set_out_weights(self, weights):
        """
            Set output layer weights of Q-Network and target-Network
            :param weights: weights use for replace
        """
        if self.num_layers == 2:
            self.set_weights_by_layer(weights, 2)
        elif self.num_layers == 1:
            self.set_weights_by_layer(weights, 1)
        else:
            raise NotImplementedError()

    def set_h1h2_weights(self, weights_h1, weights_h2):
        """
            Set first and second hidden layer weights of Q-Network and target-Network
            :param weights_h1: first hidden layer weights
            :param weights_h2: second hidden layer weights
        """
        self.set_h1_weights(weights_h1)
        self.set_h2_weights(weights_h2)

    def set_h2out_weights(self, weights_h2, weights_output):
        """
            Set second hidden layer and output weights of Q-Network and target-Network
            :param weights_h2: second hidden layer weights
            :param weights_output: output layer weights
        """
        self.set_h2_weights(weights_h2)
        self.set_out_weights(weights_output)

    def set_h1out_weights(self, weights_h1, weights_output):
        """
            Set first hidden layer and output weights of Q-Network and target-Network
            :param weights_h1: first hidden layer weights
            :param weights_output: output layer weights
        """
        self.set_h1_weights(weights_h1)
        self.set_out_weights(weights_output)

    def shuffle_weights(self):
        """
            Randomly permute the weights in `model`, or the given `weights`.
            This is a fast approximation of re-initializing the weights of a model.
            Assumes weights are distributed independently of the dimensions of the weight tensors
            (i.e., the weights have the same distribution along each dimension).
            Credit: https://gist.github.com/jkleint/eb6dc49c861a1c21b612b568dd188668
        """
        weights = self.q_network.get_weights()  # Get weights
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]  # Randomize

        # Apply to networks
        self.q_network.set_weights(weights)
        self.target_network.set_weights(weights)


# -------------------- MEMORY --------------------------

class Memory(object):  # sample stored as (s, a, r, s_, done)

    def __init__(self, capacity, seed=None):
        """
            :param capacity: maximum capacity of the memory
            :param seed: seed for random
        """
        self.capacity = capacity
        self.samples = deque(maxlen=capacity)  # container of experiences (queue)

        if seed is not None:
            random.seed(seed)
        else:
            random.seed()

    def push(self, sample):
        """
            Append a new sample to the memory, remove the oldest sample if memory was full
            :param sample: sample to add
        """
        self.samples.append(sample)  # add only one event

    def push_multi(self, samples):
        """
            Append samples to the memory
            :param samples: list of samples
        """
        for sp in samples:
            self.push(sp)

    def get(self, n):
        """
            Get n samples randomly (uniform) from the memory
            :param n: number of samples to draw
            :return: the samples
        """
        return random.sample(self.samples, n)

    def is_full(self):
        """
            :return: true if the container is full, otherwise false
        """
        return len(self.samples) >= self.capacity

    def size(self):
        """
            :return: current size of the container
        """
        return len(self.samples)

# -------------------- DQN AGENT -----------------------

class DQN(object):
    def __init__(self, input_size, action_size, id=-1, training=True, random_agent=False,
                 layers=(H1, H2), mem_capacity=MEMORY_CAPACITY, batch_size=BATCH_SIZE, gamma=GAMMA, lr=LEARNING_RATE,
                 update_target_steps=UPDATE_TARGET_STEPS, eps_start=EPSILON_START, eps_end=EPSILON_END,
                 eps_test=EPSILON_TEST,
                 exploration_steps=EXPLORATION_STEPS, use_double_dqn=True, use_prioritized_experience_replay=False,
                 seed=None):
        """
            :param input_size: number of inputs
            :param action_size: number of actions
            :param id: id
            :param training: decide whether agent trains or not
            :param random_agent: yes -> agent performs random action, else choose action
            :param layers: list of ints defining the size of each layer used in the model
            :param mem_capacity: maximum memory capacity
            :param batch_size: batch size for gradient update
            :param gamma: discount factor
            :param lr: learning rate
            :param update_target_steps: number of timesteps before updating target-network
            :param eps_start: initial value of epsilon
            :param eps_end: final value of epsilon
            :param eps_test: epsilon value during testing mode (training off)
            :param exploration_steps: number of exploration steps
            :param use_double_dqn: yes to use double-DQN, else standard DQN (with target-network)
            :param use_prioritized_experience_replay: TODO implement Prioritized experience replay
            :param seed: seed for random state
        """
        if seed is not None:
            np.random.seed(seed)

            # TODO : still get different result, for now abandon feature of seed to get reproducible results
            # Seed Tensorflow's internal random generator
            # if K.backend() == 'tensorflow':
            #     from tensorflow import set_random_seed
            #     set_random_seed(seed)

        else:
            np.random.seed()
        self.seed = seed

        # Agent's ID
        self.id = id

        # Hyperparameters
        self.layers = layers
        self.num_layers = len(layers)
        self.batch_size = batch_size
        self.mem_capacity = mem_capacity
        self.gamma = gamma
        self.lr = lr
        self.update_target_steps = update_target_steps
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_test = eps_test
        self.use_double_dqn = use_double_dqn
        self.use_prioritized_experience_replay = use_prioritized_experience_replay

        # Epsilon for epsilon greedy algorithm
        self.eps = eps_start
        self.exploration_steps = exploration_steps
        self.eps_decay = (eps_start - eps_end) / exploration_steps

        # Input - Output
        self.input_size = input_size
        self.action_size = action_size

        # Create the model
        self.model = Model(input_size=input_size, output_size=action_size, layers=layers, lr=lr)

        # Create the memory Experience Replay
        self.memory = Memory(capacity=mem_capacity, seed=seed)

        # Count the number of training iterations
        self.training_it = 0

        # Number of iterations for learning update
        self.update_counter = 0

        # Flag to be used for printing learning event
        self.printTimeToLearn = False
        self.printStopExploration = False

        # Record new experience every timestep
        self.collect_experiences = True

        self.training = True  # init training flag
        if not training:
            self.stop_training()

        self.random_agent = False  # init random agent flag
        if random_agent:
            self.go_random_agent()

        # Print summary of the DQN config
        self.summary_config()

    def summary_config(self):
        print("_________________________________________________________________")
        print("*** Summary of DQN agent config ***\n")

        print("Input: {}".format(self.input_size))
        print("Action: {}".format(self.action_size))
        print("Training: {}".format(self.training))
        print("Random agent: {}".format(self.random_agent))
        print("Seed: {}".format(self.seed))

        print("\n----------------")
        print("Hyperparameters\n")

        print("layers: {}".format(self.layers))
        print("batch size: {}".format(self.batch_size))
        print("memory capacity: {}".format(self.mem_capacity))
        print("discount factor gamma: {}".format(self.gamma))
        print("learning rate: {}".format(self.lr))
        print("update target steps: {}".format(self.update_target_steps))
        print("initial epsilon: {}".format(self.eps_start))
        print("final epsilon: {}".format(self.eps_end))
        print("exploration steps: {}".format(self.exploration_steps))
        print("epsilon decay: {}".format(self.eps_decay))
        print("epsilon test: {}".format(self.eps_test))
        print("use double dqn: {}".format(self.use_double_dqn))
        print("use prioritized experience replay: {}".format(self.use_prioritized_experience_replay))

        print("\n-----------------------")
        print("Neural network summary")
        self.model.q_network.summary()

    def preprocess(self, observation):
        """
            Preprocess an observation: add 1 dimension for batch
            :param observation: observation to preprocess
            :return: processed observation
        """
        # Input shape in Keras : (batch dim, input_dim)
        a = np.reshape(observation, [1, self.input_size])  # need to add 1 dimension for batch
        a32 = a.astype(np.float32)
        return a32

    def select_action(self, state):
        """
            * Epsilon greedy action-selection policy
            * Random action when random agent
            * Most of the time select best action when testing
            :param state: input state
            :return: returned action for action-selection policy
        """
        # Random agent
        if self.random_agent:
            random_action = np.random.randint(0, self.action_size)
            return random_action

        # Epsilon greedy
        if np.random.rand() < self.eps:
            # print('choose random !')
            return np.random.randint(0, self.action_size)

        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    def record(self, sample):
        """
            Add sample to memory
            :param sample: sample to add
        """
        if self.collect_experiences:  # self.training
            # self.memory.push(sample)
            sample = sample + (None, None) # added None for Q-values
            self.memory.push(sample)

    def train(self):
        """
            Overall training function of DQN agent
        """
        # Training
        if self.training and len(
                self.memory.samples) >= self.batch_size:  # and self.update_counter % self.ratio_train == 0:
            self.replay()  # replay from memory
            self.update_epsilon()
            self.training_it += 1  # increment training iterations counter

            # Update target network
            if self.training_it and self.training_it % self.update_target_steps == 0:
                self.model.update_target_network()

        self.update_counter += 1

    def replay(self):
        """
            DQN algorithm (or Double-DQN)
            Sample minibatch from memory and perform gradient descent
        """
        # If not enough sample in memory
        if len(self.memory.samples) < self.batch_size:
            return

        # Print "time to learn" event
        if not self.printTimeToLearn:
            self.dqn_print(msg="Time to learn")
            self.printTimeToLearn = True

        batch = self.memory.get(self.batch_size)
        batch = zip(*batch)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done, batch_q, batch_q_next = batch

        batch_state = np.array(batch_state).squeeze(axis=1)
        q = self.model.predict(batch_state, batch_len=self.batch_size)

        batch_next_state = np.array(batch_next_state).squeeze(axis=1)
        q_next_target = self.model.predict(batch_next_state, batch_len=self.batch_size, target=True)

        batch_reward = np.array(batch_reward)
        X = batch_state

        # Q-values from Master
        index_Q = [i for i, j in enumerate(batch_q_next) if j is not None] # index of q qnd q_next if not None
        if index_Q and len(index_Q) == self.batch_size:
            for i in range(len(index_Q)): # replace q-values estimate
                q_next_target[i] = batch_q_next[index_Q[i]]
                # q[i] = batch_q[index_Q[i]]
            self.dqn_print(msg="loaded q values")

        if self.use_double_dqn:
            # Double DQN
            q_next = self.model.predict(batch_next_state, batch_len=self.batch_size)

            # Q-values from Master
            if index_Q and len(index_Q) == self.batch_size:
                for i in range(len(index_Q)):  # replace q-values estimate
                    q_next[i] = batch_q_next[index_Q[i]]

            indices = np.argmax(q_next, axis=1)
            indices = np.expand_dims(indices, axis=0)  # need to add 1 dimension to be 2D array
            taken = np.take_along_axis(q_next_target, indices.T, axis=1).squeeze(axis=1)
            labels = batch_reward + self.gamma * taken
        else:
            # DQN
            labels = batch_reward + self.gamma * np.amax(q_next_target, axis=1)

        for i in range(self.batch_size):
            a = batch_action[i]

            if batch_done[i]:  # Terminal state
                r = batch_reward[i]
                q[i][a] = r
            else:
                q[i][a] = labels[i]

        Y = q
        self.model.train(X, Y, batch_len=self.batch_size)

    def update_epsilon(self):
        """
            Iteratively reduce epsilon to its final value during training
        """
        if self.eps > self.eps_end:
            self.eps -= self.eps_decay

            # Reached final epsilon
            if self.eps <= self.eps_end:
                if not self.printStopExploration:
                    self.dqn_print(msg="Finished exploration final eps = {:.2f}".format(self.eps))
                    self.printStopExploration = True

    def stop_training(self):
        """
            Stop training the Neural Network
            Stop exploration -> only exploitation
        """
        self.training = False
        self.eps = self.eps_test

        self.dqn_print(msg="Stop training. Stop exploring")

    def stop_exploring(self):
        """
            Stop exploration -> only exploitation
        """
        self.eps = self.eps_test

        self.dqn_print(msg="Stop exploring")

    def go_random_agent(self):
        """
            Agent takes only random action
        """
        self.random_agent = True
        if self.random_agent:

            self.dqn_print(msg="Go random agent")

            if self.training:
                self.stop_training()

    def stop_collect_experiences(self):
        """
            Stop appending new experience to memory
        """
        self.dqn_print(msg="Stop collecting experiences")

        self.collect_experiences = False

    def save_model(self, dir, suffix=''):
        """
            Save model (neural network, optimizer, loss, etc ..) in a file
            Also create the /brain_files/{sim_id}/ directory if it doesn't exist
            :param dir: directory to save the file
            :param suffix: suffix to the filename
        """
        model_file = dir
        model_file += Util.get_time_string()  # timestring
        # model_file += '_' + str(self.model.h1) + 'h1_' + str(self.model.h1) + 'h2' # NN architecture
        model_file += '_' + suffix
        model_file += '_' + str(self.update_counter) + 'tmstp'  # timesteps
        model_file += '_' + 'model.h5'  # model file extension

        # Create the /brain_files/ directory if it doesn't exist
        if not os.path.exists(os.path.dirname(dir)):
            try:
                os.makedirs(os.path.dirname(dir))
                print("@save_model: directory: {} doesn't exist. Creating directory.".format(dir))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        self.dqn_print(msg="Save agent's model" + " -> file: {}".format(model_file))

        self.model.q_network.save(model_file)

    def save_mem(self, dir, suffix=''):
        """
            Save agent's experiences in csv file
            Also create the /brain_files/{sim_id}/ directory if it doesn't exist
            :param dir: directory to save the file
            :param suffix: suffix to the filename
        """
        memory_file = dir
        memory_file += Util.get_time_string()  # timestring
        memory_file += '_' + suffix
        memory_file += '_' + str(self.update_counter) + 'tmstp'  # timesteps
        memory_file += '_' + 'mem.csv'  # memory file extension

        # Create the /brain_files/ directory if it doesn't exist
        if not os.path.exists(os.path.dirname(dir)):
            try:
                os.makedirs(os.path.dirname(dir))
                print("@save_model: directory: {} doesn't exist. Creating directory.".format(dir))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        self.dqn_print(msg="Save agent's memory" + " -> file: {}".format(memory_file))

        header = ("state", "action", "reward", "next_state", "done", "q", "q_next")

        fo = open(memory_file, 'a')
        writer = csv.writer(fo)
        writer.writerow(header)  # write header
        for i in range(len(self.memory.samples)):
            state = Util.remove_blank(np.array2string(self.memory.samples[i][0])).replace(' ]', ']').replace('[ ', '[')
            action = str(self.memory.samples[i][1])
            reward = str(self.memory.samples[i][2])
            next_state = Util.remove_blank(np.array2string(self.memory.samples[i][3])).replace(' ]', ']').replace('[ ',
                                                                                                                  '[')
            done = str(self.memory.samples[i][4])

            # Record the Q-values
            s = self.memory.samples[i][0]
            _s = self.memory.samples[i][3]
            q = self.model.predict(s, batch_len=1)
            q_next = self.model.predict(_s, batch_len=1)

            experience = (state, action, reward, next_state, done, q, q_next)
            writer.writerow(experience)  # write experience

        fo.close()  # close file properly

    def load_model(self, model_file):
        """
            Load model from given file and set Q-Network and target-Network
            :param model_file: model file to load
        """
        self.dqn_print(msg="Load full model (nn and optimizer)" + " <- file: {}".format(model_file))

        self.model.q_network = load_model(model_file)
        self.model.target_network_network = load_model(model_file)

    def load_full_weights(self, model_file):
        """
            Load weights from given file and set Q-Network and target-Network
            :param model_file: model file to load
        """
        self.dqn_print(msg="Load all weights" + " <- file: {}".format(model_file))

        self.model.q_network.load_weights(model_file)
        self.model.target_network.load_weights(model_file)

    def load_h1_weights(self, model_file):
        """
            Load first hidden layer weights from given file and set Q-Network and target-Network
            :param model_file: model file to load
        """
        self.dqn_print(msg="Load 1st hidden layer weights" + " <- file: {}".format(model_file))

        # Load master
        model_copy = clone_model(self.model.q_network)
        model_copy.load_weights(model_file)

        # 1st hidden layer weights
        weights_h1 = model_copy.layers[0].get_weights()

        # Set weights
        self.model.set_h1_weights(weights_h1)

        if np.array_equal(self.model.q_network.layers[0].get_weights(), weights_h1):
            sys.exit('Error! Q-Network h1 weights is not equal to the h1 weights from file')

    def load_h2_weights(self, model_file):
        """
            Load second hidden layer weights from given file and set Q-Network and target-Network
            :param model_file: model file to load
        """
        self.dqn_print(msg="Load 2nd hidden layer weights" + " <- file: {}".format(model_file))

        # Load master
        model_copy = clone_model(self.model.q_network)
        model_copy.load_weights(model_file)

        # 2nd hidden layer weights
        weights_h2 = model_copy.layers[1].get_weights()

        # Set weights
        self.model.set_h2_weights(weights_h2)

        if np.array_equal(self.model.q_network.layers[1].get_weights(), weights_h2):
            sys.exit('Error! Q-Network h2 weights is not equal to the h2 weights from file')

    def load_out_weights(self, model_file):
        """
            Load output layer weights from given file and set Q-Network and target-Network
            :param model_file: model file to load
        """
        self.dqn_print(msg="Load output layer weights" + " <- file: {}".format(model_file))

        # print('before', self.model.q_network.get_weights())

        # Load master
        model_copy = clone_model(self.model.q_network)
        model_copy.load_weights(model_file)

        # print('master', model_copy.get_weights())

        # Output layer weights
        if self.num_layers == 2:
            weights_output = model_copy.layers[2].get_weights()
        elif self.num_layers == 1:
            weights_output = model_copy.layers[1].get_weights()
        else:
            raise NotImplementedError()

        # Set weights
        self.model.set_out_weights(weights_output)

        # print('after', self.model.q_network.get_weights())

        if self.num_layers == 2:
            if np.array_equal(self.model.q_network.layers[2].get_weights(), weights_output):
                sys.exit('Error! Q-Network output weights is not equal to the output weights from file')
        elif self.num_layers == 1:
            if np.array_equal(self.model.q_network.layers[1].get_weights(), weights_output):
                sys.exit('Error! Q-Network output weights is not equal to the output weights from file')

    def load_h1h2_weights(self, model_file):
        """
            Load first and second hidden layers weights from given file and set Q-Network and target-Network
            :param model_file: model file to load
        """
        self.dqn_print(msg="Load 1st and 2nd hidden layer weights" + " <- file: {}".format(model_file))

        # print('before', self.model.q_network.get_weights())

        # Load master
        model_copy = clone_model(self.model.q_network)
        model_copy.load_weights(model_file)
        # ('model_copy layers', [<keras.layers.core.Dense object at 0x1163e36d0>, <keras.layers.core.Activation object at 0x1163e3790>, <keras.layers.core.Dense object at 0x1163e37d0>, <keras.layers.core.Activation object at 0x1163e38d0>, <keras.layers.core.Dense object at 0x1163e3910>])
        # h2 weights are located in layers[2] of layers list

        # print('master', model_copy.get_weights())

        # 1st hidden layer weights
        weights_h1 = model_copy.layers[0].get_weights()

        # 2nd hidden layer weights
        weights_h2 = model_copy.layers[1].get_weights()

        # Set weights
        self.model.set_h1h2_weights(weights_h1, weights_h2)

        # print('after', self.model.q_network.get_weights())

    def load_h2out_weights(self, model_file):
        """
            Load first and second hidden layers weights from given file and set Q-Network and target-Network
            :param model_file: model file to load
        """
        self.dqn_print(msg="Load h2 and output weights" + " <- file: {}".format(model_file))

        # Load master
        model_copy = clone_model(self.model.q_network)
        model_copy.load_weights(model_file)

        # 2nd hidden layer weights
        weights_h2 = model_copy.layers[1].get_weights()

        # Output layer weights
        weights_output = model_copy.layers[2].get_weights()

        # Set weights
        self.model.set_h2out_weights(weights_h2, weights_output)

    def load_h1out_weights(self, model_file):
        """
            Load first hidden layers and output layers weights from given file and set Q-Network and target-Network
            :param model_file: model file to load
        """
        self.dqn_print(msg="Load h1 and output weights" + " <- file: {}".format(model_file))

        # Load master
        model_copy = clone_model(self.model.q_network)
        model_copy.load_weights(model_file)

        # 1st hidden layer weights
        weights_h1 = model_copy.layers[0].get_weights()

        # Output layer weights
        weights_output = model_copy.layers[2].get_weights()

        # Set weights
        self.model.set_h1out_weights(weights_h1, weights_output)

    def load_mem(self, memory_file, size=-1):
        """
            Load experiences from given memory file
            :param memory_file: memory file to load (CSV format)
            :param size: number of experiences to load
        """
        experiences = []
        data = pd.read_csv(memory_file)

        if size == -1:
            n_exp = data.shape[0]
            self.dqn_print(msg="Load memory: {} exp".format(n_exp) + " <- file: {}".format(memory_file))
        else:
            self.dqn_print(msg="Load memory: {} exp".format(size) + " <- file: {}".format(memory_file))

        remove_bracket = lambda x: x.replace('[', '').replace(']', '')
        string_to_array = lambda x: np.expand_dims(np.fromstring(x, sep=' '), axis=0)

        data['state'] = data['state'].map(remove_bracket).map(string_to_array)
        data['next_state'] = data['next_state'].map(remove_bracket).map(string_to_array)

        if size == -1:
            """
                Default: Load full memory
            """
            for i, row in data.iterrows():
                exp = (row['state'].astype(np.float32), row['action'], row['reward'],
                       row['next_state'].astype(np.float32), row['done'],
                       None, None)
                experiences.append(exp)
        else:
            """
                Load specified number of experiences
            """
            shuffled_data = sklearn.utils.shuffle(data)
            for i, row in shuffled_data[0:size].iterrows():
                exp = (row['state'].astype(np.float32), row['action'], row['reward'],
                       row['next_state'].astype(np.float32), row['done'],
                       None, None)
                experiences.append(exp)

        # Add experiences to memory
        self.memory.push_multi(experiences)

        print('memory size: {}'.format(self.memory.size()))

    def load_mem_q_values(self, memory_file, size=-1):
        """
            Load experiences with Q-values estimates from given memory file
            :param memory_file: memory file to load (CSV format)
            :param size: number of experiences to load
        """
        self.dqn_print(msg="Load memory and Q-values: {} exp".format(size) + " <- file: {}".format(memory_file))

        experiences = []
        data = pd.read_csv(memory_file)

        remove_bracket = lambda x: x.replace('[', '').replace(']', '')
        string_to_array = lambda x: np.expand_dims(np.fromstring(x, sep=' '), axis=0)

        data['state'] = data['state'].map(remove_bracket).map(string_to_array)
        data['next_state'] = data['next_state'].map(remove_bracket).map(string_to_array)

        # Q-values estimate
        data['q'] = data['q'].map(remove_bracket).map(string_to_array)
        data['q_next'] = data['q_next'].map(remove_bracket).map(string_to_array)

        if size == -1:
            """
                Default: Load full memory
            """
            for i, row in data.iterrows():
                exp = (row['state'].astype(np.float32), row['action'], row['reward'],
                       row['next_state'].astype(np.float32), row['done'],
                       row['q'].astype(np.float32), row['q_next'].astype(np.float32))
                experiences.append(exp)
        else:
            """
                Load specified number of experiences
            """
            shuffled_data = sklearn.utils.shuffle(data)
            for i, row in shuffled_data[0:size].iterrows():
                exp = (row['state'].astype(np.float32), row['action'], row['reward'],
                       row['next_state'].astype(np.float32), row['done'],
                       row['q'].astype(np.float32), row['q_next'].astype(np.float32))
                experiences.append(exp)

        # Add experiences to memory
        self.memory.push_multi(experiences)

    def dqn_print(self, msg=""):
        """
            Custom logger in DQN class
            :param msg: message to print
        """
        print_color(color=PRINT_CYAN,
                    msg="agent: {:4.0f}, ".format(self.id) +
                        "{: <35s}, ".format(msg) +
                        "update_tmstp: {:8.0f}, ".format(self.update_counter) +
                        "training_it: {:8.0f}, ".format(self.training_it) +
                        "sim_t: {}, ".format(Global.get_sim_time()) +
                        "global_t: {}, ".format(Global.get_time()) +
                        "world_t: {}".format(Util.get_time_string2()))
