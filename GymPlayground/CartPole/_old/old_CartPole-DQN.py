# OpenGym CartPole-v0
# -------------------

import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import gym
from collections import deque

PROBLEM = 'CartPole-v0'

BRAIN_FILE = 'CartPole-v0_model.h5'

#-------------------- NETWORK --------------------------

class Network(object):

    def __init__(self, inputCnt, actionCnt):  # input neurons 5 (state), output neurons 3 (actions)
        self.inputCnt = inputCnt
        self.actionCnt = actionCnt

        self.model = self._buildModel()  # load_model(BRAIN_FILE)

    def _buildModel(self):
        # Sequential() creates the foundation of the layers.
        model = Sequential()

        # 'Dense' define fully connected layers
        model.add(Dense(24, activation='relu', input_dim=self.inputCnt))  # input (5) -> hidden
        model.add(Dense(24, activation='relu'))                         # hidden -> hidden
        model.add(Dense(self.actionCnt, activation='linear'))             # hidden -> output (2)

        # Compile model
        model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))  # optimizer for stochastic gradient descent

        return model

    def train(self, x, y, nb_epoch=1, verbose=0, batch_len=1):
        # Fit the model
        self.model.fit(x, y, epochs=nb_epoch, verbose=verbose, batch_size=batch_len)

    def predict(self, state):
        '''
            Predict q-values from input state
        '''
        return self.model.predict(state)

#-------------------- MEMORY --------------------------

class Memory(object):  # sample stored as ( s, a, r, s_ )

    def __init__(self, capacity):
        self.capacity = capacity
        self.samples = deque(maxlen=self.capacity)

    def push(self, sample):
        '''
            Append a new event to the memory
        '''
        self.samples.append(sample)

#        if len(self.samples) > self.capacity:
#            del self.samples[0]

    def sample(self, n):
        '''
            Get n samples randomly
        '''
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

#-------------------- DQN Agent ------------------------

BATCH_SIZE = 32
MEMORY_CAPACITY = 2000  # 100000
GAMMA = 0.9  # discount factor
LEARNING_RATE = 0.001  # learning rate

EPSILON_MAX = 1  # exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

class DqnAgent(object):

    def __init__(self, input_size, nb_action):
        self.model = Network(input_size, nb_action)
        self.memory = Memory(MEMORY_CAPACITY)
        self.epsilon = EPSILON_MAX
        self.brain_file = BRAIN_FILE

    def select_action(self, state):  # epsilon greedy
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()  # take random action

        action_values = self.model.predict(state)
        return np.argmax(action_values[0])  # Select the best q value index as action

    def record(self, sample):
        self.memory.push(sample)

    def replay(self):
        # If not enough sample in memory
        if len(self.memory.samples) < BATCH_SIZE:
            return

        batch = self.memory.sample(BATCH_SIZE)

        x_batch, y_batch = [], []

        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                q_values = self.model.predict(next_state)[0]
                target = reward + GAMMA * np.amax(q_values)

            labels = self.model.predict(state)
            labels[0][action] = target # only the one action passed in the sample will have the actual target

            x_batch.append(state[0]) # remove fake dimension
            y_batch.append(labels[0])

        x_batch = np.array(x_batch) # convert list to array -> automatically add dimension
        y_batch = np.array(y_batch)
        self.model.train(x_batch, y_batch, batch_len=BATCH_SIZE) # Keras models are trained on Numpy arrays of input data and labels

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def save_model(self):
        self.model.model.save(self.brain_file)

#-------------------- MAIN ------------------------

if __name__ == "__main__":

    env = gym.make(PROBLEM)

    inputCnt = env.observation_space.shape[0] # number of input signals # -> 4
    actionCnt = env.action_space.n # number of actions # -> 2

    agent = DqnAgent(inputCnt, actionCnt)

    episodeCnt = 0
    scores = deque(maxlen=100)
    finished = False

    # EPISODES
    while not finished:
        state = env.reset()
        state = np.expand_dims(state, axis=0) # need to add 1 dimension for batch
        score = 0
        timestep = 0

        # TIME-STEP
        while True:
            #env.render()

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)

            agent.record((state, action, reward, next_state, done))
            state = next_state

            score += reward
            timestep += 1

            if done: # either game over or reached 200 timesteps
                episodeCnt += 1
                scores.append(score)
                print("episode: {}, timesteps: {}, score: {}".format(episodeCnt, timestep, score))
                break

        # train the agent with the experience of the episode
        agent.replay()

        if len(scores) >= 100:
            #del scores[0]
            average = sum(scores)/100
            print("average score (100 trials): {}".format(average))
            print("----------------------------------------")
            if average >= 195.0:
                agent.save_model()
                print("*** Finished after {} episodes ***".format(episodeCnt))
                finished = True
