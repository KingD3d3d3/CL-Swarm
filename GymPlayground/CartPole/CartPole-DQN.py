# OpenGym CartPole-v0
# -------------------

import numpy as np
import argparse
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import gym
from collections import deque
import sys
sys.path.append('..')
from DQN import DqnAgent
from FullDQN import FullDqnAgent
from DoubleDQN import DoubleDqnAgent
import os
import time
#-------------------- Agent ----------------------

class CartPoleAgent(FullDqnAgent):

    def build_model(self):
        # Sequential() creates the foundation of the layers.
        model = Sequential()

        # 'Dense' define fully connected layers
        model.add(Dense(24, activation='relu', input_dim=self.inputCnt))  # input (5) -> hidden
        model.add(Dense(24, activation='relu'))                         # hidden -> hidden
        model.add(Dense(self.actionCnt, activation='linear'))             # hidden -> output (2)

        # Compile model
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))  # optimizer for stochastic gradient descent

        return model

#-------------------- MAIN -----------------------

def preprocess(state):
    return np.reshape(state, [1, inputCnt]) # need to add 1 dimension for batch

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='OpenAI Gym CartPole-v0 Game Solver with DQN')
    parser.add_argument('--render', help='render the game', default='True')
    parser.add_argument('--save_all', help='save intermediate models', default='False')
    args = parser.parse_args()
    render = args.render == 'True'
    save_all = args.save_all == 'True'
    PROBLEM = 'CartPole-v0'
    env = gym.make(PROBLEM)
    BRAIN_FILE = 'CartPole-v0_model.h5' # save model when game is completed
    if(save_all):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        directory = timestr + '_models/'
        if not os.path.exists(directory):
            os.makedirs(directory)

    inputCnt = env.observation_space.shape[0]  # number of input signals # -> 4
    actionCnt = env.action_space.n  # number of actions # -> 2

    # agent = CartPoleAgent(inputCnt, actionCnt, batch_size=32, mem_capacity=2000,
    #                       gamma=0.9, lr=0.001, epsilon_max=1, epsilon_min=0.01,
    #                       exploration_steps=1000, brain_file=BRAIN_FILE)


    agent = CartPoleAgent(inputCnt, actionCnt, batch_size=32, mem_capacity=2000,
                          gamma=0.9, lr=0.001, epsilon_max=1, epsilon_min=0.01,
                          exploration_steps=1000, update_target_steps=1000, brain_file=BRAIN_FILE)

    episodeCnt = 0
    scores = deque(maxlen=100)
    finished = False
    average = 0
    tot_timestep = 0
    # EPISODES
    while not finished:
        state = env.reset()
        state = preprocess(state)  # need to add 1 dimension for batch
        score = 0
        timestep = 0

        # TIME-STEP
        while True:

            if render:
                env.render()

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = preprocess(next_state)

            agent.record((state, action, reward, next_state, done))
            state = next_state

            score += reward
            timestep += 1

            # train each step
            agent.replay()

            if done:  # either game over or reached 200 timesteps
                episodeCnt += 1
                scores.append(score)
                tot_timestep += timestep
                break

        # # train the agent with the experience of the episode
        # agent.replay()

        if episodeCnt % 100 == 0:
            if(save_all):
                agent.save_model(directory + str(episodeCnt) + '_' + BRAIN_FILE)

        # if len(scores) >= 100:
        average = sum(scores) / len(scores)
        if average >= 195.0:
            agent.save_model(directory + 'Finished_' + str(episodeCnt) + '_' + BRAIN_FILE)
            print("*** Finished after {} episodes ***".format(episodeCnt))
            finished = True

        if episodeCnt % 10 == 0:
            print("episode: {:5.0f}, timesteps: {:3.0f}, tot_timestep: {:8.0f}, score: {:3.0f}, average: {:3.2f}"
              .format(episodeCnt, timestep, tot_timestep, score, average))
