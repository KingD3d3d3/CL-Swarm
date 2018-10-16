# OpenGym CartPole-v0
# -------------------

import numpy as np
import argparse
import gym
from collections import deque
import sys
sys.path.append('..')
from DQN import DqnAgent
# from ..DQN import DqnAgent

from keras.models import load_model

#-------------------- Agent ----------------------

class CartPoleAgent(DqnAgent):

    def build_model(self):
        # Sequential() creates the foundation of the layers.
        model = load_model(BRAIN_FILE)
        return model

#-------------------- MAIN -----------------------

def preprocess(state):
    return np.reshape(state, [1, inputCnt]) # need to add 1 dimension for batch

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='OpenAI Gym CartPole-v0 Game Solver with DQN')
    parser.add_argument('--render', help='render the game', default='True')
    parser.add_argument('--brainfile', help='render the game', default='CartPole-v0_model.h5')
    args = parser.parse_args()
    render = args.render == 'True'
    PROBLEM = 'CartPole-v0'
    BRAIN_FILE = args.brainfile

    env = gym.make(PROBLEM)

    inputCnt = env.observation_space.shape[0]  # number of input signals # -> 4
    actionCnt = env.action_space.n  # number of actions # -> 2

    agent = CartPoleAgent(inputCnt, actionCnt, batch_size=32, mem_capacity=2000,
                          gamma=0.9, lr=0.001, epsilon_max=0.01, epsilon_min=0.01,
                          epsilon_decay=0.995, brain_file=BRAIN_FILE)

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

            state = next_state

            score += reward
            timestep += 1

            if done:  # either game over or reached 200 timesteps
                episodeCnt += 1
                scores.append(score)
                tot_timestep += timestep
                break

        if len(scores) >= 100:
            average = sum(scores) / 100

        print("episode: {:5.0f}, timesteps: {:3.0f}, tot_timestep: {:8.0f}, score: {:3.0f}, average: {:3.2f}"
              .format(episodeCnt, timestep, tot_timestep, score, average))
