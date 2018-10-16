import numpy as np
from keras.models import load_model
import gym
from collections import deque
import sys
sys.path.append('..')
from DQN import DqnAgent

PROBLEM = 'CartPole-v0'
BRAIN_FILE = 'CartPole-v0_model.h5'

#-------------------- Agent ----------------------

class CartPoleAgent(DqnAgent):

    def build_model(self):
        print("loading model file: {}".format(BRAIN_FILE))
        model = load_model(BRAIN_FILE)
        return model

#-------------------- MAIN -----------------------

if __name__ == "__main__":

    env = gym.make(PROBLEM)

    inputCnt = env.observation_space.shape[0]
    actionCnt = env.action_space.n

    agent = CartPoleAgent(inputCnt, actionCnt, batch_size=32, mem_capacity=2000,
                          gamma=0.9, lr=0.001, epsilon_max=0.01, epsilon_min=0.001)

    episodeCnt = 0
    scores = deque(maxlen=100)
    finished = False
    average = 0
    while not finished:

        state = env.reset()
        state = np.expand_dims(state, axis=0)
        score = 0
        timestep = 0

        while True:
            env.render()

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)

            agent.record((state, action, reward, next_state, done))
            state = next_state

            timestep += 1
            score += reward

            if done:
                episodeCnt += 1
                scores.append(score)
                break


        # train the agent with the experience of the episode
        agent.replay()

        print("episode: {:5.0f}, timesteps: {:3.0f}, score: {:3.0f}, average: {:3.2f}"
              .format(episodeCnt, timestep, score, average))
