import numpy as np
from keras.models import load_model
import gymplayground
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import sys
import argparse
sys.path.append('..')
from FullDQN import FullDqnAgent

BRAIN_FILE = 'Finished_352_LunarLander_v2_model.h5'

#-------------------- Agent ----------------------

class LunarLanderAgent(FullDqnAgent):

    def build_model(self):
        # Sequential() creates the foundation of the layers.
        model = Sequential()

        # 'Dense' define fully connected layers
        model.add(Dense(64, activation='relu', input_dim=self.inputCnt))  # input (5) -> hidden
        model.add(Dense(64, activation='relu'))                         # hidden -> hidden
        model.add(Dense(self.actionCnt, activation='linear'))             # hidden -> output (2)

        # Compile model
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))  # optimizer for stochastic gradient descent

        return model

#-------------------- MAIN -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OpenAI Gym CartPole-v0 Game Solver with DQN')
    parser.add_argument('--render', help='render the game', default='True')
    args = parser.parse_args()
    render = args.render == 'True'

    PROBLEM = 'LunarLander-v2'
    env = gymplayground.make(PROBLEM)

    inputCnt = env.observation_space.shape[0]
    actionCnt = env.action_space.n

    agent = LunarLanderAgent(inputCnt, actionCnt, lr=0.001, epsilon_max=0.01, epsilon_min=0.01)
    agent.load_full_weights(BRAIN_FILE)

    episodeCnt = 0
    scores = deque(maxlen=100)
    finished = False
    average = 0
    tot_timestep = 0
    while not finished:

        state = env.reset()
        state = np.expand_dims(state, axis=0)
        score = 0
        timestep = 0

        while True:

            if render:
                env.render()

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            state = next_state

            timestep += 1
            score += reward

            if done:
                episodeCnt += 1
                scores.append(score)
                tot_timestep += timestep
                break

        average = sum(scores) / len(scores)
        print("episode: {:5.0f}, timesteps: {:3.0f}, score: {:3.0f}, average: {:3.2f}"
              .format(episodeCnt, timestep, score, average))
