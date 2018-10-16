
import numpy as np
import argparse
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import gym
from collections import deque
from FullDQN import FullDqnAgent

#-------------------- Agent ----------------------

class GameAgent(FullDqnAgent):

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

    parser = argparse.ArgumentParser(description='OpenAI Gym Game Solver with DQN')
    parser.add_argument('--env', help='game to play', required=True)
    parser.add_argument('--render', help='render the game', default='True')
    parser.add_argument('--term_score', help='average reward over 100 consecutives episodes to solve the game',
                        default=float("inf"))
    args = parser.parse_args()
    render = args.render == 'True'
    ENV = args.env
    term_score = float(args.term_score)
    BRAIN_FILE = ENV + '_model.h5' # save model when game is completed

    env = gym.make(ENV)
    inputCnt = env.observation_space.shape[0]  # number of input signals # -> 4
    actionCnt = env.action_space.n  # number of actions # -> 2
    print('inputCnt', inputCnt)
    print('actionCnt', actionCnt)
    agent = GameAgent(inputCnt, actionCnt, batch_size=32, mem_capacity=20000,
                      gamma=0.9, lr=0.001, epsilon_max=1, epsilon_min=0.001,
                      epsilon_decay=0.995, brain_file=BRAIN_FILE)

    episodeCnt = 0
    scores = deque(maxlen=100)
    finished = False
    average = 0
    tot_timestep = 0

    # EPISODES
    while not finished:
        state = env.reset()
        state = preprocess(state)
        score = 0
        timestep = 0

        # TIME-STEP
        while True:

            if render:
                env.render()

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = preprocess(next_state)

            experience = (state, action, reward, next_state, done)
            agent.update(experience)
            state = next_state

            score += reward
            timestep += 1

            # # train the agent with the experience of the episode
            agent.replay()

            if done:  # either game over or reached 200 timesteps
                episodeCnt += 1
                scores.append(score)
                tot_timestep += timestep
                break

        # train the agent with the experience of the episode
        # agent.replay()

        if len(scores) >= 100:
            average = sum(scores) / 100.0
            if average >= term_score:
                #agent.save_model()
                print("*** Finished after {} episodes ***".format(episodeCnt))
                finished = True

        print("episode: {:5.0f}, timesteps: {:3.0f}, tot_timestep: {:8.0f}, score: {:3.0f}, average: {:3.2f}"
              .format(episodeCnt, timestep, tot_timestep, score, average))
