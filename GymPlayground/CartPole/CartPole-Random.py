# OpenGym CartPole-v0
# -------------------

import gym

PROBLEM = 'CartPole-v0'

NB_EPISODES = 20

env = gym.make(PROBLEM)

for i_episode in range(NB_EPISODES):
            state = env.reset()  # initialize environment and get initial state
            for t in range(100):
                env.render()  # render environment
                action = env.action_space.sample()  # take random action
                state, reward, done, info = env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
