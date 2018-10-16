# OpenGym Taxi-v2
# -------------------

#-------------------- LIBRARIES ------------------------

import numpy as np
import gym

PROBLEM = 'Taxi-v2'
env = gym.make(PROBLEM)

statesCnt = env.observation_space.n # number of possible states # -> 500
actionCnt = env.action_space.n # number of actions # -> 6

#-------------------- Random Agent ------------------------

#env.reset()
#reward = None
#count = 0
#
#while reward != 20:
#    #env.render()
#    action = env.action_space.sample()
#    state, reward, done, info = env.step(action)
#    count += 1
#print("number of iteration : {}".format(count))


#-------------------- Q-learning Agent ------------------------

Q = np.zeros([statesCnt, actionCnt])
alpha = 0.618 # learning rate
gamma = 0.9   # discount factor

NB_EPISODES = 5000

for episode in xrange(1): # range(NB_EPISODES + 1):
    done = False
    G, reward = 0, 0
    state = env.reset()
    while not done:
            action = np.argmax(Q[state]) # choose action with highest q-value
            next_state, reward, done, info = env.step(action) # get action and store next state
            if done:
                print("state", state)
                print("next_state", next_state)
                print(np.max(Q[next_state]))
                print(Q[state,action])
                print("reward", reward)
            #  update Q-table with q-value for this (state,action) using Bellman equation & TD
            Q[state,action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state,action])
            if done:
                print(Q[state,action])
            G += reward
            state = next_state
    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode, G))
