
import argparse
import gymplayground
from collections import deque
import time
try:
    from AI.DQN import DQN
except NameError as err:
    print(err, "--> our error message")
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running from command line -> Import libraries as package")
    from ...AI.DQN import DQN

if __name__ == "__main__":

    PROBLEM = 'CartPole-v0' #'LunarLander-v2'
    env = gymplayground.make(PROBLEM)
    render = True

    # Call env.render() at the beginning before to predict or train neural network (dummy NN processing to avoid the freeze)
    if render:
        env.render()

    input_size = env.observation_space.shape[0]  # number of input signals
    action_size = env.action_space.n  # number of actions

    brain = DQN(input_size=input_size, action_size=action_size, id=0, training=True, random_agent=False,
                h1=32, h2=32, mem_capacity=100000, batch_size=32, gamma=0.99, lr=0.001, eps_start=1, eps_end=0.05,
                exploration_steps=10000, update_target_steps=1000, use_double_dqn=True)

    episodeCnt = 0
    scores = deque(maxlen=100)
    finished = False
    average = 0
    tot_timestep = 0

    # Episodes
    while not finished:
        state = brain.preprocess(env.reset()) # initial state
        score = 0
        timestep = 0

        # Timesteps
        while True:

            if render:
                env.render()

            action = brain.select_action(state)
            observation, reward, done, info = env.step(action)
            next_state = brain.preprocess(observation)

            brain.record((state, action, reward, next_state, done))
            state = next_state

            score += reward
            timestep += 1

            # train each step
            brain.train()

            if done:  # either game over or reached 200 timesteps
                episodeCnt += 1
                scores.append(score)
                tot_timestep += timestep
                break

        average = sum(scores) / len(scores)
        if average >= 200.0:
            print("*** Finished after {} episodes ***".format(episodeCnt))
            finished = True

        if episodeCnt % 10 == 0:
            print("episode: {:5.0f}, timesteps: {:3.0f}, tot_timestep: {:8.0f}, score: {:3.0f}, average: {:3.2f}"
                  .format(episodeCnt, timestep, tot_timestep, score, average))