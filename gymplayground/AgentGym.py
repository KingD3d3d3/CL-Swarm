import gym
from collections import deque
from AI.DQN import DQN
import gymplayground.debug_gym as debug_gym
import gymplayground.global_gym as global_gym
from res.print_colors import *

# ---------------------------------- Agent -----------------------------------------

class AgentGym(object):
    def __init__(self, render=False, id=-1, num_agents=0, config=None, max_ep=2000, env_name='', solved_score=100000,
                 collaboration=False):
        """
            :param render: turn on/off visualization
            :param id: agent's id
            :param num_agents: total number of agents in the simulation
            :param config: configuration parameters for DQN and environment parameters
            :param max_ep: maximum number of episodes during simulation
            :param env_name: environment name
            :param solved_score: threshold score to consider the environment solved
            :param collaboration: decide whether agent is collaborating or not
        """
        # Agent's ID
        self.id = id

        # Gym environment and seed
        self.env = None
        self.env_name = env_name
        self.seed = None

        self.render = render

        self.input_size = None
        self.action_size = None

        # List of agents
        self.num_agents = num_agents
        self.agents = [self]

        self.config = config  # configuration file
        self.solved_score = solved_score  # average score agent needs to reach to consider the problem solved
        self.max_episodes = max_ep

        self.best_agent = False  # flag that indicates if it is the best agent according to average scores

        self.collaboration = collaboration
        # ------------------ Variables to set at each simulation --------------------

        self.brain = None

        self.episodes = 0  # number of episodes during current simulation
        self.scores_100ep = deque(maxlen=100)  # keep total scores of last 100 episodes
        self.scores_10ep = deque(maxlen=10)  # keep total scores of last 10 episodes

        self.average_score_100ep = 0 # average score over last 100 episodes
        self.average_score_10ep = 0 # average score over last 10 episodes

        self.problem_done = False  # when problem is done, the simulation end
        self.problem_solved = False  # problem is solved Flag
        self.episode_inited = False  # at the beginning of an episode we need to rest environment
        self.episode_done = False
        self.tot_timesteps = 0  # total number of timesteps of all episodes passed during 1 simulation

        # Need to be initialized because they are property
        self.state = None
        self.score = 0  # score of the last episode
        self.timesteps = 0  # count number of timesteps during 1 episode

        # Collaborative Learning
        self.cl_allweights = False
        self.cl_exp = 0
        self.exchange_freq = 0

    def setup(self, training=True, random_agent=False, seed=None):
        """
            Setup Gym agent for current simulation
            :param training: decide whether agent trains or not
            :param random_agent: yes -> agent performs random action, else choose action
            :param seed: seed for random
        """
        # Gym environment and seed
        self.env = gym.make(self.env_name)
        self.env_name = self.env_name
        if seed is not None:
            self.env.seed(seed)
        self.seed = seed

        self.input_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        # Call env.render() at the beginning before to predict or train neural network (dummy NN processing to avoid the freeze in runtime)
        if self.render:
            self.env.render()

        if random_agent:
            training = False  # no training for random agent
            random_agent = random_agent

        # Create agent's brain
        self.brain = DQN(input_size=self.input_size, action_size=self.action_size, id=self.id,
                         training=training, random_agent=random_agent, **self.config.hyperparams, seed=self.seed)

        # Reset variables
        self.episodes = 0
        self.score = 0
        self.scores_100ep = deque(maxlen=100)
        self.scores_10ep = deque(maxlen=10)
        self.average_score_100ep = 0
        self.average_score_10ep = 0
        self.problem_done = False
        self.problem_solved = False
        self.episode_inited = False
        self.episode_done = False
        self.tot_timesteps = 0
        self.state = None
        self.timesteps = 0

    def update(self):
        """
            Main function of the agent
        """
        # ------------------------ Simulation running ------------------------------------------------------------------------
        if not self.problem_done:

            if not self.episode_inited:

                # Synchronize episodes between agents
                if not self.synchronized_episodes():
                    return

                if self.collaboration:
                    self.collaborative_learning()

                self.state = self.brain.preprocess(self.env.reset())  # initial state
                self.score = 0
                self.timesteps = 0
                self.episode_inited = True
                self.episode_done = False

            # -------------------------- An episode --------------------------
            if self.render:
                self.env.render()

            action = self.brain.select_action(self.state)
            observation, reward, done, info = self.env.step(action)
            next_state = self.brain.preprocess(observation)

            self.brain.record((self.state, action, reward, next_state, done))
            self.state = next_state

            self.brain.train()

            self.score += reward
            self.timesteps += 1

            if done:  # either game over or reached maximum timesteps of episode
                self.episodes += 1
                self.scores_100ep.append(self.score)
                self.scores_10ep.append(self.score)
                self.episode_done = True
            else:
                return
            # -----------------------------------------------------------------

            # Calculate average over the last episodes
            self.average_score_100ep = sum(self.scores_100ep) / len(self.scores_100ep)
            self.average_score_10ep = sum(self.scores_10ep) / len(
                self.scores_10ep)  # Util.weighted_average_10(self.scores_10ep) # SMA -> sum(self.scores_10ep) / len(self.scores_10ep)

            self.episode_inited = False  # re-initiate the environment for the next episode
            self.tot_timesteps += self.timesteps  # increment total number of timesteps of all episodes

            # Record event (every episode)
            if global_gym.record:
                debug_gym.print_event(env=self.env_name, agent=self, episode=self.episodes, score=self.score,
                                      avg_score_100ep=self.average_score_100ep, avg_score_10ep=self.average_score_10ep,
                                      timesteps=self.timesteps, tot_timesteps=self.tot_timesteps,
                                      record=True, debug=False)

            # Periodically print current average of reward
            if self.episodes % 10 == 0:
                debug_gym.print_event(env=self.env_name, agent=self, episode=self.episodes, score=self.score,
                                      avg_score_100ep=self.average_score_100ep, avg_score_10ep=self.average_score_10ep,
                                      timesteps=self.timesteps, tot_timesteps=self.tot_timesteps,
                                      record=False, debug=True)

            # Problem solved
            if self.average_score_100ep >= self.solved_score and len(
                    self.scores_100ep) >= 100:  # need reach solved score and at least 100 episodes to terminate
                print("env: {:<15s}, agent: {:4.0f}, *** Solved after {} episodes *** reached solved score: {}"
                      .format(self.env_name, self.id, self.episodes, self.solved_score))
                self.problem_done = True
                self.problem_solved = True
                return

            # Reached maximum limit of episodes
            if self.episodes >= self.max_episodes:
                print("env: {:<15s}, agent: {:4.0f}, *** Reached maximum number of episodes: {} ***"
                      .format(self.env_name, self.id, self.episodes))
                self.problem_done = True
                return
                # --------------------------------------------------------------------------------------------------------------

    def synchronized_episodes(self):
        """
            Check episode synchronization between agents
            :return: True if episodes are synchronized, else False
        """
        for a in self.agents:
            if self != a:
                if a.episodes != self.episodes:
                    return False
        return True

    def collaborative_learning(self):
        """
            Decentralized Collaborative Learning
        """
        self.best_agent = False  # reset best agent

        # Frequency of communication between both agents
        if not (self.episodes and self.episodes % self.exchange_freq == 0):
            return

        # Part I - Find the best agent
        for agent in self.agents:
            if self != agent:  # skip myself
                if self.average_score_10ep < agent.average_score_10ep or agent.best_agent:  # 10ep SMA metric best agent
                    # if self.average_score < agent.average_score or agent.best_agent: # 100ep metric best agent
                    # if self.average_score > agent.average_score or agent.best_agent: # 100ep metric worst agent
                    self.best_agent = False
                    break
                else:
                    self.best_agent = True

        # Part II - Exchange knowledge
        if self.best_agent:
            for agent in self.agents:
                if self != agent:  # skip myself

                    # All weights
                    if self.cl_allweights:
                        print_color(color=PRINT_RED, msg="episode: {}, agent: {} gives all weights to agent: {}"
                                    .format(self.episodes, self.id, agent.id))
                        agent.brain.model.set_weights(self.brain.model.q_network.get_weights())
