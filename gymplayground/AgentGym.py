import gym
from collections import deque

try:
    # Running in PyCharm
    import res.colors as Color
    # from ..res import colors as Color
    from AI.DQN import DQN
    import Util
    import res.print_colors as PrintColor
    import gymplayground.debug_gym as debug_gym
    import gymplayground.global_gym as global_gym
    import Global
except NameError as err:
    print(err, "--> our error message")
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running from command line -> Import libraries as package")
    from ..res import colors as Color
    from ..AI.DQN import DQN
    from .. import Util
    from ..res import print_colors as PrintColor
    from . import debug_gym
    from . import global_gym
    from .. import Global


# ---------------------------------- Agent -----------------------------------------

class AgentGym(object):
    def __init__(self, render=False, id=-1, num_agents=0, config=None, max_ep=2000, env_name='', solved_score=100000):

        # Agent's ID
        self.id = id

        self.env = gym.make(env_name)  # create Gym environment
        self.render = render

        # Call env.render() at the beginning before to predict or train neural network (dummy NN processing to avoid the freeze)
        if render:
            self.env.render()

        self.input_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        # Number of agents
        self.num_agents = num_agents

        self.config = config # configuration file
        self.solved_score = solved_score # average score agent needs to reach to consider the problem solved
        self.max_episodes = max_ep

        # ------------------ Variables to set at each simulation --------------------

        self.brain = None

        self.episodes = 0 # number of episodes during current simulation
        self.scores = deque(maxlen=100) # keep total scores of last 100 episodes
        self.problem_done = False # when problem is done, the simulation end
        self.problem_solved = False # problem is solved Flag
        self.episode_inited = False # at the beginning of an episode we need to rest environment
        self.episode_done = False
        self.tot_timesteps = 0 # total number of timesteps of all episodes passed during 1 simulation

        # Need to be initialized because they are property
        self.state = None
        self.score = 0 # keep score of 1 episode
        self.timesteps = 0 # count number of timesteps during 1 episode

    def setup(self, training=True, random_agent=False):

        if random_agent:
            training = False # no training for random agent
            random_agent = random_agent

        # Create agent's brain
        self.brain = DQN(input_size=self.input_size, action_size=self.action_size, id=self.id, brain_file="",
                         training=training, random_agent=random_agent, **self.config.hyperparams)

        self.episodes = 0
        self.scores = deque(maxlen=100)
        self.problem_done = False
        self.problem_solved = False
        self.episode_inited = False
        self.episode_done = False
        self.tot_timesteps = 0

        self.state = None
        self.score = 0
        self.timesteps = 0

    def update(self):
        """
            Main function of the agent
        """
        # ------------------------ Game running ------------------------------------------------------------------------
        if not self.problem_done:

            if not self.episode_inited:
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
                self.scores.append(self.score)
                self.episode_done = True
            else:
                return
            # -----------------------------------------------------------------

            # Calculate average over the last episodes
            average = sum(self.scores) / len(self.scores)

            self.episode_inited = False # re-initiate the environment for the next episode
            self.tot_timesteps += self.timesteps # increment total number of timesteps of all episodes

            # Record event (every episode)
            if global_gym.record: #and self.episodes % 10 == 0: # TODO every 10 episodes, to be changed to 1 episode
                debug_gym.print_event(agent=self, episode=self.episodes, score=self.score, avg_score=average,
                                      timesteps=self.timesteps, tot_timesteps=self.tot_timesteps, record=True, debug=False)

            # Periodically print current average of reward
            if self.episodes % 10 == 0:
                debug_gym.print_event(agent=self, episode=self.episodes, score=self.score, avg_score=average,
                                      timesteps=self.timesteps, tot_timesteps=self.tot_timesteps, record=False, debug=True)

            # Problem solved
            if average >= self.solved_score and len(self.scores) >= 100:  # need reach solved score and at least 100 episodes to terminate
                print("agent: {:4.0f}, *** Solved after {} episodes *** reached solved score: {}".format(self.id, self.episodes, self.solved_score))
                self.problem_done = True
                self.problem_solved = True
                return

            # Reached maximum limit of episodes
            if self.episodes >= self.max_episodes:
                print(
                    "agent: {:4.0f}, *** Reached maximum number of episodes: {} ***".format(self.id, self.episodes))
                self.problem_done = True
                return
            # --------------------------------------------------------------------------------------------------------------
