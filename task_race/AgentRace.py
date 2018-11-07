
from collections import deque
from AI.DQN import DQN
from task_race.envs.racecircleleft import RaceCircleLeft
from task_race.envs.racecircleright import RaceCircleRight
import sys

# ---------------------------------- Agent -----------------------------------------

class AgentRace(object):
    def __init__(self, render=False, id=-1, num_agents=0, config=None, max_ep=5000, env_name='', solved_timesteps=120,
                 seed=None):

        self.id = id # agent's ID

        # create race environment
        if env_name == 'RaceCircleLeft':
            self.env = RaceCircleLeft(display=render)
        elif env_name == 'RaceCircleRight':
            self.env = RaceCircleRight(display=render)
        else:
            print('cannot find environment: {}'.format(env_name))
            sys.exit()

        self.seed = seed
        self.render = render
        # Call env.render() at the beginning before to predict or train neural network (dummy NN processing to avoid the freeze)
        if render:
            self.env.render()

        self.input_size = self.env.input_size
        self.action_size = self.env.action_size

        # List of agents
        self.num_agents = num_agents
        self.agents = [self]

        self.config = config # configuration file
        self.solved_timesteps = solved_timesteps # average score agent needs to reach to consider the problem solved
        self.max_episodes = max_ep

        self.best_agent = False # flag that indicates if it is the best agent according to average scores

        # ------------------ Variables to set at each simulation --------------------

        self.brain = None

        self.episodes = 0 # number of episodes during current simulation
        self.scores = deque(maxlen=100) # keep total scores of last 100 episodes
        self.average_score = 0
        self.timesteps_list = deque(maxlen=20)  # keep timesteps of last 20 episodes
        self.average_timestep = 0
        self.tot_timesteps = 0 # total number of timesteps of all episodes passed during 1 simulation
        self.problem_done = False # when problem is done, the simulation end
        self.problem_solved = False # problem is solved Flag
        self.episode_inited = False # at the beginning of an episode we need to rest environment
        self.episode_done = False
        self.state = None
        self.score = 0 # keep score of 1 episode
        self.timesteps = 0 # count number of timesteps during 1 episode
        self.best_average = self.env.max_episode_steps


    def setup(self, training=True, random_agent=False):

        if random_agent:
            training = False # no training for random agent
            random_agent = random_agent

        # Create agent's brain
        self.brain = DQN(input_size=self.input_size, action_size=self.action_size, id=self.id, brain_file="",
                         training=training, random_agent=random_agent, **self.config.hyperparams, seed=self.seed)

        self.episodes = 0
        self.scores = deque(maxlen=100)
        self.average_score = 0
        self.timesteps_list_size = 20
        self.timesteps_list = deque(maxlen=self.timesteps_list_size)
        self.average_timestep = 0
        self.problem_done = False
        self.problem_solved = False
        self.episode_inited = False
        self.episode_done = False
        self.tot_timesteps = 0
        self.state = None
        self.score = 0
        self.timesteps = 0
        self.best_average = self.env.max_episode_steps

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

                if not self.env.goal_reached:
                    self.timesteps = self.env.max_episode_steps
                self.timesteps_list.append(self.timesteps)
            else:
                return
            # -----------------------------------------------------------------

            # Calculate average over the last episodes
            self.average_score = sum(self.scores) / len(self.scores)
            self.average_timestep = sum(self.timesteps_list) / len(self.timesteps_list)

            self.episode_inited = False # re-initiate the environment for the next episode
            self.tot_timesteps += self.timesteps # increment total number of timesteps of all episodes

            # Calculate best average
            if self.average_timestep < self.best_average and len(self.timesteps_list) >= self.timesteps_list_size:
                self.best_average = self.average_timestep

            # Periodically print event
            if self.episodes % 1 == 0:
                print("episode: {:5.0f}, timesteps: {:3.0f}, tot_timestep: {:8.0f}, score: {:3.0f}, "
                    "average_score: {:3.2f}, average_tmstp: {:3.2f}, best_avg: {:3.2f}"
                        .format(self.episodes, self.timesteps, self.tot_timesteps,
                                self.score, self.average_score, self.average_timestep, self.best_average))

            # Problem solved
            if self.average_timestep <= self.solved_timesteps and len(self.timesteps_list) >= self.timesteps_list_size:  # last 20 run
                print(
                    "agent: {:4.0f}, *** Solved after {} episodes *** reached solved score timestep: {}".format(
                        self.id,
                        self.episodes,
                        self.solved_timesteps))
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


