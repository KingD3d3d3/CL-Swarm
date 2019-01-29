
from collections import deque
from AI.DQN import DQN
from task_race.envs.racecircleleft import RaceCircleLeft
from task_race.envs.racecircleright import RaceCircleRight
from task_race.envs.racecombined import RaceCombined
import sys
import task_race.debug_race as debug_race
import task_race.global_race as global_race
from res.print_colors import *

# ---------------------------------- Agent -----------------------------------------

class AgentRace(object):
    def __init__(self, display=False, id=-1, num_agents=0, config=None, max_ep=1000, env_name='', solved_timesteps=-1,
                 manual=False, give_exp=False, ttr=2000):

        self.id = id # agent's ID

        # create race environment
        if env_name == 'RaceCircleLeft':
            self.env = RaceCircleLeft(display=display, manual=manual)
        elif env_name == 'RaceCircleRight':
            self.env = RaceCircleRight(display=display, manual=manual)
        elif env_name == 'RaceCombined':
            self.env = RaceCombined(display=display, manual=manual)
        else:
            print('cannot find environment: {}'.format(env_name))
            sys.exit()
        self.env_name = env_name

        self.display = display
        # Call env.render() at the beginning before to predict or train neural network (dummy NN processing to avoid the freeze)
        if display:
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

        self.give_exp = give_exp
        # ------------------ Variables to set at each simulation --------------------

        self.brain = None

        self.seed = None

        self.episodes = 0 # number of episodes during current simulation
        # self.scores = deque(maxlen=100) # keep total scores of last 100
        self.tmstp_list_size = 100
        self.timesteps_list = deque(maxlen=self.tmstp_list_size)  # keep timesteps of last 20 episodes
        self.average_timestep = 0
        self.tot_timesteps = 0 # total number of timesteps of all episodes passed during 1 simulation
        self.problem_done = False # when problem is done, the simulation end
        self.problem_solved = False # problem is solved Flag
        self.episode_inited = False # at the beginning of an episode we need to rest environment
        self.episode_done = False
        self.state = None
        self.timesteps = 0 # count number of timesteps during 1 episode
        self.best_average = self.env.max_episode_steps
        self.experience = None

        self.ttr = ttr # time to reach to stop sending experiences
        self.printStopSendingExperiences = False # Flag to be used for printing learning event

    def setup(self, training=True, random_agent=False, seed=None):

        # Seed for DQN algo
        self.seed = seed

        if random_agent:
            training = False # no training for random agent
            random_agent = random_agent

        # Create agent's brain
        self.brain = DQN(input_size=self.input_size, action_size=self.action_size, id=self.id, brain_file="",
                         training=training, random_agent=random_agent, **self.config.hyperparams, seed=self.seed)

        self.episodes = 0
        self.timesteps_list = deque(maxlen=self.tmstp_list_size)
        self.average_timestep = 0
        self.problem_done = False
        self.problem_solved = False
        self.episode_inited = False
        self.episode_done = False
        self.tot_timesteps = 0

        self.state = None
        self.timesteps = 0
        self.best_average = self.env.max_episode_steps
        self.experience = None

        self.printStopSendingExperiences = False # Flag to be used for printing learning event

    def update(self):
        """
            Main function of the agent
        """
        # ------------------------ Game running ------------------------------------------------------------------------
        if not self.problem_done:

            if not self.episode_inited:

                # Synchronize episodes between agents
                if not self.synchronized_episodes():
                    return

                self.state = self.brain.preprocess(self.env.reset())  # initial state
                self.timesteps = 0
                self.episode_inited = True
                self.episode_done = False
                self.experience = None

            # -------------------------- An episode --------------------------
            if self.display:
                self.env.render()

            action = self.brain.select_action(self.state)
            observation, reward, done, info = self.env.step(action)
            # print(" t: {}, reward: {}".format(self.timesteps, reward))
            next_state = self.brain.preprocess(observation)
            self.experience = (self.state, action, reward, next_state, done)

            self.brain.record(self.experience)
            self.state = next_state

            if self.give_exp:
                receiver = self.agents[0]
                self.give_experience(receiver)

            self.brain.train()

            self.timesteps += 1
            self.tot_timesteps += 1 # increment total number of timesteps of all episodes

            if done:  # either game over or reached maximum timesteps of episode
                self.episodes += 1
                self.episode_done = True

                if not self.env.goal_reached:
                    self.timesteps = self.env.max_episode_steps
                self.timesteps_list.append(self.timesteps)
            else:
                return
            # -----------------------------------------------------------------

            self.episode_inited = False # re-initiate the environment for the next episode

            # Calculate average over the last episodes
            self.average_timestep = sum(self.timesteps_list) / len(self.timesteps_list)


            # Calculate best average
            if self.average_timestep < self.best_average and len(self.timesteps_list) >= self.tmstp_list_size:
                self.best_average = self.average_timestep

            # Record event (every episode)
            if global_race.record:
                debug_race.print_event(env=self.env_name, agent=self, episode=self.episodes, tmstp=self.timesteps,
                                       avg_tmstp=self.average_timestep, d2g=self.env.d2g, tot_tmstp=self.tot_timesteps,
                                      record=True, debug=False)

            # Periodically print event
            if self.episodes % 10 == 0:
                debug_race.print_event(env=self.env_name, agent=self, episode=self.episodes, tmstp=self.timesteps,
                                       avg_tmstp=self.average_timestep, d2g=self.env.d2g, tot_tmstp=self.tot_timesteps,
                                       record=False, debug=True)

            # Problem solved
            if self.average_timestep <= self.solved_timesteps and len(self.timesteps_list) >= self.tmstp_list_size:  # last 20 run
                print(
                    "agent: {:4.0f}, *** Solved after {} episodes *** reached solved timestep: {}".format(
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

    def synchronized_episodes(self):
        """
            True if episodes are synchronized between agents, else False
        """
        for a in self.agents:
            if self != a:
                if a.episodes != self.episodes:
                    return False
        return True

    def give_experience(self, receiver):
        """
            TODO Write doc
        """
        if self.id == 0: # agent 0 doesn't give experience
            return

        # Frequency of communication between agents
        if self.episodes > self.ttr: # exceed max limit of time to give experiences

            if not self.printStopSendingExperiences: # print "stop sending experiences"
                print_color(color=PRINT_RED, msg="episode: {}, agent: {} stop sending experiences to agent: {}"
                            .format(self.episodes, self.id, receiver.id))
                self.printStopSendingExperiences = True
            return

        # Part II - Exchange knowledge
        if self.episode_inited: # currently running
            # print("agent: {} gives experience to agent: {}".format(self.id, self.agents[receiver].id))
            receiver.brain.record(self.experience)  # push experience
