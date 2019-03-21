
from collections import deque
from AI.DQN import DQN
from task_race.envs.racecircleleft import RaceCircleLeft
from task_race.envs.racecircleright import RaceCircleRight
from task_race.envs.racecombined import RaceCombined
import task_race.debug_race as debug_race
import task_race.global_race as global_race
from res.print_colors import *

# ---------------------------------- Agent -----------------------------------------

class AgentRace(object):
    def __init__(self, display=False, id=-1, num_agents=0, config=None, max_ep=1000, env_name='', solved_timesteps=-1,
                 manual=False, give_exp=False, tts=2000):
        """
            :param display: turn on/off visualization
            :param id: agent's id
            :param num_agents: total number of agents in the simulation
            :param config: configuration parameters for DQN and environment parameters
            :param max_ep: maximum number of episodes during simulation
            :param env_name: environment name
            :param solved_timesteps: threshold score to consider the environment solved
            :param manual: true -> allows to control the car with keyboard's arrow
            :param give_exp: true -> agents give experience every timestep to the 1st agent (agent 0)
            :param tts: (time to stop) giving experience. Until when agents give their experiences
        """
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

        self.give_exp = give_exp

        # ------------------ Variables to set at each simulation --------------------
        self.brain = None
        self.seed = None
        self.episodes = 0 # number of episodes during current simulation
        self.tmstp_list_size = 100
        self.timesteps_list = deque(maxlen=self.tmstp_list_size)  # keep timesteps of last 100 episodes
        self.average_timestep_100ep = 0 # average timestep over last 100 episodes
        self.tot_timesteps = 0 # total number of timesteps of all episodes passed during 1 simulation
        self.problem_done = False # when problem is done, the simulation end
        self.problem_solved = False # problem is solved Flag
        self.episode_inited = False # at the beginning of an episode we need to reset environment
        self.episode_done = False
        self.state = None
        self.timesteps = 0 # count number of timesteps during 1 episode
        self.experience = None

        self.tts = tts # time when stop sending experiences
        self.print_stop_sending_exp = False # print 'stop sending experiences' when True

    def setup(self, training=True, random_agent=False, seed=None):
        """
            Setup agent for the current simulation
            :param training: decide whether agent trains or not
            :param random_agent: yes -> agent performs random action, else choose action
            :param seed: seed for random
        """
        # Seed for DQN algo
        self.seed = seed

        if random_agent:
            training = False # no training for random agent
            random_agent = random_agent

        # Create agent's brain
        self.brain = DQN(input_size=self.input_size, action_size=self.action_size, id=self.id,
                         training=training, random_agent=random_agent, **self.config.hyperparams, seed=self.seed)

        self.episodes = 0
        self.timesteps_list = deque(maxlen=self.tmstp_list_size)
        self.average_timestep_100ep = 0
        self.problem_done = False
        self.problem_solved = False
        self.episode_inited = False
        self.episode_done = False
        self.tot_timesteps = 0

        self.state = None
        self.timesteps = 0
        self.experience = None

        self.print_stop_sending_exp = False

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

                # passed the time limit
                if not self.env.goal_reached:
                    self.timesteps = self.env.max_episode_steps

                self.timesteps_list.append(self.timesteps)
            else:
                return
            # -----------------------------------------------------------------

            self.episode_inited = False # re-initiate the environment for the next episode

            # Calculate average over the last episodes
            self.average_timestep_100ep = sum(self.timesteps_list) / len(self.timesteps_list)

            # Record event (every episode)
            if global_race.record:
                debug_race.print_event(env=self.env_name, agent=self, episode=self.episodes, tmstp=self.timesteps,
                                       avg_tmstp=self.average_timestep_100ep, d2g=self.env.d2g, tot_tmstp=self.tot_timesteps,
                                       record=True, debug=False)

            # Periodically print event
            if self.episodes % 10 == 0:
                debug_race.print_event(env=self.env_name, agent=self, episode=self.episodes, tmstp=self.timesteps,
                                       avg_tmstp=self.average_timestep_100ep, d2g=self.env.d2g, tot_tmstp=self.tot_timesteps,
                                       record=False, debug=True)

            # Problem solved
            if self.average_timestep_100ep <= self.solved_timesteps and len(self.timesteps_list) >= self.tmstp_list_size:  # last 20 run
                print(
                    "agent: {:4.0f}, *** Solved after {} episodes *** reached solved timestep: {}".format(
                        self.id,
                        self.episodes,
                        self.solved_timesteps))
                self.problem_done = True
                self.problem_solved = True
                return

            # Reached maximum number of episodes
            if self.episodes >= self.max_episodes:
                print(
                    "agent: {:4.0f}, *** Reached maximum number of episodes: {} ***".format(self.id, self.episodes))
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

    def give_experience(self, receiver):
        """
            Agent give experience of the current timestep to the receiver agent
            :param receiver: agent that receives experience
        """
        if self.id == 0: # agent 0 doesn't give experience
            return

        # Exceed max limit of time to give experiences (tts)
        if self.episodes > self.tts:

            if not self.print_stop_sending_exp: # print "stop sending experiences"
                print_color(color=PRINT_RED, msg="episode: {}, agent: {} stop sending experiences to agent: {}"
                            .format(self.episodes, self.id, receiver.id))
                self.print_stop_sending_exp = True
            return

        # Send experience
        if self.episode_inited: # episode currently running
            # print("agent: {} gives experience to agent: {}".format(self.id, self.agents[receiver].id))
            receiver.brain.record(self.experience)  # push experience
