
import pygame
from pygame.locals import *
import time
import importlib
import res.Util as Util
from res.print_colors import *
import task_race.simulation_parameters_race as sim_param_race
import Global
import csv
from task_race.AgentRace import AgentRace
import task_race.debug_race as debug_race
import task_race.global_race as global_race

class TestbedRace(object):

    def __init__(self, sim_param=None):

        self.sim_param = sim_param

        # Load config from file # assume path is cfg/<env>/<env>_<agent type>.  e.g. 'config/cartpole/cartpole_dqn'
        sys.path.append('config/' + sim_param.cfg.split('_')[0])
        config = importlib.import_module(sim_param.cfg)
        env = config.environment['env_name']
        self.config = config

        # 2nd agent
        env2 = None
        self.config2 = None
        if sim_param.cfg2:
            sys.path.append('config/' + sim_param.cfg2.split('_')[0])
            config2 = importlib.import_module(sim_param.cfg2)
            env2 = config2.environment['env_name']
            self.config2 = config2

        # 3rd agent
        env3 = None
        self.config3 = None
        if sim_param.cfg3:
            sys.path.append('config/' + sim_param.cfg3.split('_')[0])
            config3 = importlib.import_module(sim_param.cfg3)
            env3 = config3.environment['env_name']
            self.config3 = config3

        print("###########################")
        print("#### CL Testbed Race ######")
        print("###########################")
        # -------------------- Simulation Parameters ----------------------

        self.display = sim_param.display
        self.can_handle_events = sim_param.display
        self.num_agents = sim_param.num_agents

        self.training = sim_param.training
        self.random_agent = sim_param.random_agent
        self.exploration = sim_param.exploration
        self.collect_experiences = sim_param.collect_experiences

        # Max number of episodes
        if sim_param.max_ep:
            self.max_ep = sim_param.max_ep
        else:
            self.max_ep = config.environment['max_ep']

        # Average timestep agent needs to reach to consider the problem solved
        if sim_param.solved_timesteps:
            self.solved_timesteps = sim_param.solved_timesteps
        else:
            self.solved_timesteps = config.environment['solved_timesteps']

        self.load_model = sim_param.load_model
        self.load_all_weights = sim_param.load_all_weights
        self.load_h1h2_weights = sim_param.load_h1h2_weights
        self.load_h1_weights = sim_param.load_h1_weights
        self.load_h2_weights = sim_param.load_h2_weights
        self.load_out_weights = sim_param.load_out_weights
        self.load_h2out_weights = sim_param.load_h2out_weights
        self.load_h1out_weights = sim_param.load_h1out_weights
        self.load_mem = sim_param.load_mem
        self.load_mem2 = sim_param.load_mem2

        self.file_to_load = sim_param.file_to_load
        self.file_to_load2 = sim_param.file_to_load2

        self.save_model = sim_param.save_model
        self.save_mem = sim_param.save_mem
        self.save_model_freq_ep = sim_param.save_model_freq_ep
        self.save_mem_freq_ep = sim_param.save_mem_freq_ep

        # Directory for saving files
        self.suffix = ''
        self.sim_dir = ''
        self.simlogs_dir = ''
        self.brain_dir = ''
        self.env_name = env  # e.g. "CartPole_v0"
        self.env_name2 = None
        self.env_name3 = None
        if sim_param.cfg2:
            self.env_name2 = env2
        if sim_param.cfg3:
            self.env_name3 = env3

        # Directory for saving events, model files or memory files
        global_race.record = sim_param.record
        if global_race.record or sim_param.save_model or \
                sim_param.save_mem or sim_param.save_model_freq_ep or sim_param.save_mem_freq_ep:
            self.suffix = sim_param_race.sim_suffix()
            self.sim_dir = sim_param_race.sim_dir()
            Util.create_dir(self.sim_dir)  # create the directory
            print("\nRecord directory: {}".format(sim_param_race.sim_dir()))

        # Print event only in debug mode
        global_race.debug = sim_param.debug

        # Simulation running flag
        self.running = True
        self.sim_count = 0 # count number of simulation
        self.pause = False

        self.give_exp = sim_param.give_exp
        # Create the agents
        self.agents = []
        for i in range(sim_param.num_agents):
            self.agents.append(AgentRace(display=sim_param.display, id=i, num_agents=self.num_agents, config=config,
                                         max_ep=self.max_ep, solved_timesteps=self.solved_timesteps, env_name=env,
                                        manual=sim_param.manual))
        if sim_param.cfg2:
            agent2 = AgentRace(display=sim_param.display, id=1, num_agents=self.num_agents, config=self.config2,
                               max_ep=self.max_ep, env_name=env2, solved_timesteps=self.solved_timesteps,
                               manual=sim_param.manual, give_exp=self.give_exp)
            self.agents.append(agent2)
        if sim_param.cfg3:
            agent3 = AgentRace(display=sim_param.display, id=2, num_agents=self.num_agents, config=self.config3,
                               max_ep=self.max_ep, env_name=env3, solved_timesteps=self.solved_timesteps,
                               manual=sim_param.manual, give_exp=self.give_exp)
            self.agents.append(agent3)

        # Add reference to others agents to each agent
        for agent in self.agents:
            agent.agents = self.agents

        self.seeds = sim_param.seed
        print('seeds', self.seeds)
        self.max_sim = sim_param.multi_sim

    def setup_simulations(self, sim_id=0, file_to_load=''):
        """
            Setup simulation
        """
        # Set ID of simulation
        global_race.sim_id = sim_id

        print("\nEnvironment: {}".format(self.env_name))
        if self.env_name2:
            print("Environment 2: {}".format(self.env_name2))
        if self.env_name3:
            print("Environment 3: {}".format(self.env_name3))

        debug_race.xprint(color=PRINT_GREEN, msg="Begin simulation")
        debug_race.xprint(msg="Setup")

        if file_to_load:
            self.file_to_load = file_to_load

        # Variables
        self.pause = False
        self.sim_count += 1
        self.running = True

        # Record simulation
        self.simlogs_dir = self.sim_dir + "sim_logs/"
        if global_race.record:
            debug_race.xprint(msg="Start recording".format(sim_id))

            # CSV event file
            suffix = self.env_name + '_sim' + str(global_race.sim_id) + '_' + self.suffix
            filename = debug_race.create_record_file(dir=self.simlogs_dir, suffix=suffix)
            global_race.simlogs_fo = open(filename, 'a')
            global_race.simlogs_writer = csv.writer(global_race.simlogs_fo)
            global_race.simlogs_writer.writerow(debug_race.header)  # write header of the record file

        # Brain directory
        if self.save_model or self.save_mem or self.save_model_freq_ep or self.save_mem_freq_ep:
            self.brain_dir = self.sim_dir + "brain_files/" + str(global_race.sim_id) + "/"
            Util.create_dir(self.brain_dir)

        # Setup agents
        self.setup_agents()

        debug_race.xprint(msg="Setup complete. Start simulation")

    def setup_agents(self):
        """
            Setup agent
        """
        debug_race.xprint(msg="Setup agents")

        for agent in self.agents:

            # Seed
            if self.seeds and len(self.seeds) >= self.max_sim:
                seed = self.seeds[global_race.sim_id] + agent.id
            else:
                seed = None

            # Setup agent's location and brain
            agent.setup(training=self.training, random_agent=self.random_agent, seed=seed)

            if not self.exploration:
                agent.brain.stop_exploring()

            # Load model to agent
            if self.load_model:
                agent.brain.load_model(self.file_to_load)

            # Load full weights to agent
            if self.load_all_weights:
                agent.brain.load_full_weights(self.file_to_load)

            # Load 1st hidden layer weights to agent
            if self.load_h1_weights:
                agent.brain.load_h1_weights(self.file_to_load)

            # Load 1st and 2nd hidden layer weights to agent
            if self.load_h1h2_weights:
                agent.brain.load_h1h2_weights(self.file_to_load)

            # ----------------------------------------------------
            # Load h2 weights to agent
            if self.load_h2_weights:
                agent.brain.load_h2_weights(self.file_to_load)

            # Load output weights to agent
            if self.load_out_weights:
                agent.brain.load_out_weights(self.file_to_load)

            # Load h2 output weights to agent
            if self.load_h2out_weights:
                agent.brain.load_h2out_weights(self.file_to_load)

            # Load h1 output weights to agent
            if self.load_h1out_weights:
                agent.brain.load_h1out_weights(self.file_to_load)
            # ----------------------------------------------------

            # Load memory to agent
            if self.load_mem:
                agent.brain.load_mem(self.file_to_load, self.load_mem)

            # Load memory2 to agent
            if self.load_mem2:
                agent.brain.load_mem(self.file_to_load2, self.load_mem2)

            # Collect experiences
            if not self.collect_experiences:
                agent.brain.stop_collect_experiences()


    def handle_events(self):
        """
            Check and handle the event queue
        """
        if not self.can_handle_events:
            return

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                """
                    ESCP: Quit game
                """
                # The user closed the window or pressed escape
                self.running = False
            if event.type == KEYDOWN and event.key == K_p:
                """
                    P: Pause game
                """
                self.pause = not self.pause  # Pause the game
                if self.pause:
                    print('Paused simulation')

    def run_simulation(self):
        """"
            Main
            Game
            Loop
        """
        while self.running:

            self.handle_events()

            # Pause the game
            if self.pause:
                # clock.tick compute how many milliseconds have passed since the previous call.
                # If we don't call that during pause, clock.tick will compute time spend during pause
                # thus timer is counting during simulation pause -> we want to avoid that !
                if self.display:
                    for agent in self.agents:
                        agent.env.delta_time = agent.env.clock.tick(60.0) / 1000.0
                else:
                    for agent in self.agents:
                        agent.env.delta_time = agent.env.clock.tick() / 1000.0
                continue

            self.simulation_logic()

            # Update agents
            for agent in self.agents:
                # print('agent: {}, world: {}'.format(agent.id, id(agent.env.world)))
                agent.update()

            # Step counter
            Global.timesteps += 1

    def simulation_logic(self):
        """
            Simulation logic
        """
        # Save neural networks model frequently based on episode count
        if self.save_model_freq_ep:
            if self.agents[0].episode_done and \
                    self.agents[0].episodes and self.agents[0].episodes % self.save_model_freq_ep == 0:
                suffix = self.env_name + '_' + str(self.agents[0].episodes) + 'ep'
                self.agents[0].brain.save_model(dir=self.brain_dir, suffix=suffix)

        # End simulation when all agents had done the problem
        for agent in self.agents:
            if not agent.problem_done:
                self.running = True
                break # break the loop at the first active agent
            self.running = False

    def end_simulation(self):
        """
            Last function called before ending simulation
        """
        debug_race.xprint(msg="Exit simulation")

        # Close environment
        for agent in self.agents:
            agent.env.close()

        # Save model or memory experiences of agents
        if self.save_model or self.save_mem:

            for agent in self.agents:
                suffix = self.env_name + '_end'

                # Solved problem suffix
                if agent.problem_solved:
                    suffix += '_solved' + str(agent.episodes) + 'ep'  # add number of episodes it took to solve the problem

                if self.save_model:
                    agent.brain.save_model(dir=self.brain_dir, suffix=suffix)  # save model

                if self.save_mem:
                    agent.brain.save_mem(dir=self.brain_dir, suffix=suffix) # save memory of experiences

        # Close record file
        if global_race.record:
            global_race.simlogs_fo.close()  # close event file
            debug_race.xprint(msg="Stop recording")

            global_race.reset_simulation_global()  # Reset global variables

    def save_summary(self, suffix=''):
        """
            Simulation summary (completion time, number of simulations, etc)
        """
        timestr = time.strftime('%Y%m%d_%H%M%S')
        file = open(self.sim_dir + suffix + 'summary_' + timestr + '.txt', 'w')
        file.write("--------------------------\n")
        file.write("*** Summary of testbed ***\n")
        file.write("--------------------------\n\n")
        file.write("Number of simulations: {}\n".format(self.sim_count) +
                   "Total simulations time: {}\n".format(Global.get_time()) +
                   "Total simulations timesteps: {}\n\n".format(Global.timesteps))

        file.write("---------------------\n")
        file.write("Testbed configuration\n")
        file.write("---------------------\n\n")
        file.write("Simulation parameters: {}\n\n".format(self.sim_param))

        file.write("--------------------------\n")
        file.write("Problem configuration file\n")
        file.write("--------------------------\n\n")
        file.write("Environment: {}\n".format(self.config.environment))
        file.write("Hyperparameters: {}\n".format(self.config.hyperparams))
        if self.env_name2:
            file.write("Environment 2: {}\n".format(self.config2.environment))
            file.write("Hyperparameters 2: {}\n".format(self.config2.hyperparams))
        if self.env_name3:
            file.write("Environment 3: {}\n".format(self.config3.environment))
            file.write("Hyperparameters 3: {}\n".format(self.config3.hyperparams))

        file.close()

if __name__ == '__main__':

    # Import simulation parameters
    param = sim_param_race.args

    # Create Testbed
    testbed = TestbedRace(sim_param=param)

    # -------------------- Simulation ----------------------

    # Iterate over successive simulations
    multi_sim = param.multi_sim
    for s in range(multi_sim):
        testbed.setup_simulations(s)
        testbed.run_simulation()
        testbed.end_simulation()

    # -------------------------------------------------------

    print("\n_______________________")
    print("All simulation finished\n"
          "Record directory: {}\n".format(testbed.sim_dir) +
          "Number of simulations: {}\n".format(testbed.sim_count) +
          "Total simulations time: {}\n".format(Global.get_time()) +
          "Total simulations timesteps: {}".format(Global.timesteps))

    # Simulation summary (completion time, number of simulation, etc)
    if param.record:
        testbed.save_summary()
