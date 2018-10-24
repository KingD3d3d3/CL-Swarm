import time
import importlib
import csv
try:
    # Running in PyCharm
    from gymplayground.AgentGym import AgentGym
    import Global
    import gymplayground.simulation_parameters_gym as sim_param_gym
    import gymplayground.debug_gym as debug_gym
    import gymplayground.global_gym as global_gym
    import Util
    from res.print_colors import *
except NameError as err:
    print(err, "--> our error message")
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running from command line -> Import libraries as package")
    import gymplayground.simulation_parameters_gym as sim_param_gym
    from .. import Global
    from .AgentGym import AgentGym
    from . import debug_gym
    from . import global_gym
    from ..res.print_colors import *
    from .. import Util

class TestbedGym(object):
    def __init__(self, sim_param=None):

        self.sim_param = sim_param

        # Load config from file # assume path is cfg/<env>/<env>_<agent type>.  e.g. 'config/cartpole/cartpole_dqn'
        sys.path.append('config/' + sim_param.cfg.split('_')[0])
        config = importlib.import_module(sim_param.cfg)
        env = config.environment['env_name']
        self.problem_config = config

        print("########################")
        print("#### CL Testbed Gym ####")
        print("########################")
        print("Environment: {}\n".format(env))
        # -------------------- Simulation Parameters ----------------------

        self.render = sim_param.render

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

        # Average score agent needs to reach to consider the problem solved
        if sim_param.solved_score:
            self.solved_score = sim_param.solved_score
        else:
            self.solved_score = config.environment['solved_score']

        self.load_model = sim_param.load_model
        self.load_all_weights = sim_param.load_all_weights
        self.load_h1h2_weights = sim_param.load_h1h2_weights
        self.load_h1_weights = sim_param.load_h1_weights
        self.load_h2_weights = sim_param.load_h2_weights
        self.load_out_weights = sim_param.load_out_weights
        self.load_h2out_weights = sim_param.load_h2out_weights
        self.load_h1out_weights = sim_param.load_h1out_weights
        self.load_mem = sim_param.load_mem

        self.file_to_load = sim_param.file_to_load

        self.save_model = sim_param.save_model
        self.save_mem = sim_param.save_mem
        self.save_network_freq_ep = sim_param.save_network_freq_ep
        self.save_memory_freq_ep = sim_param.save_memory_freq_ep

        # Directory for saving files
        self.suffix = ''
        self.sim_dir = ''
        self.simlogs_dir = ''
        self.brain_dir = ''
        self.env_name = env  # e.g. "CartPole_v0"

        # global_gym.debug = sim_param.debug == 'True'
        global_gym.record = sim_param.record
        if global_gym.record:
            self.suffix = sim_param_gym.sim_suffix()
            self.sim_dir = sim_param_gym.sim_dir()
            Util.create_dir(self.sim_dir) # create the directory
            print("Record directory: {}".format(sim_param_gym.sim_dir()))

        # Simulation running flag
        self.running = True
        self.sim_count = 0 # count number of simulation

        # Create the agents
        self.agents = [AgentGym(render=sim_param.render, id=i, num_agents=self.num_agents, config=config, max_ep=self.max_ep, solved_score=self.solved_score, env_name=env)
            for i in range(self.num_agents)]

    def setup_simulations(self, sim_id=1, file_to_load=''):
        """
            Setup simulation
        """
        # Set ID of simulation
        global_gym.sim_id = sim_id

        debug_gym.xprint(color=PRINT_GREEN, msg="Begin simulation")
        debug_gym.xprint(msg="Setup")

        if file_to_load:
            self.file_to_load = file_to_load

        self.sim_count += 1
        self.running = True

        # Record simulation
        self.simlogs_dir = self.sim_dir + "sim_logs/"
        if global_gym.record:

            debug_gym.xprint(msg="Start recording".format(sim_id))

            # CSV event file
            suffix = self.env_name + '_sim' + str(global_gym.sim_id) + '_' + self.suffix
            filename = debug_gym.create_record_file(dir=self.simlogs_dir, suffix=suffix)
            global_gym.simlogs_fo = open(filename, 'a')
            global_gym.simlogs_writer = csv.writer(global_gym.simlogs_fo)
            global_gym.simlogs_writer.writerow(debug_gym.header) # write header of the record file

            # Brain directory
            if self.save_model or self.save_mem:
                self.brain_dir = self.sim_dir + "brain_files/" + str(global_gym.sim_id) + "/"
                Util.create_dir(self.brain_dir)

        # Setup agents
        self.setup_agents()

        debug_gym.xprint(msg="Setup complete. Start simulation")

    def setup_agents(self):
        """
            Setup agents
        """
        debug_gym.xprint(msg="Setup agents")

        for agent in self.agents:

            # Setup agent's location and brain
            agent.setup(training=self.training, random_agent=self.random_agent)

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

            # Collect experiences
            if not self.collect_experiences:
                agent.brain.stop_collect_experiences()

    def run_simulation(self):
        """"
            Main
            Game
            Loop
        """
        while self.running:

            self.simulation_logic()

            # Stop simulation if when all agents solved the problem
            if not self.running:
                break

            # Update agents
            for agent in self.agents:
                agent.update()

            # Step counter
            Global.sim_timesteps += 1
            Global.timesteps += 1

    def simulation_logic(self):
        """
            Simulation logic
        """
        # # Save neural networks model frequently
        # if global_gym.record and self.save_network_freq:
        #     if Global.timestep != 0 and Global.timestep % self.save_network_freq == 0:
        #         self.agents[0].save_model(dir=self.brain_dir, suffix=self.suffix)

        # # Save memory frequently
        # if global_gym.record and self.save_memory_freq:
        #     if Global.timestep != 0 and Global.timestep % self.save_memory_freq == 0:
        #         self.agents[0].save_mem(dir=self.brain_dir, suffix=self.suffix)

        # # Save neural networks model frequently based on training iterations
        # if global_gym.record and self.save_network_freq_training_it:
        #     training_it = self.agents[0].training_it()
        #     if training_it != 0 and training_it >= self.start_save_nn_from_it and training_it % self.save_network_freq_training_it == 0:
        #         it = str(training_it) + 'it_'
        #         self.environment.agents[0].save_model(dir=self.brain_dir, suffix=it + self.suffix)

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
        debug_gym.xprint(msg="Exit simulation")

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
        if global_gym.record:
            global_gym.simlogs_fo.close() # close event file
            debug_gym.xprint(msg="Stop recording")

        global_gym.reset_simulation_global()  # Reset global variables

    def save_summary(self):
        """
            Simulation summary (completion time, number of simulations, etc)
        """
        timestr = time.strftime('%Y%m%d_%H%M%S')
        file = open(self.sim_dir + 'summary_' + timestr + '.txt', 'w')
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
        file.write("Environment: {}\n".format(self.problem_config.environment))
        file.write("Hyperparameters: {}\n".format(self.problem_config.hyperparams))

        file.close()


if __name__ == '__main__':

    # Import simulation parameters
    param = sim_param_gym.args

    # Create Testbed
    testbed = TestbedGym(sim_param=param)

    # -------------------- Simulation ----------------------

    # Iterate over successive simulations
    multi_sim = param.multi_sim
    for s in range(1, multi_sim + 1):
        testbed.setup_simulations(s)
        testbed.run_simulation()
        testbed.end_simulation()

    # -------------------------------------------------------

    print("\n_______________________")
    print("All simulation finished\n"
          "Number of simulations: {}\n".format(testbed.sim_count) +
          "Total simulations time: {}\n".format(Global.get_time()) +
          "Total simulations timesteps: {}".format(Global.timesteps))

    # Simulation summary (completion time, number of simulation, etc)
    if param.record:
        testbed.save_summary()
