

import pygame
from pygame.locals import *
import os
import errno
import csv
import numpy as np
import matplotlib.pyplot as plt
try:
    # Running in PyCharm
    from Setup import *
    import Util
    import debug_homing
    from res.print_colors import *
    import global_homing
    import res.print_colors as PrintColor
    from EnvironmentHoming import EnvironmentHoming
    from simulation_parameters import *
    import Global
except NameError as err:
    print(err, "--> our error message")
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running from command line -> Import libraries as package")
    from ..Setup import *
    from .. import Util
    import debug_homing
    from ..res.print_colors import *
    import global_homing
    from ..res import print_colors as PrintColor
    from .EnvironmentHoming import EnvironmentHoming
    from .simulation_parameters import *
    from .. import Global

class TestbedHoming(object):

    def __init__(self, sim_param=None, sim_dir="./simulation_data/default/", sim_suffix=""):

        # -------------------- Simulation Parameters ----------------------

        self.render = sim_param.render == 'True'
        self.can_handle_events = sim_param.render == 'True'  # Can handle events only when render is True
        global_homing.debug = sim_param.debug == 'True'
        global_homing.record = sim_param.record == 'True'
        if global_homing.record:
            debug_homing.xprint(color=PRINT_GREEN, msg="Recording to directory: {}".format(sim_dir))
        self.fixed_ur_timestep = sim_param.fixed_ur_timestep == 'True'

        self.training = sim_param.training == 'True'
        self.exploration = sim_param.exploration == 'True'
        self.collect_experiences = sim_param.collect_experiences == 'True'
        self.max_timesteps = int(sim_param.max_timesteps)
        self.max_training_it = int(sim_param.max_training_it)
        self.random_agent = sim_param.random_agent == 'True'

        self.load_model = sim_param.load_model == 'True'
        self.load_full_weights = sim_param.load_full_weights == 'True'
        self.load_h1h2_weights = sim_param.load_h1h2_weights == 'True'
        self.load_h1_weights = sim_param.load_h1_weights == 'True'

        self.load_h2_weights = sim_param.load_h2_weights == 'True'
        self.load_out_weights = sim_param.load_out_weights == 'True'
        self.load_h2out_weights = sim_param.load_h2out_weights == 'True'
        self.load_h1out_weights = sim_param.load_h1out_weights == 'True'

        self.load_memory = int(sim_param.load_mem)
        self.file_to_load = sim_param.file_to_load
        self.suffix = sim_suffix

        self.save_network_freq = int(sim_param.save_network_freq)
        self.save_network_freq_training_it = int(sim_param.save_network_freq_training_it)
        self.save_memory_freq = int(sim_param.save_memory_freq)
        self.start_save_nn_from_it = int(sim_param.start_save_nn_from_it)

        self.wait_learning_score_and_save_model = float(sim_param.wait_learning_score_and_save_model)
        self.record_ls = sim_param.record_ls == 'True'

        # Create environment
        self.environment = EnvironmentHoming(render=self.render, fixed_ur_timestep=self.fixed_ur_timestep)

        # Variables
        self.running = True
        self.pause = False
        self.simulation_dir = sim_dir
        self.simlogs_dir = ""
        self.simfile_suffix = "homing"
        self.brain_dir = ""
        self.ls_dir = ""
        self.learning_scores = []  # initializing the mean score curve (sliding window of the rewards) with respect to timestep
        self.goal_reached_count = 0
        self.collision_count = 0
        self.best_ls = 0

    def setup_simulation(self, sim_id=1, file_to_load="", h1=-1, h2=-1):

        if file_to_load != "":
            self.file_to_load = file_to_load

        # Set ID of simulation
        global_homing.simulation_id = sim_id
        debug_homing.xprint(msg='sim_id: {}, Starting Setup'.format(sim_id))

        # Variables
        self.running = True
        self.pause = False
        self.learning_scores = []  # initializing the mean score curve (sliding window of the rewards) with respect to timestep
        self.goal_reached_count = 0
        self.collision_count = 0

        # Record simulation
        self.simlogs_dir = self.simulation_dir + "sim_logs/"
        if global_homing.record:
            debug_homing.xprint(msg="sim_id: {}, Start recording".format(sim_id))
            filename = global_homing.fileCreate(dir=self.simlogs_dir,
                                       suffix=self.simfile_suffix + '_sim' + str(global_homing.simulation_id) + '_' + self.suffix,
                                       extension=".csv")
            global_homing.simlogs_fo = open(filename, 'a')
            global_homing.simlogs_writer = csv.writer(global_homing.simlogs_fo)

        # Brain directory
        self.brain_dir = self.simulation_dir + "brain_files/" + str(global_homing.simulation_id) + "/"

        # Learning score directory
        self.ls_dir = self.simulation_dir + "ls_files/" + str(global_homing.simulation_id) + "/"

        # Setup agents
        self.setup_agents(h1, h2)

        debug_homing.xprint(msg='sim_id: {}, Setup complete, Start simulation'.format(sim_id))

    def setup_agents(self, h1=-1, h2=-1):
        """
            Setup agents
        """

        debug_homing.xprint(msg='sim_id: {}, Setup Agents'.format(global_homing.simulation_id))

        for agent in self.environment.agents:

            # Setup agent's location and brain
            agent.setup(training=self.training, random_agent=self.random_agent, h1=h1, h2=h2)

            if self.file_to_load != "":
                debug_homing.xprint(msg='sim_id: {}, Loadind File Setup'.format(global_homing.simulation_id))

            if not self.exploration:
                agent.stop_exploring()

            # Load model to agent
            if self.load_model:
                agent.load_model(self.file_to_load)

            # Load full weights to agent
            if self.load_full_weights:
                agent.load_full_weights(self.file_to_load)

            # Load 1st hidden layer weights to agent
            if self.load_h1_weights:
                agent.load_h1_weights(self.file_to_load)

            # Load 1st and 2nd hidden layer weights to agent
            if self.load_h1h2_weights:
                agent.load_h1h2_weights(self.file_to_load)

            # ----------------------------------------------------
            # Load h2 weights to agent
            if self.load_h2_weights:
                agent.load_h2_weights(self.file_to_load)

            # Load output weights to agent
            if self.load_out_weights:
                agent.load_out_weights(self.file_to_load)

            # Load h2 output weights to agent
            if self.load_h2out_weights:
                agent.load_h2out_weights(self.file_to_load)

            # Load h1 output weights to agent
            if self.load_h1out_weights:
                agent.load_h1out_weights(self.file_to_load)
            # ----------------------------------------------------


            # Load memory to agent
            if self.load_memory != -1:
                agent.load_mem(self.file_to_load, self.load_memory)

            # Collect experiences
            if not self.collect_experiences:
                agent.stop_collect_experiences()

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
                    debug_homing.xprint(msg='sim_id: {}, Paused simulation'.format(global_homing.simulation_id))
            if event.type == KEYDOWN and event.key == K_r:
                """
                    R: Record/ Stop record
                """
                if not global_homing.record:  # Record simulation
                    debug_homing.xprint(msg="Start recording")
                    global_homing.record = True
                    filename = global_homing.fileCreate(dir=self.simlogs_dir,
                                                               suffix=self.simfile_suffix + '_sim' + str(global_homing.simulation_id) + '_')
                    global_homing.fo = open(filename, 'a')
                    global_homing.writer = csv.writer(global_homing.simlogs_fo)
                else:  # Stop recording
                    debug_homing.xprint(msg="Stop recording")
                    global_homing.record = False
                    global_homing.simlogs_fo.close()
            if event.type == KEYDOWN and event.key == K_s:
                """
                    S: Save model and memory
                """
                self.environment.agents[0].save_brain(dir=self.brain_dir, suffix=self.suffix)
            if event.type == KEYDOWN and event.key == K_b:
                """
                    B: Load model and Stop training
                """
                self.environment.agents[0].load_model(self.file_to_load)
                self.environment.agents[0].stop_training()
            if event.type == KEYDOWN and event.key == K_l:
                """
                    L: Load full weights and Stop training
                """
                self.environment.agents[0].load_full_weights(self.file_to_load)
                self.environment.agents[0].stop_training()
            if event.type == KEYDOWN and event.key == K_w:
                """
                    W: Load h1 weights and Stop training
                """
                self.environment.agents[0].load_h1_weights(self.file_to_load)
                self.environment.agents[0].stop_training()
            if event.type == KEYDOWN and event.key == K_z:
                """
                    Z: Load h1 h2 weights and Stop training
                """
                self.environment.agents[0].load_h1h2_weights(self.file_to_load)
                self.environment.agents[0].stop_training()

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
                if self.render:
                    self.environment.deltaTime = self.environment.clock.tick(TARGET_FPS) / 1000.0
                else:
                    self.environment.deltaTime = self.environment.clock.tick() / 1000.0
                continue

            self.simulation_logic()

            # Stop simulation if we reach termination condition
            if not self.running:
                break

            # Update environment
            self.environment.update()
            self.environment.fps_physic_step()
            self.environment.draw()

            # Step counter
            Global.sim_timesteps += 1

            # Time counter
            global_homing.timer += self.environment.deltaTime

    def simulation_logic(self):
        """
            Simulation logic
        """
        # Total number of goal reached
        count = 0
        for j in range(self.environment.numAgents):
            count += self.environment.agents[j].goalReachedCount
        self.goal_reached_count = count

        # Total number of collision count
        count = 0
        for k in range(self.environment.numAgents):
            count += self.environment.agents[k].collisionCount
        self.collision_count = count

        # Find the highest learning score (at least after a certain period of time)
        if Global.sim_timesteps >= 10000: # 1000
            ls = self.environment.agents[0].learning_score()
            if self.best_ls < ls:
                self.best_ls = ls

        # Keep track of learning scores over time
        if self.record_ls:
            ls = self.environment.agents[0].learning_score()  # learning score of agent 0
            self.learning_scores.append(ls)  # appending the learning score

        # Save neural networks model frequently
        if global_homing.record and self.save_network_freq != -1:
            if Global.sim_timesteps != 0 and Global.sim_timesteps % self.save_network_freq == 0:
                self.environment.agents[0].save_model(dir=self.brain_dir, suffix=self.suffix)

        # Save memory frequently
        if global_homing.record and self.save_memory_freq != -1:
            if Global.sim_timesteps != 0 and Global.sim_timesteps % self.save_memory_freq == 0:
                self.environment.agents[0].save_mem(dir=self.brain_dir, suffix=self.suffix)

                if self.wait_learning_score_and_save_model != -1:
                    print("highest ls", testbed.best_ls)

        # Save neural networks model frequently based on training iterations
        if global_homing.record and self.save_network_freq_training_it != -1:
            training_it = self.environment.agents[0].training_it()
            if training_it != 0 and training_it >= self.start_save_nn_from_it and training_it % self.save_network_freq_training_it == 0:
                it = str(training_it) + 'it_'
                self.environment.agents[0].save_model(dir=self.brain_dir, suffix=it + self.suffix)

        # # ----------------------------------------------------------------------------------
        # TODO code to be deleted
        if self.environment.agents[0].training_it() >= 10000 and self.environment.agents[0].learning_score() >= 0.084:
            printColor(msg="Agent: {:3.0f}, ".format(self.environment.agents[0].id) +
                           "{:>25s}".format(
                               "Reached {} learning score".format(self.environment.agents[0].learning_score())) +
                           ", tmstp: {:10.0f}".format(Global.sim_timesteps) +
                           ", t: {}".format(Global.get_time()))
            self.running = False
            self.environment.agents[0].save_brain(dir=self.brain_dir)
            self.end_simulation()

            printColor(color=PRINT_RED, msg="BRUTE FORCE EXIT")

            # End the game !
            pygame.quit()
            exit()
            sys.exit()
        # ----------------------------------------------------------------------------------


        # Reached max number of training timesteps
        if self.max_training_it != -1 and self.environment.agents[0].training_it() >= self.max_training_it:

            # Find the highest learning score
            ls = self.environment.agents[0].learning_score()
            if self.best_ls < ls:
                self.best_ls = ls
                # print('best_ls', self.best_ls)
            # Reach LS to find master
            if self.wait_learning_score_and_save_model != -1:
                self.wait_reach_ls_and_save()
                return

            printColor(msg="Agent: {:3.0f}, ".format(self.environment.agents[0].id) +
                           "{:>25s}".format("Reached {} training iterations".format(self.max_training_it)) +
                           ", tmstp: {:10.0f}".format(Global.sim_timesteps) +
                           ", t: {}".format(Global.get_time()))
            self.running = False

        # Reached max number of timesteps
        if self.max_timesteps != -1 and Global.sim_timesteps >= self.max_timesteps:

            self.running = False



    def wait_reach_ls_and_save(self):
        # Wait to reach specified learning score
        if self.wait_learning_score_and_save_model != -1:

            # Reached learning score
            if self.environment.agents[0].learning_score() >= self.wait_learning_score_and_save_model:

                printColor(msg="Agent: {:3.0f}, ".format(self.environment.agents[0].id) +
                               "{:>25s}".format(
                                   "Reached {} learning score".format(self.environment.agents[0].learning_score())) +
                               ", tmstp: {:10.0f}".format(Global.sim_timesteps) +
                               ", t: {}".format(Global.get_time()))
                self.running = False
                self.environment.agents[0].save_brain(dir=self.brain_dir)

            # Not reached yet
            else:
                self.running = True

                # Reset brain every 150000 training it
                if self.environment.agents[0].training_it() != 0 \
                        and self.environment.agents[0].training_it() % 150000 == 0:
                    self.environment.agents[0].reset_brain()

                # if Global.timestep != 0 and Global.timestep % 150000 == 0:
                #     self.environment.agents[0].reset_brain()

    def end_simulation(self):
        """
            Last function called before ending simulation
        """
        debug_homing.xprint(msg='sim_id: {}, Exit'.format(global_homing.simulation_id))

        if global_homing.record:
            global_homing.simlogs_fo.close()
            debug_homing.xprint(msg="Stop recording")

        if self.record_ls:
            self.plot_learning_scores(save_png=False, save_csv=False)

        if global_homing.debug:
            print(debug_homing.dico_event)

        print("At simID: {}, highest ls: {}".format(global_homing.simulation_id, self.best_ls))

    def plot_learning_scores(self, save_png=False, save_csv=False):
        plt.plot(self.learning_scores)
        plt.xlabel('Training Iterations')
        plt.ylabel('Learning Score')
        plt.title('Agent\'s Learning Score over Training Iterations')
        plt.grid(True)
        plt.show(block=False)

        if save_png:
            timestring = global_homing.timestr #time.strftime("%Y_%m_%d_%H%M%S")
            #directory = "./learning_scores/"
            ls_file = self.ls_dir + timestring + "_ls.csv"  # learning scores file
            ls_fig = self.ls_dir + timestring + "_ls.png"  # learning scores figure image

            if not os.path.exists(os.path.dirname(self.ls_dir)):
                try:
                    os.makedirs(os.path.dirname(self.ls_dir))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            plt.savefig(ls_fig)

            if save_csv:
                header = ("timestep", "learning_scores")

                timesteps = np.arange(1, len(self.learning_scores) + 1)
                ls_over_tmstp = zip(timesteps, self.learning_scores)

                with open(ls_file, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(ls_over_tmstp)
                pass

if __name__ == '__main__':

    # Import simulation parameters
    simulation_parameters = args

    # Simulation suffix and directory for records
    simulation_suffix = simulation_suffix(simulation_parameters)
    simulation_directory = simulation_dir(simulation_parameters) # include simulation suffix

    # Keep track of total number of timesteps of all simulations
    total_timesteps = 0

    # -------------------- Simulation ----------------------

    # Create Testbed
    testbed = TestbedHoming(sim_param=simulation_parameters, sim_dir=simulation_directory, sim_suffix=simulation_suffix)

    multi_simulation = int(simulation_parameters.multi_simulation)
    for i in range(multi_simulation):
        simID = i + 1
        debug_homing.xprint(color=PRINT_GREEN, msg="Start Simulation: {}".format(simID))

        testbed.setup_simulation(simID)
        testbed.run_simulation()
        testbed.end_simulation()

        total_timesteps += Global.sim_timesteps # Increment total timesteps
        global_homing.reset_simulation_global() # Reset global variables

    print("All simulation finished\n" 
          "Number of simulations: {}\n"
          "Total simulations time: {}\n"
          "Total timesteps: {}".format(multi_simulation, Global.get_time(), total_timesteps))

    print("highest ls", testbed.best_ls)

    if simulation_parameters.record == 'True':
        # Save whole simulation summary in file (completion time, number of simulation, etc)
        timestr = time.strftime("%Y%m%d_%H%M%S")
        file = open(simulation_directory + "summary_" + timestr + ".txt", "w")
        file.write("Number of simulations: {}\n"
                   "Total simulations time: {}\n"
                   "Total timesteps: {}".format(multi_simulation, Global.get_time(), total_timesteps))
        file.close()

    # End the game !
    pygame.quit()
    exit()
    sys.exit()