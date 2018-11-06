

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
    from res.print_colors import *
    import res.print_colors as PrintColor
    from task_race.EnvironmentRace import EnvironmentRace
    import task_race.simulation_parameters_race as sim_param_race
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
    from .. import Global
    import task_race.simulation_parameters_race as sim_param_race
    from task_race.EnvironmentRace import EnvironmentRace

class TestbedRace(object):

    def __init__(self, sim_param=None, sim_dir="./simulation_data/default/", sim_suffix=""):

        self.sim_param = sim_param
        # -------------------- Simulation Parameters ----------------------

        self.render = sim_param.render
        self.can_handle_events = sim_param.render

        self.training = sim_param.training
        self.random_agent = sim_param.random_agent
        self.exploration = sim_param.exploration
        self.collect_experiences = sim_param.collect_experiences

        self.num_agents = sim_param.num_agents

        # Max number of episodes
        if sim_param.max_ep:
            self.max_ep = sim_param.max_ep

        # Average score agent needs to reach to consider the problem solved
        if sim_param.solved_score:
            self.solved_score = sim_param.solved_score

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
        self.save_model_freq_ep = sim_param.save_model_freq_ep
        self.save_mem_freq_ep = sim_param.save_mem_freq_ep

        # Directory for saving files
        self.suffix = ''
        self.sim_dir = ''
        self.simlogs_dir = ''
        self.brain_dir = ''
        self.env_name = 'race'  # e.g. "CartPole_v0"

        # Directory for saving events, model files or memory files
        if sim_param.save_model or \
                sim_param.save_mem or sim_param.save_model_freq_ep or sim_param.save_mem_freq_ep:
            self.suffix = sim_param_race.sim_suffix()
            self.sim_dir = sim_param_race.sim_dir()
            Util.create_dir(self.sim_dir)  # create the directory
            print("Record directory: {}".format(sim_param_race.sim_dir()))


        # Create environment
        self.environment = EnvironmentRace(render=self.render, solved_score=self.solved_score, seed=sim_param.seed)

        # Simulation running flag
        self.running = True
        self.sim_count = 0 # count number of simulation
        self.pause = False

    def setup_simulation(self, sim_id=0, file_to_load=''):

        # Variables
        self.running = True
        self.pause = False

        if file_to_load:
            self.file_to_load = file_to_load

        # Record simulation
        self.simlogs_dir = self.sim_dir + "sim_logs/"

        # Brain directory
        if self.save_model or self.save_mem or self.save_model_freq_ep or self.save_mem_freq_ep:
            self.brain_dir = self.sim_dir + "brain_files/" + str(0) + "/"
            Util.create_dir(self.brain_dir)

        # Setup agents
        self.setup_agents()

    def setup_agents(self):
        """
            Setup agent
        """
        agent = self.environment.agent
        agent.setup(training=self.training)
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
                if self.render:
                    self.environment.delta_time = self.environment.clock.tick(TARGET_FPS) / 1000.0
                else:
                    self.environment.delta_time = self.environment.clock.tick() / 1000.0
                continue

            self.simulation_logic()

            # Stop simulation if we reach termination condition
            if not self.running:
                break

            self.environment.update()

    def simulation_logic(self):
        if self.environment.agent.problem_done:
            self.running = False
        else:
            self.running = True

    def end_simulation(self):
        """
            Last function called before ending simulation
        """
        print('Exit')

        # Save model or memory experiences of agents
        if self.save_model or self.save_mem:

            suffix = self.env_name + '_end'

            # Solved problem suffix
            if self.environment.agent.problem_solved:
                suffix += '_solved' + str(
                    self.environment.agent.episodes) + 'ep'  # add number of episodes it took to solve the problem

            if self.save_model:
                self.environment.agent.brain.save_model(dir=self.brain_dir, suffix=suffix)  # save model

            if self.save_mem:
                self.environment.agent.brain.save_mem(dir=self.brain_dir, suffix=suffix)  # save memory of experiences


if __name__ == '__main__':

    # Import simulation parameters
    param = sim_param_race.args

    # Create Testbed
    testbed = TestbedRace(sim_param=param)

    # -------------------- Simulation ----------------------


    testbed.setup_simulation()
    testbed.run_simulation()
    testbed.end_simulation()
