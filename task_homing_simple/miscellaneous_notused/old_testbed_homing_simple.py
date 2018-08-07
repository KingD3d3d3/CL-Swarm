from __future__ import division

import random
import argparse
import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, vec2)
from pygame.locals import *
# import matplotlib.pyplot as plt
import time
import os
import errno
import csv
import numpy as np
import sys
import datetime
import gc
import matplotlib.pyplot as plt
try:
    # Running in PyCharm
    from AgentHomingSimple import AgentHomingSimple
    from AgentHomingPerfect import AgentHomingPerfect
    import res.colors as Color
    from Border import Border
    from Circle import StaticCircle
    from Box import StaticBox
    from MyContactListener import MyContactListener
    from Setup import *
    from Util import worldToPixels, pixelsToWorld
    import Util
    import debug_homing_simple
    from res.print_colors import printColor
    import global_homing_simple
    import res.print_colors as PrintColor
    import Global
except:
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running from command line -> Import libraries as package')
    from .AgentHomingSimple import AgentHomingSimple
    from .AgentHomingPerfect import AgentHomingPerfect
    from ..Border import Border
    from ..Circle import StaticCircle
    from ..res import colors as Color
    from .MyContactListener import MyContactListener
    from ..Setup import *
    from ..Util import worldToPixels, pixelsToWorld
    from .. import Util
    import debug_homing_simple
    from ..res.print_colors import printColor
    import global_homing_simple
    from ..res import print_colors as PrintColor
    from .. import Global


class TestbedHomingSimple(object):
    def __init__(self, screen_width, screen_height, target_fps, ppm, physics_timestep, vel_iters, pos_iters,
                 simulation_id=1, simulation_dir="./simulation_data/", simulation_file_suffix="_homing_simple",
                 file_to_load="", sim_param=None, suffix=""):

        self.simulation_id = simulation_id
        global_homing_simple.simulation_id = simulation_id
        debug_homing_simple.xprint(msg='simulation_id: {}, Starting Setup'.format(self.simulation_id))

        self.suffix = suffix

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.target_fps = target_fps
        self.ppm = ppm
        self.physics_timestep = physics_timestep
        self.vel_iters = vel_iters
        self.pos_iters = pos_iters

        self.render = sim_param.render == 'True'
        self.can_handle_events = sim_param.render == 'True'  # Can handle events only when render is True
        self.print_fps = sim_param.print_fps == 'True'
        global_homing_simple.debug = sim_param.debug == 'True'
        global_homing_simple.record = sim_param.record == 'True'
        self.fixed_ur_timestep = sim_param.fixed_ur_timestep == 'True'
        self.training = sim_param.training == 'True'
        self.collision_avoidance = sim_param.collision_avoidance == 'True'
        self.save_brain = sim_param.save_brain == 'True'
        self.load_model = sim_param.load_model == 'True'
        self.load_full_weights = sim_param.load_full_weights == 'True'
        self.load_h1_weights = sim_param.load_h1_weights == 'True'
        self.load_h1h2_weights = sim_param.load_h1h2_weights == 'True'
        self.max_timesteps = int(sim_param.max_timesteps)
        self.max_training_it = int(sim_param.max_training_it)
        self.save_network_freq = int(sim_param.save_network_freq)
        self.save_network_freq_training_it = int(sim_param.save_network_freq_training_it)
        self.wait_one_more_goal = sim_param.wait_one_more_goal == 'True'
        self.wait_learning_score_and_save_model = float(sim_param.wait_learning_score_and_save_model)
        self.exploration = sim_param.exploration == 'True'
        self.collect_experiences = sim_param.collect_experiences == 'True'
        self.save_memory_freq = int(sim_param.save_memory_freq)
        self.load_memory = int(sim_param.load_memory)
        self.record_ls = sim_param.record_ls == 'True'
        self.learning_scores = []  # initializing the mean score curve (sliding window of the rewards) with respect to timestep
        self.random_agent = sim_param.random_agent == 'True'

        # Record simulation
        self.simulation_dir = simulation_dir + "sim_logs/"
        self.simulation_file_suffix = suffix + simulation_file_suffix
        if global_homing_simple.record:
            debug_homing_simple.xprint(msg="Start recording")
            filename = global_homing_simple.fileCreate(dir=self.simulation_dir,
                   suffix=self.simulation_file_suffix + '_sim' + str(simulation_id) + '_')
            global_homing_simple.fo = open(filename, 'a')
            global_homing_simple.writer = csv.writer(global_homing_simple.fo)

        self.prev_goalreached = 0
        self.prev_goalreached_stored = False

        # Brain directory
        self.brain_dir = simulation_dir + "brain_files/" + str(simulation_id) + "/"

        # Learning score directory
        self.ls_dir = simulation_dir + "ls_files/" + str(simulation_id) + "/"

        # -------------------- Pygame Setup ----------------------

        self.deltaTime = 1.0 / target_fps  # 0.016666
        self.fps = 1.0 / self.deltaTime
        self.accumulator = 0

        self.screen = None
        if self.render:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), 0, 32)
            pygame.display.set_caption('Testbed Parameters Sharing')
        self.clock = pygame.time.Clock()
        self.myfont = pygame.font.SysFont("monospace", 15)

        self.running = True
        self.pause = False
        self.recording = False

        # -------------------- Environment and PyBox2d World Setup ----------------------

        debug_homing_simple.xprint(msg='simulation_id: {}, Physics Setup'.format(self.simulation_id))

        # Goal positions
        self.goal1 = (100, 100)
        self.goal2 = (self.screen_width - self.goal1[0], self.screen_height - self.goal1[1]) # 1180, 620

        # Create the world
        self.world = world(gravity=(0, 0), doSleep=True, contactListener=MyContactListener())  # gravity = (0, -10)

        # Border of the map
        self.border = Border(screen=self.screen, world=self.world)

        # Agents
        self.numAgents = 1  # total numbers of agents in the simulation
        self.agents = []
        for j in xrange(self.numAgents):
            # randX = random.randint(2, self.screen_width / self.ppm - 2)
            # randY = random.randint(2, self.screen_height / self.ppm - 2)
            # randAngle = degToRad(random.randint(0, 360))
            start_pos = pixelsToWorld(self.goal2)  # start from goal 2

            toGoal = Util.normalize(pixelsToWorld(self.goal1) - start_pos)
            forward = vec2(0, 1)
            angleDeg = Util.angle(forward, toGoal)
            angle = Util.degToRad(angleDeg)
            angle = -angle  # start by looking at goal1
            a = AgentHomingSimple(screen=self.screen, world=self.world, x=start_pos.x, y=start_pos.y, angle=angle,
                                  radius=1.5, id=j, numAgents=self.numAgents, training=self.training,
                                  collision_avoidance=self.collision_avoidance, random_agent=self.random_agent)
            # a = AgentHomingPerfect(screen=self.screen, world=self.world, x=start_pos.x, y=start_pos.y, angle=angle,
            #                       radius=1.5, id=j, numAgents=self.numAgents)
            self.agents.append(a)

        self.file_to_load = file_to_load  # can be default ""
        # if file_to_load != "":
        #     self.file_to_load = file_to_load
        # else:
        #     # Default file to load
        #     directory = "./brain_files/"
        #     model_file = directory + "brain" + "_model.h5"  # neural network model file
        #     file_to_load = model_file

        if file_to_load != "":
            debug_homing_simple.xprint(msg='simulation_id: {}, Loadind File Setup'.format(self.simulation_id))

        if not self.exploration:
            self.agents[0].stop_exploring()

        # Load model to agent
        if self.load_model:
            self.agents[0].load_model(self.file_to_load)
            # self.agents[0].stop_exploring()

        # Load full weights to agent
        if self.load_full_weights:
            self.agents[0].load_full_weights(self.file_to_load)
            # self.agents[0].stop_exploring()

        # Load 1st hidden layer weights to agent
        if self.load_h1_weights:
            self.agents[0].load_h1_weights(self.file_to_load)

        # Load 1st and 2nd hidden layer weights to agent
        if self.load_h1h2_weights:
            self.agents[0].load_h1h2_weights(self.file_to_load)

        # Load memory to agent
        if self.load_memory != -1:
            self.agents[0].load_memory(self.file_to_load, self.load_memory)

        # Total number of goal reached
        self.goal_reached_count = 0

        if not self.collect_experiences:
            self.agents[0].stop_collect_experiences()

        debug_homing_simple.xprint(msg='simulation_id: {}, Setup complete, Start simulation'.format(self.simulation_id))

        # self.best_ls = -1

    def draw(self):
        """
            Rendering part
        """
        if not self.render:
            return

        # Reset screen's pixels
        self.screen.fill((0, 0, 0, 0))

        # Draw goals
        goalFont = pygame.font.SysFont("monospace", 25)
        goal1Pos = 100

        # Goal 1
        pygame.draw.circle(self.screen, Color.Red, self.goal1, 20)
        self.screen.blit(goalFont.render('1', True, Color.White), (goal1Pos - 8, goal1Pos - 12))

        # Goal 2
        pygame.draw.circle(self.screen, Color.Red, self.goal2, 20)
        self.screen.blit(goalFont.render('2', True, Color.White),
                         (self.screen_width - goal1Pos - 8, self.screen_height - goal1Pos - 12))

        # Moving Objects
        for j in xrange(self.numAgents):
            self.agents[j].draw()

        # Boundary
        self.border.draw()

        # Show FPS
        Util.PrintFPS(self.screen, self.myfont, 'FPS : ' + str('{:3.2f}').format(self.fps)
                      + ' Simulation : {}'.format(self.simulation_id))

        # pygame.draw.rect(self.screen, Color.Cyan, (1260, 0, 20, 720))
        # pygame.draw.circle(self.screen, Color.Cyan, (0, 720), 20)

        # Flip the screen
        pygame.display.flip()

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
                    debug_homing_simple.xprint(msg='simulation_id: {}, Paused simulation'.format(self.simulation_id))
            if event.type == KEYDOWN and event.key == K_r:
                """
                    R: Record/ Stop record
                """
                if not global_homing_simple.record:  # Record simulation
                    debug_homing_simple.xprint(msg="Start recording")
                    global_homing_simple.record = True
                    filename = global_homing_simple.fileCreate(dir=self.simulation_dir,
                                   suffix=self.simulation_file_suffix + '_sim' + str(self.simulation_id) + '_')
                    global_homing_simple.fo = open(filename, 'a')
                    global_homing_simple.writer = csv.writer(global_homing_simple.fo)
                else:  # Stop recording
                    debug_homing_simple.xprint(msg="Stop recording")
                    global_homing_simple.record = False
                    global_homing_simple.fo.close()
            if event.type == KEYDOWN and event.key == K_s:
                """
                    S: Save model and memory
                """
                self.agents[0].save_brain(dir=self.brain_dir, suffix=self.suffix)
            if event.type == KEYDOWN and event.key == K_b:
                """
                    B: Load model
                """
                self.agents[0].load_model(self.file_to_load)
                self.agents[0].stop_training()
            if event.type == KEYDOWN and event.key == K_l:
                """
                    L: Load full weights
                """
                self.agents[0].load_full_weights(self.file_to_load)
                self.agents[0].stop_training()
            if event.type == KEYDOWN and event.key == K_w:
                """
                    W: Load h1 weights
                """
                self.agents[0].load_h1_weights(self.file_to_load)
                self.agents[0].stop_training()
            if event.type == KEYDOWN and event.key == K_z:
                """
                    Z: Load h1 h2 weights
                """
                self.agents[0].load_h1h2_weights(self.file_to_load)
                self.agents[0].stop_training()
            if event.type == KEYDOWN and event.key == K_p:  # plot Agent's learning scores
                # self.plot_learning_scores()
                pass

    def update(self):
        """
            Update game logic
        """
        # Update the agents
        for j in xrange(self.numAgents):
            self.agents[j].update()

        # Total number of goal reached
        count = 0
        for j in xrange(self.numAgents):
            count += self.agents[j].goalReachedCount
        self.goal_reached_count = count

        if self.record_ls:
            ls = self.agents[0].learning_score()  # learning score of agent 0
            self.learning_scores.append(ls)  # appending the learning score

        # Save neural networks model frequently
        if self.save_network_freq != -1:
            if Global.timestep % self.save_network_freq == 0:
                # print('timestep', Global.timestep)
                self.agents[0].save_model(dir=self.brain_dir, suffix=self.suffix)

        # Save memory frequently
        if self.save_memory_freq != -1:
            if Global.timestep % self.save_memory_freq == 0:
                self.agents[0].save_memory(dir=self.brain_dir, suffix=self.suffix)

        # Save neural networks model frequently based on training iterations
        if self.save_network_freq_training_it != -1:
            training_it = self.agents[0].training_iterations()
            if training_it != 0 and training_it % self.save_network_freq_training_it == 0:
                suff = '_' + str(training_it) + "it"
                self.agents[0].save_model(dir=self.brain_dir, suffix=self.suffix + suff)

        # Reached max number of training timesteps
        if self.max_training_it != -1 and self.agents[0].training_iterations() >= self.max_training_it:
            printColor(msg="Agent: {:3.0f}, ".format(self.agents[0].id) +
                           "{:>25s}".format("Reached {} training iterations".format(self.max_training_it)) +
                           ", tmstp: {:10.0f}".format(Global.timestep) +
                           ", t: {}".format(Global.get_time()))
            self.running = False
            return

        # # Find the highest learning score
        # ls = self.agents[0].learning_score()
        # if self.best_ls < ls:
        #     self.best_ls = ls

        # Reached max number of timesteps
        if self.max_timesteps != -1 and Global.timestep >= self.max_timesteps:

            # End if no need to wait 1 last goal
            # End if no need to reach specified learning score
            if not self.wait_one_more_goal and self.wait_learning_score_and_save_model == -1:
                self.running = False
            else:
                self.running = True

            # Wait 1 last goal
            if self.wait_one_more_goal:
                if not self.prev_goalreached_stored:
                    self.prev_goalreached = self.agents[0].goalReachedCount
                    self.prev_goalreached_stored = True

                # wait 1 more last Reached-goal
                current = self.agents[0].goalReachedCount
                if current - self.prev_goalreached == 1:
                    self.running = False
                else:
                    self.running = True

            # Wait to reach specified learning score
            if self.wait_learning_score_and_save_model != -1:
                if self.agents[0].learning_score() >= self.wait_learning_score_and_save_model:
                    printColor(msg="Agent: {:3.0f}, ".format(self.agents[0].id) +
                                   "{:>25s}".format("Reached {} learning score".format(self.agents[0].learning_score())) +
                                   ", tmstp: {:10.0f}".format(Global.timestep) +
                                   ", t: {}".format(Global.get_time()))
                    self.running = False
                    self.agents[0].save_brain(dir=self.brain_dir)
                else:
                    self.running = True

    def fps_physic_step(self):
        """
            FPS and Physic's Step Part
        """
        if self.render:
            self.deltaTime = self.clock.tick(self.target_fps) / 1000.0
            self.fps = self.clock.get_fps()

            if self.fixed_ur_timestep:
                # "Fixed your timestep" technique
                # Physics is stepped by a fixed amount i.e. 1/60s.
                # Faster machine i.e. render at 120fps -> step the physics one every 2 frames
                # Slower machine i.e. render at 30fps -> run the physics twice
                self.accumulator += self.deltaTime
                while self.accumulator >= self.physics_timestep:
                    # Physic step
                    self.world.Step(self.physics_timestep, self.vel_iters, self.pos_iters)
                    self.world.ClearForces()

                    self.accumulator -= self.physics_timestep
            else:
                # Basic Physic step
                self.world.Step(self.physics_timestep, self.vel_iters, self.pos_iters)
                self.world.ClearForces()

        elif not self.render:
            self.deltaTime = self.clock.tick() / 1000.0
            self.fps = self.clock.get_fps()

            if self.deltaTime <= self.target_fps:  # Frame is faster than target (60fps) -> simulation run faster

                # Physic step
                # self.world.Step(self.physics_timestep, self.vel_iters, self.pos_iters)
                self.world.Step(self.physics_timestep, self.vel_iters, self.pos_iters)
                self.world.ClearForces()
            elif self.fixed_ur_timestep:
                self.accumulator += self.deltaTime

                while self.accumulator >= self.physics_timestep:
                    # Physic step
                    self.world.Step(self.physics_timestep, self.vel_iters, self.pos_iters)
                    self.world.ClearForces()

                    self.accumulator -= self.physics_timestep

    def run(self):
        """
            Main Game Loop
        """
        while self.running:

            self.handle_events()

            # Pause the game
            if self.pause:
                # clock.tick compute how many milliseconds have passed since the previous call.
                # If we don't call that during pause, clock.tick will compute time spend during pause
                # thus timer is counting during simulation pause -> we want to avoid that !
                if self.render:
                    self.deltaTime = self.clock.tick(self.target_fps) / 1000.0
                elif not self.render:
                    self.deltaTime = self.clock.tick() / 1000.0
                continue  # go back to loop entry

            self.update()
            self.fps_physic_step()
            self.draw()

            if self.print_fps:
                printColor(msg='FPS : ' + str('{:3.2f}').format(self.fps))

            # Step counter
            Global.timestep += 1

            # Time counter
            global_homing_simple.timer += self.deltaTime

    def end(self):
        """
            Last function called before leaving the application
        """
        debug_homing_simple.xprint(msg='simulation_id: {}, Exit'.format(self.simulation_id))

        if global_homing_simple.record:
            global_homing_simple.record = False
            global_homing_simple.fo.close()
            debug_homing_simple.xprint(msg="Stop recording")

        # Manually deleting some objects and free memory
        del self.world
        # gc.collect()

        if self.record_ls:
            self.plot_learning_scores(save_png=False, save_csv=False)

        # print("highest ls", self.best_ls)

    def plot_learning_scores(self, save_png=False, save_csv=False):
        plt.plot(self.learning_scores)
        plt.xlabel('Training Iterations')
        plt.ylabel('Learning Score')
        plt.title('Agent\'s Learning Score over Training Iterations')
        plt.grid(True)
        plt.show(block=False)

        if save_png:
            timestring = global_homing_simple.timestr #time.strftime("%Y_%m_%d_%H%M%S")
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

    pygame.init()

    # -------------------- Simulation Parameters ----------------------

    parser = argparse.ArgumentParser(description='Testbed Parameters Sharing')
    parser.add_argument('--render', help='render the simulation', default='True')
    parser.add_argument('--print_fps', help='print fps', default='False')
    parser.add_argument('--debug', help='print simulation log', default='True')
    parser.add_argument('--record', help='record simulation log in file', default='False')
    parser.add_argument('--fixed_ur_timestep', help='fixed your timestep', default='False')
    parser.add_argument('--training', help='train agent', default='True')
    parser.add_argument('--collision_avoidance', help='agent learns collision avoidance behavior', default='True')
    parser.add_argument('--save_brain', help='save neural networks model and memory', default='False')
    parser.add_argument('--load_model',
                        help='load model to agent', default='False')
    parser.add_argument('--load_full_weights',
                        help='load full weights of neural networks from master to learning agent', default='False')
    parser.add_argument('--load_h1h2_weights',
                        help='load hidden layer 1 and 2 weights of neural networks from master to learning agent',
                        default='False')
    parser.add_argument('--load_h1_weights',
                        help='load hidden layer 1 weights of neural networks from master to learning agent',
                        default='False')
    parser.add_argument('--max_timesteps', help='maximum number of timesteps for 1 simulation', default='-1')
    parser.add_argument('--multi_simulation', help='multiple simulation at the same time', default='1')
    parser.add_argument('--save_network_freq', help='save neural networks model every defined timesteps', default='-1')
    parser.add_argument('--wait_one_more_goal', help='wait one last goal before to close application', default='False')
    parser.add_argument('--wait_learning_score_and_save_model',
                        help='wait agent to reach specified learning score before to close application', default='-1')
    parser.add_argument('--exploration', help='agent takes random action at the beginning (exploration)',
                        default='True')
    parser.add_argument('--collect_experiences', help='append a new experience to memory at each timestep',
                        default='True')
    parser.add_argument('--save_memory_freq', help='save memory every defined timesteps', default='-1')
    parser.add_argument('--load_memory', help='load defined number of experiences to agent', default='-1')
    parser.add_argument('--file_to_load', help='name of the file to load NN weights or memory', default='')
    parser.add_argument('--suffix', help='custom suffix to add', default='')
    parser.add_argument('--max_training_it', help='maximum number of training iterations for 1 simulation',
                        default='-1')
    parser.add_argument('--save_network_freq_training_it',
                        help='save neural networks model every defined training iterations', default='-1')
    parser.add_argument('--record_ls', help='record learning score of agent', default='False')
    parser.add_argument('--random_agent', help='agent is taking random action', default='False')
    args = parser.parse_args()

    multi_simulation = int(args.multi_simulation)

    # Prefix
    suffix = ""
    if args.load_full_weights == 'True':
        suffix = "loadfull"
    elif args.load_h1h2_weights == 'True':
        suffix = "loadh1h2"
    elif args.load_h1_weights == 'True':
        suffix = "loadh1"
    elif args.load_model == 'True':
        suffix = "loadmodel"

    if args.load_memory != '-1':
        suffix += "load" + args.load_memory + "xp"

    # Normal case
    if suffix == "":
        suffix = "normal"

    if args.exploration == 'False':
        suffix += "_noexplore"

    if args.random_agent == 'True':
        suffix += "_random"

    # General purpose suffix
    if args.suffix != '' and args.suffix != "":
        suffix += '_' + args.suffix

    # Run simulation
    max_timesteps = int(args.max_timesteps)
    max_training_it = int(args.max_training_it)
    total_timesteps = 0
    timestr = time.strftime("%Y%m%d_%H%M%S")

    if max_training_it != -1:
        string_counter = str(max_training_it) + "it_"
    elif max_timesteps != -1:
        string_counter = str(max_timesteps) + "tmstp_"
    else:
        string_counter = ""

    # directory_name = "./simulation_data/" + timestr + "_" + str(multi_simulation) + "sim_" + \
    #                  string_counter + suffix + "/"
    directory_name = "./simulation_data/" + suffix + "_" + string_counter + str(multi_simulation) + "sim_" + timestr + "/"

    for i in xrange(multi_simulation):
        simID = i + 1
        sys.stdout.write(PrintColor.PRINT_GREEN)
        print("Instantiate simulation: {}".format(simID))
        sys.stdout.write(PrintColor.PRINT_RESET)

        simulation = TestbedHomingSimple(SCREEN_WIDTH, SCREEN_HEIGHT, TARGET_FPS, PPM,
                                         PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS,
                                         simulation_id=simID, simulation_dir=directory_name, sim_param=args,
                                         file_to_load=args.file_to_load, suffix=suffix)
        simulation.run()
        simulation.end()
        total_timesteps += Global.timestep
        global_homing_simple.reset_simulation_global()
    print("All simulation finished")

    if args.record == 'True':
        # Save whole simulation summary in file (completion time, number of simulation, etc)
        timestr = time.strftime("%Y%m%d_%H%M%S")
        file = open(directory_name + "summary_" + timestr + ".txt", "w")
        file.write("Number of simulations: {}\n"
                   "Total simulations time: {}\n"
                   "Total timesteps: {}".format(multi_simulation, Global.get_time(), total_timesteps))
        file.close()