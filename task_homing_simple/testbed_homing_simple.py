from __future__ import division

import random
import argparse
import pygame
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, vec2)
from pygame.locals import *
#import matplotlib.pyplot as plt
import time
import os
import errno
import csv
import numpy as np
import sys

try:
    # Running in PyCharm
    from AgentHomingSimple import AgentHomingSimple
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

class TestbedParametersSharing(object):
    def __init__(self, screen_width, screen_height, target_fps, ppm, physics_timestep, vel_iters, pos_iters,
                 simulation_id=0, simulation_dir="./simulation_data/", simulation_file_extension="_homing_simple.csv"):

        self.simulation_id = simulation_id
        debug_homing_simple.xprint(msg='simulation_id: {}, Starting Setup'.format(self.simulation_id))

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.target_fps = target_fps
        self.ppm = ppm
        self.physics_timestep = physics_timestep
        self.vel_iters = vel_iters
        self.pos_iters = pos_iters

        self.render = args.render == 'True'
        self.print_fps = args.print_fps == 'True'
        global_homing_simple.debug = args.debug == 'True'
        global_homing_simple.record = args.record == 'True'
        self.fixed_ur_timestep = args.fixed_ur_timestep == 'True'
        self.training = args.training == 'True'
        self.collision_avoidance = args.collision_avoidance == 'True'
        self.save_brain = args.save_brain == 'True'
        self.save_learning_score = args.save_learning_score == 'True'
        self.load_full_weights = args.load_full_weights == 'True'
        self.load_h1_weights = args.load_h1_weights == 'True'
        self.load_h1h2_weights = args.load_h1h2_weights == 'True'
        self.max_timesteps = int(args.max_timesteps)
        self.save_network_freq = int(args.save_network_freq)

        self.learning_scores = []  # initializing the mean score curve (sliding window of the rewards) with respect to timestep

        # Record simulation
        self.simulation_dir = simulation_dir
        self.simulation_file_extension = simulation_file_extension
        if global_homing_simple.record:
            debug_homing_simple.xprint(msg="Start recording")
            filename = global_homing_simple.fileCreate(dir=self.simulation_dir, extension=self.simulation_file_extension)
            global_homing_simple.fo = open(filename, 'a')
            global_homing_simple.writer = csv.writer(global_homing_simple.fo)

        self.prev_goalreached = 0
        self.prev_goalreached_stored = False

        # Brain directory
        self.brain_dir = simulation_dir + "brain_files/" + str(simulation_id) + "/"

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

        # Goal positions
        self.goal1 = (100, 100)
        self.goal2 = (self.screen_width - self.goal1[0], self.screen_height - self.goal1[1])

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
            #randAngle = degToRad(random.randint(0, 360))
            start_pos = pixelsToWorld(self.goal2) # start from goal 2

            toGoal = Util.normalize(pixelsToWorld(self.goal1) - start_pos)
            forward = vec2(0, 1)
            angleDeg = Util.angle(forward, toGoal)
            angle = Util.degToRad(angleDeg)
            angle = -angle # start by looking at goal1
            a = AgentHomingSimple(screen=self.screen, world=self.world, x=start_pos.x, y=start_pos.y, angle=angle,
                                  radius=1.5, id=j, numAgents=self.numAgents, training=self.training,
                                  collision_avoidance=self.collision_avoidance)
            self.agents.append(a)

        # Load full weights to agent
        if self.load_full_weights:
            self.agents[0].load_weights()
            self.agents[0].stop_exploring()

        # Load full weights to agent
        if self.load_h1_weights:
            self.agents[0].load_h1_weights()

        # Load full weights to agent
        if self.load_h1h2_weights:
            self.agents[0].load_h1h2_weights()

        debug_homing_simple.xprint(msg='simulation_id: {}, Setup complete, Start simulation'.format(self.simulation_id))

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
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # The user closed the window or pressed escape
                self.running = False
            if event.type == KEYDOWN and event.key == K_p:
                self.pause = not self.pause  # Pause the game
                if self.pause:
                    debug_homing_simple.xprint(msg='simulation_id: {}, Paused simulation'.format(self.simulation_id))
            if event.type == KEYDOWN and event.key == K_r:
                if not global_homing_simple.record:  # Record simulation
                    debug_homing_simple.xprint(msg="Start recording")
                    global_homing_simple.record = True
                    filename = global_homing_simple.fileCreate(dir=self.simulation_dir, extension=self.simulation_file_extension)
                    global_homing_simple.fo = open(filename, 'a')
                    global_homing_simple.writer = csv.writer(global_homing_simple.fo)
                else:  # Stop recording
                    debug_homing_simple.xprint(msg="Stop recording")
                    global_homing_simple.record = False
                    global_homing_simple.fo.close()
            if event.type == KEYDOWN and event.key == K_s:
                self.agents[0].save_brain(dir=self.brain_dir)
            if event.type == KEYDOWN and event.key == K_b:
                self.agents[0].load_model()
                self.agents[0].stop_training()
            if event.type == KEYDOWN and event.key == K_l:
                self.agents[0].load_weights()
                self.agents[0].stop_training()
            if event.type == KEYDOWN and event.key == K_w:
                self.agents[0].load_h1_weights()
                self.agents[0].stop_training()
            if event.type == KEYDOWN and event.key == K_p:  # plot Agent's learning scores
                #self.plot_learning_scores()
                pass

    def update(self):
        """
            Update game logic
        """
        # Update the agents
        for j in xrange(self.numAgents):
            self.agents[j].update()

        #ls = self.agents[0].learning_score()  # learning score of agent 0
        #self.learning_scores.append(ls)  # appending the learning score

        # Save neural networks model frequently
        if self.save_network_freq != -1:
            if Global.timestep % self.save_network_freq == 0:
                #print('timestep', Global.timestep)
                self.agents[0].save_model(dir=self.brain_dir)

        # Reached max number of timesteps
        if self.max_timesteps != -1 and Global.timestep >= self.max_timesteps:

            if not self.prev_goalreached_stored:
                self.prev_goalreached = self.agents[0].goalReachedCount
                self.prev_goalreached_stored = True

            # wait 1 more last Reached-goal
            current = self.agents[0].goalReachedCount
            if current - self.prev_goalreached == 1:
                self.running = False

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
                #self.world.Step(self.physics_timestep, self.vel_iters, self.pos_iters)
                self.world.Step(self.physics_timestep, self.vel_iters, self.pos_iters)
                self.world.ClearForces()
            else:
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

        #self.plot_learning_scores()

    # def plot_learning_scores(self, save=False):
    #     plt.plot(self.learning_scores)
    #     plt.xlabel('Timestep')
    #     plt.ylabel('Learning Score')
    #     plt.title('Agent\'s Learning Score over Timestep')
    #     plt.grid(True)
    #     plt.show(block=False)
    #
    #     if save:
    #         timestr = global_homing_simple.timestr #time.strftime("%Y_%m_%d_%H%M%S")
    #         directory = "./learning_scores/"
    #         ls_file = directory + timestr + "_ls.csv"  # learning scores file
    #         ls_fig = directory + timestr + "_ls.png"  # learning scores figure image
    #
    #         if not os.path.exists(os.path.dirname(directory)):
    #             try:
    #                 os.makedirs(os.path.dirname(directory))
    #             except OSError as exc:  # Guard against race condition
    #                 if exc.errno != errno.EEXIST:
    #                     raise
    #
    #         plt.savefig(ls_fig)
    #
    #         header = ("timestep", "learning_scores")
    #
    #         timesteps = np.arange(1, len(self.learning_scores) + 1)
    #         ls_over_tmstp = zip(timesteps, self.learning_scores)
    #
    #         with open(ls_file, 'w') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(header)
    #             writer.writerows(ls_over_tmstp)
    #         pass


if __name__ == '__main__':

    pygame.init()

    # -------------------- Simulation Parameters ----------------------

    parser = argparse.ArgumentParser(description='Testbed Parameters Sharing')
    parser.add_argument('--render', help='render the simulation', default='True')
    parser.add_argument('--print_fps', help='print fps', default='False')
    parser.add_argument('--debug', help='print simulation log', default='True')
    parser.add_argument('--record', help='record simulation log in file', default='False')
    parser.add_argument('--fixed_ur_timestep', help='fixed your timestep', default='True')
    parser.add_argument('--training', help='train agent', default='True')
    parser.add_argument('--collision_avoidance', help='agent learns collision avoidance behavior', default='True')
    parser.add_argument('--save_brain', help='save neural networks model and memory', default='False')
    parser.add_argument('--load_full_weights',
                        help='load full weights of neural networks from master to learning agent', default='False')
    parser.add_argument('--load_h1_weights',
                        help='load hidden layer 1 weights of neural networks from master to learning agent',
                        default='False')
    parser.add_argument('--load_h1h2_weights',
                        help='load hidden layer 1 and 2 weights of neural networks from master to learning agent',
                        default='False')
    parser.add_argument('--save_learning_score', help='save learning scores and plot of agent', default='False')
    parser.add_argument('--max_timesteps', help='maximum number of timesteps for 1 simulation', default='-1')
    parser.add_argument('--multi_simulation', help='multiple simulation at the same time', default='1')
    parser.add_argument('--save_network_freq', help='save neural networks model every defined timesteps', default='-1')
    args = parser.parse_args()

    multi_simulation = int(args.multi_simulation)

    # Run simulation
    max_timesteps = int(args.max_timesteps)
    total_timesteps = 0
    timestr = time.strftime("%Y_%m_%d_%H%M%S")
    directory_name = "./simulation_data/" + timestr + "_" + str(multi_simulation) + "sim_" + str(max_timesteps) + "timesteps/"
    for i in xrange(multi_simulation):
        simID = i + 1
        sys.stdout.write(PrintColor.PRINT_GREEN)
        print("Instantiate simulation: {}".format(simID))
        sys.stdout.write(PrintColor.PRINT_RESET)

        simulation = TestbedParametersSharing(SCREEN_WIDTH, SCREEN_HEIGHT, TARGET_FPS, PPM,
                                              PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS,
                                              simulation_id=simID, simulation_dir=directory_name)
        simulation.run()
        simulation.end()
        total_timesteps += Global.timestep
        global_homing_simple.reset_simulation_global()
    print("All simulation finished")

    # Save whole simulation summary in file (completion time, number of simulation, etc)
    file = open(directory_name + "summary.txt", "w")
    file.write("Number of simulations: {}\n"
               "Total simulations time: {}\n"
               "Total timesteps: {}".format(multi_simulation, Global.get_time(), total_timesteps))
    file.close()