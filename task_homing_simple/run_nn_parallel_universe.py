from __future__ import division
import glob
import os
import sys
import csv
import pygame
import argparse
import time
try:
    # Running in PyCharm
    from Setup import *
    import res.print_colors as PrintColor
    import Global
    from testbed_homing_simple import TestbedHomingSimple
    import global_homing_simple
except:
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running from command line -> Import libraries as package')
    from .testbed_homing_simple import TestbedHomingSimple
    from ..Setup import *
    from ..res import print_colors as PrintColor
    from .. import Global
    import global_homing_simple

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
    parser.add_argument('--save_network_freq', help='save neural networks model every defined timesteps',
                        default='-1')
    parser.add_argument('--wait_one_more_goal', help='wait one last goal before to close application', default='True')
    parser.add_argument('--handle_events', help='listen to keyboard events', default='True')
    args = parser.parse_args()

    # ----------------------------------------------------------------------

    # Input directory
    dir_name = "simulation_data/2018_05_31_150133_2sim_10000timesteps/"
    brain_dir = dir_name + "brain_files/"
    brain_dir = os.path.abspath(brain_dir) + '/'
    if not os.path.isdir(os.path.dirname(brain_dir)):
        sys.exit('Not a directory: {}'.format(brain_dir))

    # File prefix
    file_prefix = "normal"
    if args.load_full_weights:
        file_prefix = "load_full"
    elif args.load_h1h2_weights:
        file_prefix = "load_h1h2"
    elif args.load_h1_weights:
        file_prefix = "load_h1"

    # Recursively go to each folder of directory
    for x in os.walk(brain_dir):

        curr_dir = x[0] + '/'

        fo = None  # file object to open file for recording
        writer = None  # writer object to record events

        if glob.glob(curr_dir + "*.h5"):
            # create csv file
            timestr = time.strftime("%Y_%m_%d_%H%M%S")
            filename = curr_dir + timestr + '_' + file_prefix + "_score.csv"
            fo = open(filename, 'a')
            writer = csv.writer(fo)

        # For each saved Neural Networks model
        count = 0
        for f in sorted(glob.glob(curr_dir + "*.h5")):

            # Run simulation
            sys.stdout.write(PrintColor.PRINT_GREEN)
            print("Reading NN file: {}".format(f))
            sys.stdout.write(PrintColor.PRINT_RESET)

            simulation = TestbedHomingSimple(SCREEN_WIDTH, SCREEN_HEIGHT, TARGET_FPS, PPM,
                                             PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS,
                                             simulation_id=1, simulation_dir="", file_to_load=f, sim_param=args)
            simulation.run()
            simulation.end()
            global_homing_simple.reset_simulation_global()

            # f gives the whole path, let's save only the filename
            nn_file = os.path.basename(f)
            msg_csv = (nn_file, str(simulation.goal_reached_count))

            # Append score to csv file
            writer.writerow(msg_csv)
            count += 1

        # Close file properly
        if fo:
            fo.close()




