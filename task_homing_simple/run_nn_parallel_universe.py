from __future__ import division
import glob
import os
import sys
import csv
import pygame
import argparse
import time
import re
try:
    # Running in PyCharm
    from Setup import *
    import res.print_colors as PrintColor
    import Global
    from testbed_homing_simple import TestbedHomingSimple
    import global_homing_simple
    import Util
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
    from .. import Util

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
    parser.add_argument('--load_model',
                        help='load model to agent', default='True')
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
    parser.add_argument('--stop_exploring', help='stop exploring, only exploitation', default='False')
    args = parser.parse_args()

    # --render
    # False
    # --training
    # False
    # --collision_avoidance
    # False
    # --max_timesteps
    # 5000
    # --wait_one_more_goal
    # False
    # --handle_events
    # False

    args.render = 'False'
    args.training = 'False'
    args.collision_avoidance = 'False'
    args.max_timesteps = '100'
    args.wait_one_more_goal = 'False'
    args.handle_events = 'False'

    # ----------------------------------------------------------------------

    simulation_count = 0 # counter of number of simulations
    total_timesteps = 0

    # Input directory
    dir_name = "simulation_data/many_folder_nn_test/"
    #dir_name = dir_name + "brain_files/"
    dir_name = os.path.abspath(dir_name) + '/'
    if not os.path.isdir(os.path.dirname(dir_name)):
        sys.exit('Not a directory: {}'.format(dir_name))

    # Recursively go to each folder of directory
    for x in os.walk(dir_name):

        curr_dir = x[0] + '/'

        fo = None  # file object to open file for recording
        writer = None  # writer object to record events

        if glob.glob(curr_dir + "*.h5"):
            # create csv file
            #timestr = time.strftime("%Y_%m_%d_%H%M%S")
            #filename = curr_dir + timestr + '_' + args.max_timesteps + "tmstp" + "_score.csv"
            filename = curr_dir + Util.getTimeString() + '_' + args.max_timesteps + "tmstp" + "_score.csv"
            print('csv file: {}'.format(filename))
            fo = open(filename, 'a')
            writer = csv.writer(fo)

        # For each saved Neural Networks model
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
            total_timesteps += Global.timestep
            global_homing_simple.reset_simulation_global()

            # f gives the whole path, let's save only the filename
            nn_file = os.path.basename(f)
            nn_file = re.sub(r'.*_(?P<time>\d+)tmstp_.*', r'\g<time>', nn_file)
            msg_csv = (nn_file, str(simulation.goal_reached_count))

            # Append score to csv file
            writer.writerow(msg_csv)
            simulation_count += 1

        # Close file properly
        if fo:
            fo.close()

    # Save whole simulation summary in file (completion time, number of simulation, etc)
    file = open(dir_name + "parallel_universe_summary.txt", "w")
    file.write("Number of simulations: {}\n"
               "Total simulations time: {}\n"
               "Total timesteps: {}".format(simulation_count, Global.get_time(), total_timesteps))
    file.close()

    print("All simulation finished, Total simulations time: {}".format(Global.get_time()))
    pygame.quit()
    exit()
    sys.exit()


