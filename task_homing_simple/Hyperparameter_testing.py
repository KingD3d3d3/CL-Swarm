from __future__ import division
import glob
import os
import sys
import csv
import pygame
import argparse
import time
import re
import matplotlib.pyplot as plt
import os
import errno
import numpy as np
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

def plot_learning_scores(y, save=False):
    fig, ax = plt.subplots(figsize=(8, 6))


    n = len(y)

    # # Normalize y
    # max_reward = 0.1
    # y_normed = [e / max_reward for e in y]
    # # Normalize x
    # step=1
    # x = range(1, n + step, step)
    # x_normed = [e / n for e in x]
    # ax.plot(x_normed, y_normed)
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # area = np.trapz(y_normed, x_normed)

    ax.plot(y)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 0.1)

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Average Learning Score')
    ax.set_title('Agent\'s Learning Score over Training Iterations')
    ax.legend()
    ax.grid()

    # Compute the area using the composite trapezoidal rule.
    area = np.trapz(y)
    text = "Area Under the Curve : {:3.2f}".format(area)
    ax.text(0.988, 0.02, text, fontsize=12, verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.7))

    plt.show(block=False)

    if save:
        directory = "./learning_scores/"

        timestring = global_homing_simple.timestr
        ls_fig = directory + timestring + "_ls.png"  # learning scores figure image

        if not os.path.exists(os.path.dirname(directory)):
            try:
                os.makedirs(os.path.dirname(directory))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        plt.savefig(ls_fig)


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
    parser.add_argument('--save_learning_score', help='save learning scores and plot of agent', default='False')
    parser.add_argument('--max_timesteps', help='maximum number of timesteps for 1 simulation', default='-1')
    parser.add_argument('--multi_simulation', help='multiple simulation at the same time', default='1')
    parser.add_argument('--save_network_freq', help='save neural networks model every defined timesteps', default='-1')
    parser.add_argument('--wait_one_more_goal', help='wait one last goal before to close application', default='False')
    parser.add_argument('--wait_learning_score_and_save_model',
                        help='wait agent to reach specified learning score before to close application', default='-1')
    parser.add_argument('--handle_events', help='listen to keyboard events', default='True')
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

    # -------------------- Run NN special Parameters ----------------------
    parser.add_argument('--dir_name', help='directory name', default="")
    args = parser.parse_args()

    args.render = 'False'
    args.collision_avoidance = 'False'
    args.handle_events = 'False'
    args.record_ls = 'True'

    # ----------------------------------------------------------------------

    ls_list = []

    multi_simulation = int(args.multi_simulation)

    for i in xrange(multi_simulation):
        simID = i + 1
        sys.stdout.write(PrintColor.PRINT_GREEN)
        print("Instantiate simulation: {}".format(simID))
        sys.stdout.write(PrintColor.PRINT_RESET)

        simulation = TestbedHomingSimple(SCREEN_WIDTH, SCREEN_HEIGHT, TARGET_FPS, PPM,
                                         PHYSICS_TIME_STEP, VEL_ITERS, POS_ITERS,
                                         simulation_id=simID, simulation_dir="", sim_param=args,
                                         file_to_load=args.file_to_load, suffix="")
        simulation.run()
        simulation.end()
        ls_list.append(simulation.learning_scores)
        global_homing_simple.reset_simulation_global()

    print("All simulation finished, Total simulations time: {}".format(Global.get_time()))

    Y_avg = [sum(e) / len(e) for e in zip(*ls_list)]
    plot_learning_scores(Y_avg, save=True)


    pygame.quit()
    exit()
    sys.exit()


