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
    import debug_homing
    from res.print_colors import *
    from testbed_homing import TestbedHoming
    import global_homing
    import Util
    from simulation_parameters import *
except:
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running from command line -> Import libraries as package')
    from .testbed_homing import TestbedHoming
    from ..Setup import *
    from ..res import print_colors as PrintColor
    from .. import Global
    import global_homing
    from ..res import print_colors as PrintColor
    import debug_homing
    from .. import Util
    from .simulation_parameters import *

directory = "./learning_scores_hyperparam/"
h1 = -1
h2 = -1

def plot_learning_scores(y, save=False, suffix=""):
    fig, ax = plt.subplots(figsize=(8, 6))

    n = len(y)

    ax.plot(y)
    ax.set_xlim(0, n)
    ax.set_ylim(-1, 0.1)

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('IQM Learning Score')
    ax.set_title('Agent\'s Learning Score over Training Iterations')
    ax.legend(loc='lower right')
    ax.grid()

    # Compute the area using the composite trapezoidal rule.
    area = np.trapz(y)
    print(area)
    text = "Area Under the Curve : {:3.2f}".format(area)
    ax.text(0.988, 0.02, text, fontsize=12, verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.7))

    plt.show(block=False)

    if save:
        directory = "./learning_scores_hyperparam/"

        timestring = global_homing.timestr
        ls_fig = directory + suffix + "_" + timestring + "_ls.png"  # learning scores figure image

        if not os.path.exists(os.path.dirname(directory)):
            try:
                os.makedirs(os.path.dirname(directory))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        plt.savefig(ls_fig)


if __name__ == '__main__':

    # Hyperparam setting
    max_h1 = 10
    max_h2 = 10

    pygame.init()

    # Import simulation parameters
    sim_param = args
    sim_param.render = 'False'
    sim_param.record_ls = 'True' # need true to get ls of agent at each tmstp
    sim_param.max_timesteps = '50000' # '30000'
    sim_param.multi_simulation = 10

    # ----------------------------------------------------------------------

    simulation_count = 0 # counter of number of simulations
    total_timesteps = 0
    ls_list = []
    multi_simulation = sim_param.multi_simulation # int(sim_param.multi_simulation)


    # ----------------------------------------------------------------------
    # # # Test every hyperparam combinations
    # for h2 in xrange(0, max_h2 + 1):
    #     for h1 in xrange(1, max_h1 + 1):

    # Directly test
    # h1 = 1
    # h2 = 0

    # Instantiate simulation
    testbed = TestbedHoming(sim_param=sim_param)

    for i in xrange(multi_simulation):
        simID = i + 1
        debug_homing.xprint(color=PRINT_GREEN, msg="Start Simulation: {}".format(simID))

        testbed.setup_simulation(sim_id=simID, h1=h1, h2=h2)
        testbed.run_simulation()
        testbed.end_simulation()
        ls_list.append(testbed.learning_scores)

        total_timesteps += Global.timestep # Increment total timesteps
        global_homing.reset_simulation_global() # Reset global variables

    Y_IQM = [Util.interquartile_mean(e) for e in zip(*ls_list)]
    # plot_learning_scores(Y_IQM, save=True, suffix="h1_{}_h2_{}".format(h1, h2))
    plot_learning_scores(Y_IQM, save=True, suffix="h2_{}_h1_{}".format(h2, h1))

    # ----------------------------------------------------------------------




    # Save whole simulation summary in file (completion time, number of simulation, etc)
    timestr = time.strftime("%Y%m%d_%H%M%S")
    file = open(directory + "hyperparam_testing_summary_" + timestr + ".txt", "w")
    file.write("Number of simulations: {}\n"
               "Total simulations time: {}\n"
               "Total timesteps: {}".format(simulation_count, Global.get_time(), total_timesteps))
    file.close()

    # End the game !
    print("All simulation finished, Total simulations time: {}".format(Global.get_time()))
    pygame.quit()
    exit()
    sys.exit()


