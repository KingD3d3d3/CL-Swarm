
import glob
import os
import pandas as pd
import csv
import pygame
import re
try:
    # Running in PyCharm
    from Setup import *
    import res.print_colors as PrintColor
    import Global
    from testbed_homing_simple import TestbedHomingSimple
    import global_homing_simple
    import Util
    from simulation_parameters import *
except NameError as err:
    print(err, "--> our error message")
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running from command line -> Import libraries as package")
    from .testbed_homing_simple import TestbedHomingSimple
    from ..Setup import *
    from ..res import print_colors as PrintColor
    from .. import Global
    import global_homing_simple
    from .simulation_parameters import *
    from .. import Util

if __name__ == '__main__':

    # Import simulation parameters
    sim_param = args

    sim_param.debug = 'False'
    sim_param.render = 'False'
    sim_param.training = 'False'
    sim_param.max_timesteps = '10000'
    sim_param.load_full_weights = 'True' # we only need to load weights to NN, since we're not training anymore

    # ----------------------------------------------------------------------

    simulation_count = 0 # counter of number of simulations
    total_timesteps = 0

    # Input directory
    dir_name = sim_param.dir_name
    dir_name = os.path.abspath(dir_name) + '/'
    if not os.path.isdir(os.path.dirname(dir_name)):
        sys.exit('Not a directory: {}'.format(dir_name))

    print('Folders to visit')
    print([x[0] for x in os.walk(dir_name)])

    # Create Testbed
    testbed = TestbedHomingSimple(sim_param=sim_param)

    # Recursively go to each folder of directory
    for x in os.walk(dir_name):

        curr_dir = x[0] + '/'

        fo = None  # file object to open file for recording
        writer = None  # writer object to record events
        filename = ''

        if glob.glob(curr_dir + "*.h5"):
            # create csv file
            filename = curr_dir + Util.getTimeString() + "_" + args.max_timesteps + "tmstp" + "_score.csv"
            print('csv file: {}'.format(filename))
            fo = open(filename, 'a')
            writer = csv.writer(fo)

        # For each saved Neural Networks model
        for f in sorted(glob.glob(curr_dir + "*.h5")):

            # Run simulation
            sys.stdout.write(PrintColor.PRINT_GREEN)
            print("Reading NN file: {}".format(f))
            sys.stdout.write(PrintColor.PRINT_RESET)

            testbed.setup_simulation(file_to_load=f)
            testbed.run_simulation()
            testbed.end_simulation()

            total_timesteps += Global.sim_timesteps # Increment total timesteps
            global_homing_simple.reset_simulation_global() # Reset global variables

            # f gives the whole path, let's save only the filename
            nn_file = os.path.basename(f)
            # nn_file = re.sub(r'.*_(?P<time>\d+)tmstp_.*', r'\g<time>', nn_file)
            nn_file = re.sub(r'.*_(?P<time>\d+)it_.*', r'\g<time>', nn_file)
            msg_csv = (nn_file, str(testbed.goal_reached_count))

            # Append score to csv file
            writer.writerow(msg_csv)
            simulation_count += 1

        # Close file properly
        if fo:
            fo.close()

        # Order the file
        if filename:
            data = pd.read_csv(filename, header=None)
            data_sorted = data.sort_values(by=0, axis=0) # sort by first column of the Dataframe
            data_sorted.to_csv(filename, index=False, header=False) # write sorted data to the same result csv file


    # Save whole simulation summary in file (completion time, number of simulation, etc)
    timestr = time.strftime("%Y%m%d_%H%M%S")
    file = open(dir_name + "parallel_universe_summary_" + timestr + ".txt", "w")
    file.write("Number of simulations: {}\n"
               "Total simulations time: {}\n"
               "Total timesteps: {}".format(simulation_count, Global.get_time(), total_timesteps))
    file.close()

    print("All simulation finished, Total simulations time: {}".format(Global.get_time()))
    pygame.quit()
    exit()
    sys.exit()


