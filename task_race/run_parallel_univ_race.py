
import glob
import os
import pandas as pd
import csv
import re
from res.print_colors import *
import Global
from task_race.testbed_race import TestbedRace
import task_race.debug_race as debug_race
import res.Util as Util
import task_race.simulation_parameters_race as sim_param_race

def evaluate(t_bed):
    """
        Evaluation method based on the environment of the testbed
    """
    print('evaluate environment {}'.format(t_bed.env_name))

    # Timesteps for 1 episode
    timesteps = t_bed.agents[0].timesteps
    print('timesteps {}'.format(timesteps))

    if t_bed.env_name == 'RaceCircleLeft':
        # Success count
        threshold = 150
        if timesteps <= threshold:
            success = 1
        else:
            success = 0
        print('success: {}'.format(success))

        return timesteps, success

    elif t_bed.env_name == 'RaceCircleRight':
        # Success count
        threshold = 150
        if timesteps <= threshold:
            success = 1
        else:
            success = 0
        print('success: {}'.format(success))

        return timesteps, success

    elif t_bed.env_name == 'RaceCombined':
        # Success count
        threshold = 240
        if timesteps <= threshold:
            success = 1
        else:
            success = 0
        print('success: {}'.format(success))

        return timesteps, success

    else:
        return None

if __name__ == '__main__':

    # Import simulation parameters
    param = sim_param_race.args

    param.debug = False
    param.display = False
    param.training = False
    param.max_ep = 1 # 1 episode is enough since no environment randomness
    param.load_all_weights = True
    param.collect_experiences = False
    param.solved_timesteps = -1 # just a high unreachable number so that the agent will play specified nums of episodes
    param.save_record_rpu = True
    param.save_seed = False

    dir_name = param.dir_name # input directory
    dir_name = os.path.abspath(dir_name) + '/'
    if not os.path.isdir(os.path.dirname(dir_name)):
        sys.exit('Not a directory: {}'.format(dir_name))
    # ----------------------------------------------------------------------

    print('Folders to visit')
    print([x[0] for x in os.walk(dir_name)])

    # Create Testbed
    testbed = TestbedRace(sim_param=param)

    # Recursively go to each folder of directory
    for x in os.walk(dir_name):

        curr_dir = x[0] + '/'

        fo = None  # file object to open file for recording
        writer = None  # writer object to record events
        filename = ''

        if glob.glob(curr_dir + "*.h5"):
            # create csv file
            filename = curr_dir + Util.get_time_string() + "_" + str(param.max_ep) + "ep_" + testbed.env_name + "_eval.csv"
            print('csv file: {}'.format(filename))
            fo = open(filename, 'a')
            writer = csv.writer(fo)

        # For each saved Neural Networks model
        for f in sorted(glob.glob(curr_dir + "*.h5")):

            print('') # newline print
            debug_race.xprint(color=PRINT_GREEN, msg="Run NN file: {}".format(f))

            # Simulation lifecycle
            testbed.setup_simulations(file_to_load=f)
            testbed.run_simulation()
            testbed.end_simulation()

            nn_file = os.path.basename(f) # f gives the whole path, let's save only the filename
            nn_file = re.sub(r'.*_(?P<episode>\d+)ep_.*', r'\g<episode>', nn_file) # extract the episode number
            eval_timesteps, eval_success = evaluate(testbed)
            seed = testbed.seed_list[0] # only 1 agent in run parallel univ

            msg_csv = (nn_file, str(eval_timesteps), str(eval_success), seed)
            writer.writerow(msg_csv) # Append score to csv file

        # Close file properly
        if fo:
            fo.close()

        # Order the file
        if filename:
            data = pd.read_csv(filename, header=None)
            data_sorted = data.sort_values(by=0, axis=0) # sort by first column of the Dataframe
            header = ('nn_episode', 'timesteps', 'success', 'seed')
            data_sorted.to_csv(filename, index=False, header=header) # write sorted data to the same result csv file

    # -------------------------------------------------------

    print("\n_______________________")
    print("All simulation finished\n"
          "Number of simulations: {}\n".format(testbed.sim_count) +
          "Total simulations time: {}\n".format(Global.get_time()) +
          "Total simulations timesteps: {}".format(Global.timesteps))

    # Run in parallel universe summary
    testbed.sim_dir = dir_name
    testbed.save_summary(suffix='parallel_universe_')

