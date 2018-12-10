
import glob
import os
import pandas as pd
import csv
import re
from res.print_colors import *
import Global
from gymplayground.testbed_gym import TestbedGym
import gymplayground.debug_gym as debug_gym
import res.Util as Util
import gymplayground.simulation_parameters_gym as sim_param_gym

def evaluate(t_bed):
    """
        Evaluation method based on the environment of the testbed
    """
    print('evaluate environment {}'.format(t_bed.env_name))
    if t_bed.env_name == 'LunarLander-v2':
        # Average score over the last 100 episodes
        score = t_bed.agents[0].scores
        avg_score = sum(score) / len(score)
        print('average score {}'.format(avg_score))

        # Success count
        success = t_bed.agents[0].env.env.sucessful_landing_count # number of successful landing (between the 2 flags)
        t_bed.agents[0].env.env.sucessful_landing_count = 0 # reset successful landing counter
        print('success: {}'.format(success))

        return avg_score, success

    elif t_bed.env_name == 'MountainCar-v0':
        # Average score over the last 100 episodes
        score = t_bed.agents[0].scores
        avg_score = sum(score) / len(score)
        print('average score {}'.format(avg_score))

        # Success count
        threshold = -110
        success = sum(i > threshold for i in t_bed.agents[0].scores)
        print('success: {}'.format(success))

        return avg_score, success

    elif t_bed.env_name == 'CartPole-v0':
        # Average score over the last 100 episodes
        score = t_bed.agents[0].scores
        avg_score = sum(score) / len(score)
        print('average score {}'.format(avg_score))

        # Success count
        threshold = 195
        success = sum(i > threshold for i in t_bed.agents[0].scores)
        print('success: {}'.format(success))

        return avg_score, success

    else:
        return None

if __name__ == '__main__':

    # Import simulation parameters
    param = sim_param_gym.args

    param.debug = False
    param.render = False
    param.training = False
    param.max_ep = 100
    param.load_all_weights = True
    param.collect_experiences = False
    param.solved_score = 100000 # just a high unreachable number so that the agent will play specified nums of episodes
    param.save_record_rpu = True

    dir_name = param.dir_name # Input directory
    dir_name = os.path.abspath(dir_name) + '/'
    if not os.path.isdir(os.path.dirname(dir_name)):
        sys.exit('Not a directory: {}'.format(dir_name))
    # ----------------------------------------------------------------------

    print('Folders to visit')
    print([x[0] for x in os.walk(dir_name)])

    # Create Testbed
    testbed = TestbedGym(sim_param=param)

    # Recursively go to each folder of directory
    for x in os.walk(dir_name):

        curr_dir = x[0] + '/'

        fo = None  # file object to open file for recording
        writer = None  # writer object to record events
        filename = ''

        if glob.glob(curr_dir + "*.h5"):
            # create csv file
            filename = curr_dir + Util.get_time_string() + "_" + str(param.max_ep) + "ep" + "_eval.csv"
            print('csv file: {}'.format(filename))
            fo = open(filename, 'a')
            writer = csv.writer(fo)

        # For each saved Neural Networks model
        for f in sorted(glob.glob(curr_dir + "*.h5")):

            print('') # newline print
            debug_gym.xprint(color=PRINT_GREEN, msg="Run NN file: {}".format(f))

            # Simulation lifecycle
            testbed.setup_simulations(file_to_load=f)
            testbed.run_simulation()
            testbed.end_simulation()

            nn_file = os.path.basename(f) # f gives the whole path, let's save only the filename
            nn_file = re.sub(r'.*_(?P<episode>\d+)ep_.*', r'\g<episode>', nn_file) # extract the episode number
            eval_avg_score, eval_success = evaluate(testbed)
            seed = testbed.seed_list[0] # only 1 agent in run parallel univ

            msg_csv = (nn_file, str(eval_avg_score), str(eval_success), seed)
            writer.writerow(msg_csv) # Append score to csv file

        # Close file properly
        if fo:
            fo.close()

        # Order the file
        if filename:
            data = pd.read_csv(filename, header=None)
            data_sorted = data.sort_values(by=0, axis=0) # sort by first column of the Dataframe
            header = ('nn_episode', 'avg_score', 'success', 'seed')
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

