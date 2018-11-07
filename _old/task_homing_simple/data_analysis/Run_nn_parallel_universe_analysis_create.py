
import glob
import os
import warnings
import csv
import sys
import numpy as np
import pandas as pd
from sortedcontainers import SortedDict
from scipy.signal import savgol_filter
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
try:
    # Running in PyCharm
    import Util
except NameError as err:
    print(err, "--> our error message")
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running from command line -> Import libraries as package")
    from ... import Util

Average = True
IQM = False
SavGol = False

if __name__ == '__main__':

    # Count number of simulations for each case (for average)
    sim_count_list = []


    # ----------------------------------------------------------------------
    #
    dir1 = "../simulation_data/target1000/ExperienceExchange/Mem3/load50xp_mem3_30000it_100sim/"
    dir2 = "../simulation_data/target1000/ExperienceExchange/Mem3/load100xp_mem3_30000it_100sim/"
    dir3 = "../simulation_data/target1000/ExperienceExchange/Mem3/load250xp_mem3_30000it_100sim/"
    dir4 = "../simulation_data/target1000/ExperienceExchange/Mem3/load500xp_mem3_30000it_100sim/"

    dir5 = "../simulation_data/target1000/ExperienceExchange/Mem3/load1000xp_mem3_30000it_100sim/"
    dir6 = "../simulation_data/target1000/ExperienceExchange/Mem3/load2500xp_mem3_30000it_100sim/"
    dir7 = "../simulation_data/target1000/ExperienceExchange/Mem3/load5000xp_mem3_30000it_100sim/"
    dir8 = "../simulation_data/target1000/ExperienceExchange/Mem3/load10000xp_mem3_30000it_100sim/"


    # # ----------------------------------------------------------------------

    dir_list = [
                dir1,
        dir2,
        dir3,
        dir4,
        dir5,
        dir6,
        dir7,
        dir8,
                # dir_load32xp_mem1,
                # dir_load50xp_mem1,
                # dir_load75xp_mem1,
                #
                #
                # dir_load100xp_mem1,
                # dir_load250xp_mem1,
                # dir_load500xp_mem1,
                # dir_load1000xp_mem1,
                # dir_load2500xp_mem1,
                # dir_load5000xp_mem1,
                # dir_load10000xp_mem1,
                ]

    # X values
    start = 500
    step = 500
    stop = 30000
    X = range(start, stop + step, step)


    data = []

    # Create empty data list
    for d in dir_list:
        # Create a dico
        dico = SortedDict()
        for e in X:
            dico[e] = []
        data.append(dico)

    # ---------------------- For each case (Normal, Full, Load 500 xp, etc ...) ---------------------------------
    for c in range(len(dir_list)):

        d = dir_list[c]
        brain_dir = d + "brain_files/"
        brain_dir = os.path.abspath(brain_dir) + '/'

        if not os.path.isdir(os.path.dirname(brain_dir)):
            print("no directory: {}".format(brain_dir))

        sim_count = 0

        # Recursively go to each folder of directory
        #print([x[0] for x in os.walk(brain_dir)])
        for x in os.walk(brain_dir):

            curr_dir = x[0] + '/'

            if len(glob.glob(curr_dir + "*_score.csv")) == 0:
                continue

            # should be 1 file
            for f in sorted(glob.glob(curr_dir + "*_score.csv")):
                if os.stat(f).st_size == 0:
                    print("File: {} is empty".format(f))
                    continue

                df = pd.read_csv(f, header=None)
                if len(df) > 60:
                    print("file: {}, index num:{} , simcount:{}".format(f, len(df), sim_count))

                # Add data
                for i in range(len(df)):
                    key = df[0][i]
                    value = df[1][i]
                    data[c][key].append(value)

            sim_count += 1

        # Apply filter (mean, interquartile_mean, ...)
        mydico = data[c]
        for key in mydico:
            value = mydico[key] # list of all values for this key
            if value:

                if Average:
                    # print('apply average')
                    mydico[key] = np.mean(value) # apply the Mean
                elif IQM:
                    # print('apply IQM')
                    mydico[key] = Util.interquartile_mean(value)  # apply the IQM


        if SavGol:
            # print('apply Savgol')
            filtered = savgol_filter(mydico.values(), 11, 3)
            i = 0
            for k in mydico:
                mydico[k] = filtered[i]
                i += 1

        # ------ Save to csv file -------------
        timestr = Util.get_time_string()
        suffix = os.path.basename(os.path.normpath(d)) # get the last part of the path
        filter_type = ""
        if IQM:
            filter_type += "iqm_"
        elif Average:
            filter_type += "avg_"
        if SavGol:
            filter_type += "savgol_"
        score_file = d + filter_type + "globalscore_" + suffix + "_" + timestr + ".csv"  # neural network model file
        header = ("timestep", "goal_reached")
        with open(score_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(zip(X, data[c].values()))
        print("Created score file: {}".format(score_file))