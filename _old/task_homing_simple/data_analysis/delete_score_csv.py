
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
except: NameError as err:    print(err, '--> our error message')
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running from command line -> Import libraries as package")
    from ... import Util

if __name__ == '__main__':

    # Count number of simulations for each case (for average)
    sim_count_list = []

    # Normal -- DONE !!!
    # dir_normal = "../simulation_data/good_for_analysis/Normal/normal_30000it_100sim/"

    # Random --
    # dir_random = "../simulation_data/good_for_analysis/Random/random_30000tmstp_100sim/"

    # ----------------------------------------------------------------------
    # # NN weights
    dir0 = "../simulation_data/loadout_30000it_100sim/"
    # dir1 = "../simulation_data/Parallel_univ_5percent/loadh1_30000it_100sim/"
    # dir2 = "../simulation_data/Parallel_univ_5percent/loadh1h2_30000it_100sim/"
    # dir3 = "../simulation_data/Parallel_univ_5percent/loadh1out_30000it_100sim/"
    # dir4 = "../simulation_data/Parallel_univ_5percent/loadh2_30000it_100sim/"
    # dir5 = "../simulation_data/Parallel_univ_5percent/loadh2_30000it_100sim/"
    # dir6 = "../simulation_data/Parallel_univ_5percent/normal_30000it_100sim/"


    # dir1 = "../simulation_data/good_final/Normal/normal_30000it_100sim/"

    # dir2 = "../simulation_data/load40xp_mem2_30000it_100sim_20180819_125416/"
    # dir3 = "../simulation_data/load40xp_mem3_30000it_100sim_20180819_125430/"
    # dir4 = "../simulation_data/good_for_analysis/Mem2/load10000xp_mem2_30000it_100sim/"

    # # ----------------------------------------------------------------------
    # DONE !!!
    # # # Experience Exchange Memory 1
    # dir1 = "../simulation_data/good_for_analysis/Memory1/load300xp_30000it_100sim/"
    # dir2 = "../simulation_data/good_for_analysis/Memory1/load375xp_30000it_100sim/"
    # dir_load32xp_mem1 = "../simulation_data/good_for_analysis/Memory1/load32xp_30000it_100sim/"
    # dir_load50xp_mem1 = "../simulation_data/good_for_analysis/Memory1/load50xp_30000it_100sim/"
    # dir_load75xp_mem1 = "../simulation_data/good_for_analysis/Memory1/load75xp_30000it_100sim/"
    # dir_load100xp_mem1 = "../simulation_data/good_for_analysis/Memory1/load100xp_30000it_100sim/"
    # dir_load250xp_mem1 = "../simulation_data/good_for_analysis/Memory1/load250xp_30000it_100sim/"
    # dir_load500xp_mem1 = "../simulation_data/good_for_analysis/Memory1/load500xp_30000it_100sim/"
    # dir_load1000xp_mem1 = "../simulation_data/good_for_analysis/Memory1/load1000xp_30000it_100sim/"
    # dir_load2500xp_mem1 = "../simulation_data/good_for_analysis/Memory1/load2500xp_30000it_100sim/"
    # dir_load5000xp_mem1 = "../simulation_data/good_for_analysis/Memory1/load5000xp_30000it_100sim/"
    # dir_load10000xp_mem1 = "../simulation_data/good_for_analysis/Memory1/load10000xp_30000it_100sim/"
    dir_list = [dir0,
        #         dir1,
        # dir2,
        # dir3,
        # dir4,
        # dir5,
        # dir6,
        # dir7,
        # dir8,
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


    # ---------------------- For each case (Normal, Full, Load 500 xp, etc ...) ---------------------------------
    for c in range(len(dir_list)):

        d = dir_list[c]
        brain_dir = d + "brain_files/"
        brain_dir = os.path.abspath(brain_dir) + '/'
        delete_count = 0
        if not os.path.isdir(os.path.dirname(brain_dir)):
            print("no directory: {}".format(brain_dir))

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

                os.remove(f) # delete the file
                delete_count += 1
                # print("deleted file: {}".format(f))

        print("deleted {} scores file of dir : {}".format(delete_count, d))