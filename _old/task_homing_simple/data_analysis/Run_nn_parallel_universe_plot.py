
import glob
import os
import warnings

import pandas as pd
from collections import deque
from sortedcontainers import SortedDict
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.signal import savgol_filter
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
try:
    # Running in PyCharm
    import Util
except NameError as err:
    print(err, '--> our error message')
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running from command line -> Import libraries as package")
    from ... import Util


def get_x(y, list_x, list_y):
    if len(list_x) != len(list_y):
        print('list_X and list_Y not same size : {} and {}'.format(len(list_x), len(list_y)))
        return

    y_a = list_y[0]
    x_a = list_x[0]
    y_b = 0
    x_b = 0
    for i in range(len(list_y)):

        # equal
        if y == list_y[i]:
            return list_x[i]

        if y > list_y[i]:
            y_a = list_y[i]
            x_a = list_x[i]
            continue

        if y < list_y[i]:
            y_b = list_y[i]
            x_b = list_x[i]
            #             print('x_b: {}, y_b: {}'.format(x_b,y_b))
            break

    m = (y_b - y_a) / (x_b - x_a)
    p = y_b - (m * x_b)

    x = (y - p) / m
    return x

if __name__ == '__main__':


    # X values
    start = 500
    step = 500
    stop = 30000
    X = range(start, stop + step, step)


    # Target 1000 Exploit 5 percent
    # ----------------------------------------------------------------------



    # NN Weights -----------------------------------
    dir_normal = "../simulation_data/target1000/Normal/normal_30000it_100sim/"         # Normal
    dir_loadfull = "../simulation_data/target1000/NN_weights/loadfull_30000it_100sim/"     # Full
    dir_loadh1 = "../simulation_data/target1000/NN_weights/loadh1_30000it_100sim/"         # h1
    dir_loadh1h2 = "../simulation_data/target1000/NN_weights/loadh1h2_30000it_100sim/"     # h1h2
    dir_loadh1out = "../simulation_data/target1000/NN_weights/loadh1out_30000it_100sim/"   # h1out
    dir_loadh2 = "../simulation_data/target1000/NN_weights/loadh2_30000it_100sim/"         # h2
    dir_loadh2out = "../simulation_data/target1000/NN_weights/loadh2out_30000it_100sim/"   # h2out
    dir_loadout = "../simulation_data/target1000/NN_weights/loadout_30000it_100sim/"       # out

    label0 = 'Individual'
    label1 = 'All weights'
    label2 = 'Load h1'
    label3 = 'Load h1h2'
    label4 = 'Load h1out'
    label5 = 'Load h2'
    label6 = 'Load h2out'
    label7 = 'Load out'

    dir_list = [dir_normal,
                dir_loadfull,
                dir_loadh1,
                dir_loadh1h2,
                dir_loadh1out,
                dir_loadh2,
                dir_loadh2out,
                dir_loadout
                ]


    # # Mem 1 -----------------------------------
    # dir_normal = "../simulation_data/target1000/Normal/normal_30000it_100sim/"   # Normal
    # dir_loadfull = "../simulation_data/target1000/NN_weights/loadfull_30000it_100sim/"  # Full
    #
    # dir_load10000xp_mem1 = "../simulation_data/target1000/ExperienceExchange/Mem1/load10000xp_mem1_30000it_100sim/"
    # dir_load5000xp_mem1 = "../simulation_data/target1000/ExperienceExchange/Mem1/load5000xp_mem1_30000it_100sim/"
    # dir_load2500xp_mem1 = "../simulation_data/target1000/ExperienceExchange/Mem1/load2500xp_mem1_30000it_100sim/"
    # dir_load1000xp_mem1 = "../simulation_data/target1000/ExperienceExchange/Mem1/load1000xp_mem1_30000it_100sim/"
    #
    # dir_load500xp_mem1 = "../simulation_data/target1000/ExperienceExchange/Mem1/load500xp_mem1_30000it_100sim/"
    # dir_load250xp_mem1 = "../simulation_data/target1000/ExperienceExchange/Mem1/load250xp_mem1_30000it_100sim/"
    # dir_load100xp_mem1 = "../simulation_data/target1000/ExperienceExchange/Mem1/load100xp_mem1_30000it_100sim/"
    # dir_load50xp_mem1 = "../simulation_data/target1000/ExperienceExchange/Mem1/load50xp_mem1_30000it_100sim/"
    #
    # label0 = 'Individual'
    # label1 = 'All weights'
    # label2 = 'Load Mem1 10000xp'
    # label3 = 'Load Mem1 5000xp'
    # label4 = 'Load Mem1 2500xp'
    # label5 = 'Load Mem1 1000xp'
    # label6 = 'Load Mem1 500xp'
    # label7 = 'Load Mem1 250xp'
    # label8 = 'Load Mem1 100xp'
    # label9 = 'Load Mem1 50xp'
    #
    # dir_list = [dir_normal,
    #             dir_loadfull,
    #             dir_load10000xp_mem1,
    #             dir_load5000xp_mem1,
    #             dir_load2500xp_mem1,
    #             dir_load1000xp_mem1,
    #
    #             dir_load500xp_mem1,
    #             dir_load250xp_mem1,
    #             dir_load100xp_mem1,
    #             dir_load50xp_mem1
    #             ]


    # # Mem 2 -----------------------------------
    # dir_normal = "../simulation_data/target1000/Normal/normal_30000it_100sim/"  # Normal
    # dir_loadfull = "../simulation_data/target1000/NN_weights/loadfull_30000it_100sim/"  # Full
    #
    # dir_load10000xp_mem2 = "../simulation_data/target1000/ExperienceExchange/Mem2/load10000xp_mem2_30000it_100sim/"
    # dir_load5000xp_mem2 = "../simulation_data/target1000/ExperienceExchange/Mem2/load5000xp_mem2_30000it_100sim/"
    # dir_load2500xp_mem2 = "../simulation_data/target1000/ExperienceExchange/Mem2/load2500xp_mem2_30000it_100sim/"
    # dir_load1000xp_mem2 = "../simulation_data/target1000/ExperienceExchange/Mem2/load1000xp_mem2_30000it_100sim/"
    #
    # dir_load500xp_mem2 = "../simulation_data/target1000/ExperienceExchange/Mem2/load500xp_mem2_30000it_100sim/"
    # dir_load250xp_mem2 = "../simulation_data/target1000/ExperienceExchange/Mem2/load250xp_mem2_30000it_100sim/"
    # dir_load100xp_mem2 = "../simulation_data/target1000/ExperienceExchange/Mem2/load100xp_mem2_30000it_100sim/"
    # dir_load50xp_mem2 = "../simulation_data/target1000/ExperienceExchange/Mem2/load50xp_mem2_30000it_100sim/"
    #
    # label0 = 'Individual'
    # label1 = 'All weights'
    # label2 = 'Load Mem2 10000xp'
    # label3 = 'Load Mem2 5000xp'
    # label4 = 'Load Mem2 2500xp'
    # label5 = 'Load Mem2 1000xp'
    # label6 = 'Load Mem2 500xp'
    # label7 = 'Load Mem2 250xp'
    # label8 = 'Load Mem2 100xp'
    # label9 = 'Load Mem2 50xp'
    #
    # dir_list = [dir_normal,
    #             dir_loadfull,
    #             dir_load10000xp_mem2,
    #             dir_load5000xp_mem2,
    #             dir_load2500xp_mem2,
    #             dir_load1000xp_mem2,
    #
    #             dir_load500xp_mem2,
    #             dir_load250xp_mem2,
    #             dir_load100xp_mem2,
    #             dir_load50xp_mem2
    #             ]


    # # Mem 3 -----------------------------------
    # dir_normal = "../simulation_data/target1000/Normal/normal_30000it_100sim/"  # Normal
    # dir_loadfull = "../simulation_data/target1000/NN_weights/loadfull_30000it_100sim/"  # Full
    #
    # dir_load10000xp_mem3 = "../simulation_data/target1000/ExperienceExchange/Mem3/load10000xp_mem3_30000it_100sim/"
    # dir_load5000xp_mem3 = "../simulation_data/target1000/ExperienceExchange/Mem3/load5000xp_mem3_30000it_100sim/"
    # dir_load2500xp_mem3 = "../simulation_data/target1000/ExperienceExchange/Mem3/load2500xp_mem3_30000it_100sim/"
    # dir_load1000xp_mem3 = "../simulation_data/target1000/ExperienceExchange/Mem3/load1000xp_mem3_30000it_100sim/"
    #
    # dir_load500xp_mem3 = "../simulation_data/target1000/ExperienceExchange/Mem3/load500xp_mem3_30000it_100sim/"
    # dir_load250xp_mem3 = "../simulation_data/target1000/ExperienceExchange/Mem3/load250xp_mem3_30000it_100sim/"
    # dir_load100xp_mem3 = "../simulation_data/target1000/ExperienceExchange/Mem3/load100xp_mem3_30000it_100sim/"
    # dir_load50xp_mem3 = "../simulation_data/target1000/ExperienceExchange/Mem3/load50xp_mem3_30000it_100sim/"
    #
    # label0 = 'Individual'
    # label1 = 'All weights'
    # label2 = 'Load Mem3 10000xp'
    # label3 = 'Load Mem3 5000xp'
    # label4 = 'Load Mem3 2500xp'
    # label5 = 'Load Mem3 1000xp'
    # label6 = 'Load Mem3 500xp'
    # label7 = 'Load Mem3 250xp'
    # label8 = 'Load Mem3 100xp'
    # label9 = 'Load Mem3 50xp'
    #
    # dir_list = [dir_normal,
    #             dir_loadfull,
    #             dir_load10000xp_mem3,
    #             dir_load5000xp_mem3,
    #             dir_load2500xp_mem3,
    #             dir_load1000xp_mem3,
    #
    #             dir_load500xp_mem3,
    #             dir_load250xp_mem3,
    #             dir_load100xp_mem3,
    #             dir_load50xp_mem3
    #             ]


    data = []

    conv_95 = 23.18 * 0.95
    print('conv_95', conv_95)
    prefix = '' # default prefix

    # ---------------------- For each case (Normal, Full, Load 500 xp, etc ...) ---------------------------------
    for c in range(len(dir_list)):

        d = dir_list[c]


        curr_dir = os.path.abspath(d) + '/'
        # print('curr_dir', curr_dir)

        csv_file = glob.glob(curr_dir + "*iqm_savgol_globalscore*")[0] # IQM Savgol
        prefix = ' (SavGol IQM)'
        #
        # csv_file = glob.glob(curr_dir + "*iqm_globalscore*")[0] # IQM
        # prefix = ' (IQM)'
        #
        # csv_file = glob.glob(curr_dir + "*avg_globalscore*")[0] # Average
        # prefix = ' (Avg)'

        df = pd.read_csv(csv_file)
        data.append(df)

        T_conv_95 = get_x(conv_95, data[c]['timestep'], data[c]['goal_reached'])
        print('T_conv_95 of {} is {}'.format(d, T_conv_95))



    # Find GR convergence of normal training set
    conv = np.mean(data[0]['goal_reached'][40:60])
    max = np.max(data[0]['goal_reached'][40:60])
    min = np.min(data[0]['goal_reached'][40:60])
    print('convergence', conv)
    print('max', max)
    print('min', min)
    # Savgol IQM => ('convergence', 23.187448051948017), ('max', '23.461414141414146'), ('min', '22.901818181818147') =>1.2% error
    # IQM => ('convergence', 23.196), ('max', '23.58'), ('min', '22.7') => 2% error

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))  # 8, 6  20, 14
    # ax.plot(data[0]['timestep'], data[0]['goal_reached'], label=label0 + prefix, color='tab:blue', linewidth=6.0)  # Normal linewidth=6.0
    # ax.plot(data[1]['timestep'], data[1]['goal_reached'], label=label1 + prefix, color='tab:red', linewidth=6.0) # Full weights
    ax.plot(data[2]['timestep'], data[2]['goal_reached'], label=label2 + prefix, color='tab:green') # h1h2
    ax.plot(data[3]['timestep'], data[3]['goal_reached'], label=label3 + prefix, color='tab:orange') # h1
    ax.plot(data[4]['timestep'], data[4]['goal_reached'], label=label4 + prefix, color='tab:brown') # h1out
    ax.plot(data[5]['timestep'], data[5]['goal_reached'], label=label5 + prefix, color='tab:olive') # h2

    ax.plot(data[6]['timestep'], data[6]['goal_reached'], label=label6 + prefix, color='tab:pink') # h2out
    ax.plot(data[7]['timestep'], data[7]['goal_reached'], label=label7 + prefix, color='tab:gray') # out
    # ax.plot(data[8]['timestep'], data[8]['goal_reached'], label=label8 + prefix, color='tab:purple')
    # ax.plot(data[9]['timestep'], data[9]['goal_reached'], label=label9 + prefix, color='tab:cyan')

    ax.plot(data[1]['timestep'], data[1]['goal_reached'], label=label1 + prefix, color='tab:red', linewidth=6.0)  # Full weights
    ax.plot(data[0]['timestep'], data[0]['goal_reached'], label=label0 + prefix, color='xkcd:black', linewidth=6.0)  # Normal linewidth=6.0

    # ax.plot(data[10]['timestep'], data[10]['goal_reached'], label=label10 + prefix)
    # ax.plot(data[11]['timestep'], data[11]['goal_reached'], label=label11 + prefix)
    # ax.plot(data[12]['timestep'], data[12]['goal_reached'], label=label12 + prefix)
    # ax.plot(data[13]['timestep'], data[13]['goal_reached'], label=label13 + prefix)
    # ax.plot(data[14]['timestep'], data[14]['goal_reached'], label=label14 + prefix)
    # ax.plot(data[15]['timestep'], data[15]['goal_reached'], label=label15 + prefix)
    # ax.plot(data[12]['timestep'], data[12]['goal_reached'], label=label12 + prefix)
    # ax.plot(data[16]['timestep'], data[16]['goal_reached'], label=label16 + prefix)
    # ax.plot(data[17]['timestep'], data[17]['goal_reached'], label=label17 + prefix)
    # ax.plot(data[16]['timestep'], data[16]['goal_reached'], label=label18 + prefix)

    # Starting line
    ax.axvline(x=500, color='black', linestyle='--')
    ax.text(0.01, -0.052, '500', verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes)

    # Upper bound - Theoretical limit
    W = 10000
    upper_bound = round(W / 360, 1)  # 27.8 GR in 10.000 tmstp
    print("Upper bound for W = {} is : {}".format(W, upper_bound))
    ax.axhline(y=upper_bound, color='black')
    ax.annotate('Theoretical limit', xy=(start * 15, upper_bound + 0.12), xycoords='data') # x = stop * 0.81
    ax.text(-0.055, 0.94, str(upper_bound), verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
    # x = start * 1.8

    # Random walk
    random_bound = 2.21  # 25.6 GR in 10.000 tmstp
    print("Random agent bound for W = {} is : {}".format(W, random_bound))
    ax.axhline(y=random_bound, color='black')
    ax.annotate('Random walk', xy=(start * 15, random_bound + 0.12), xycoords='data')
    ax.text(-0.055, 0.085, str(random_bound), verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)


    ax.set_ylim(0, 30)
    ax.set_xlim(0, stop)
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Goals reached')
    ax.set_title("Goals reached during 10000 timesteps for each saved neural networks")
    ax.legend(loc='lower right')
    ax.grid()
    plt.show()