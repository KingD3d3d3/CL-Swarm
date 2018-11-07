
import glob
import pandas as pd
import os
from collections import deque
from sortedcontainers import SortedDict
import matplotlib.pyplot as plt
import numpy as np
import sys
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

def find_nearest_below(l, val):
    """
        Find nearest below value of val in list
        Using linear_search
    """
    # Find nearest in list
    prev = 0  # the minimum time is 0
    for e in l:
        if e > val:
            return prev
        prev = e
    return prev


def find_nearest_above(l, val):
    """
        Find nearest upper value of val in list
        Using linear_search
    """
    # Find nearest upper value in list
    for e in l:
        after = e
        if e >= val:
            return after

    # Boundary (no upper value)
    # return val + distance from val to the last element of list
    return val + (val - l[-1])


def distance_ratio(l, val):
    lower = find_nearest_below(l, val)
    upper = find_nearest_above(l, val)

    total_dist = upper - lower

    dist_lower = val - lower
    dist_upper = upper - val

    lower_ratio = dist_lower / total_dist
    upper_ratio = dist_upper / total_dist

    print('lower_ratio', lower_ratio)
    print('upper_ratio', upper_ratio)


def distance_ratio_above(l, val):
    # Exception : input is negative
    if val < 0:
        raise ValueError("val: {} is negative".format(val))

    lower = find_nearest_below(l, val)
    upper = find_nearest_above(l, val)

    # Exception
    if lower == upper:
        return 0.

    total_dist = upper - lower
    dist_upper = upper - val

    upper_ratio = dist_upper / total_dist
    return upper_ratio


def distance_ratio_below(l, val):
    # Exception : input is negative
    if val < 0:
        raise ValueError("val: {} is negative".format(val))

    lower = find_nearest_below(l, val)
    upper = find_nearest_above(l, val)

    # Exception
    if lower == upper:
        return 0.

    total_dist = upper - lower
    dist_lower = val - lower

    lower_ratio = dist_lower / total_dist
    return lower_ratio


def score(l, val, window):
    # Exception : input is negative
    if val < 0:
        raise ValueError("val: {} is negative".format(val))

    if window > val:
        raise ValueError("cannot start giving score at val: {} with window: {}".format(val, window))

    last = val
    first = last - window
    closest_upper_first = find_nearest_above(l, first)
    closest_lower_last = find_nearest_below(l, last)
    # Boundary : Upper
    if last > l[-1] and first > l[-1]:
        # print("Boundary : Upper")
        ratio_below = distance_ratio_below(l, last)
        return ratio_below

    # Boundary : Lower
    if first < l[0] and last < l[0]:
        # print("Boundary : Lower")
        ratio_below = distance_ratio_below(l, last)
        return ratio_below

    i1 = l.index(closest_upper_first)
    i2 = l.index(closest_lower_last)

    hop = i2 - i1

    ratio_above = distance_ratio_above(l, first)
    ratio_below = distance_ratio_below(l, last)

    # No elements between
    if hop == -1:
        # print("no elements between")
        return ratio_below

    return ratio_below + hop + ratio_above

def get_score_list(dir_name):

    Y_s = []

    simulation_count = 0
    # Recursively go to each folder of directory
    for x in os.walk(dir_name):

        curr_dir = x[0] + '/'

        # For each simulation file
        for f in glob.glob(curr_dir + "*.csv"):
            simulation_count += 1
            if os.stat(f).st_size == 0:
                print('empty file', f)
                continue
            data = pd.read_csv(f)

            # Get data Timestep of Goal-Reached
            X = list(data[data['event'].str.startswith('reached goal')]['timestep'])

            # Get scores for fixed-timesteps
            Y = [score(X, e, W) for e in X_fixed]

            # Append to list
            Y_s.append(Y)

    print("Score calculated")

    print('{}, simulation_count: {}'.format(dir_name, simulation_count))

    if simulation_count == 0:
        print('no csv file in the directory {}'.format(dir_name))
        sys.exit()

    # Ys = [sum(e) / len(e) for e in zip(*Y_s)] # average

    Ys = [Util.interquartile_mean(e) for e in zip(*Y_s)]



    # Get min and max
    # Y_min = [min(e) for e in zip(*Ys)]
    # Y_max = [max(e) for e in zip(*Ys)]
    #
    # # # Standard deviation
    # Y_std = [np.std(np.asarray(e)) for e in zip(*Ys)]
    # Y_var = [np.var(np.asarray(e)) for e in zip(*Ys)]

    print("max value : {}".format(np.max(Ys)))
    print("mean value : {}".format(np.mean(Ys)))
    print("min value : {}".format(np.min(Ys)))

    return Ys




if __name__ == '__main__':

    # Import data
    dir_name = "../simulation_data/normal_GR_100batch_50000it_50sim_20181007_103037/"
    # dir_name = "../simulation_data/normal_perfect_100000tmstp_10sim_20180711_140535/"
    # dir_name = "../simulation_data/master_final_normal_100000tmstp_1sim_20180712_171209/sim_logs/"
    # dir_random = "../simulation_data/good_for_analysis/GR_Fixedwindow/Random_GR/"

    W = 10000 # 10000 # fixed window size # 2000
    start = W
    stop = 50000 #100000 # end of simulation (max number of timesteps)
    step = 1000 # 1000
    X_fixed = range(start, stop + step, step)  # List of fixed-timesteps

    # Get avg score list
    Y_normal = get_score_list(dir_name)
    # Y_random = get_score_list(dir_random)

    # Plot fixed-timestep key-values
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(X_fixed, Y_normal, label='Normal')


    # ax.fill_between(X_Final, np.array(Y_Final) + np.array(Y_std), np.array(Y_Final) - np.array(Y_std),
    #                 facecolor='blue', alpha=0.3)
    # ax.fill_between(X_Final, Y_min, Y_max, facecolor='yellow', alpha=0.1)

    # Upper bound - Perfect agent
    upper_bound = round(W / 360, 1)  # 27.8 GR in 10.000 tmstp
    print("Upper bound for W = {} is : {}".format(W, upper_bound))
    ax.axhline(y=upper_bound, color='black')
    ax.annotate('Theoretical limit', xy=(start * 1.8, upper_bound + 0.12), xycoords='data')  # x = stop * 0.81
    ax.text(-0.055, 0.94, str(upper_bound), verticalalignment='top', horizontalalignment='left', transform=ax.transAxes)
    # x = start * 1.8

    # Random agent
    random_bound = 2.21  # 25.6 GR in 10.000 tmstp
    print("Random agent bound for W = {} is : {}".format(W, random_bound))
    ax.axhline(y=random_bound, color='black')
    ax.annotate('Random walk', xy=(start * 1.8, random_bound + 0.12), xycoords='data')
    ax.text(-0.055, 0.085, str(random_bound), verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes)


    # Baseline - Random agent
    # ax.plot(X_fixed, Y_random, label='Random')

    ax.set_xlim(W, stop)
    ax.set_ylim(0, 30)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Goals reached')
    ax.set_title('Goals reached inside fixed window w = {}'.format(W))

    ax.grid()
    ax.legend(loc='lower right')
    print("########################################")
    print("Show curve")
    plt.show()
    #plt.savefig(ls_fig)