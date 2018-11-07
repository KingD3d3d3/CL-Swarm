

import glob
import pandas as pd
import os
from collections import deque
from sortedcontainers import SortedDict
import matplotlib.pyplot as plt
import numpy as np


def find_nearest_below(l, val):
    """
        Find nearest below value of val in list
        Using linear_search
    """
    # Find nearest in list
    prev = 0  # the minimum time is 0
    for el in l:
        if el > val:
            return prev
        prev = el
    return prev


def find_nearest_above(l, val):
    """
        Find nearest upper value of val in list
        Using linear_search
    """
    # Find nearest upper value in list
    for el in l:
        after = el
        if el >= val:
            return after

    # Boundary (no upper value)
    # return val + distance from val to the last element of list
    return val + (val - l[-1])


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


def score_goalreached(l, val, window):
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
        ratio_below = distance_ratio_below(l, last)
        return ratio_below

    # Boundary : Lower
    if first < l[0] and last < l[0]:
        ratio_below = distance_ratio_below(l, last)
        return ratio_below

    i1 = l.index(closest_upper_first)
    i2 = l.index(closest_lower_last)

    hop = i2 - i1

    ratio_above = distance_ratio_above(l, first)
    ratio_below = distance_ratio_below(l, last)

    # No elements between
    if hop == -1:
        return ratio_below

    return ratio_below + hop + ratio_above


if __name__ == '__main__':
    dir_normal = "../simulation_data/2018_05_30_101749_10sim_100000timesteps_normal/"
    dir_loadfull = "../simulation_data/2018_05_30_110442_10sim_100000timesteps_loadfullweights/"
    dir_loadh1h2 = "../simulation_data/2018_05_30_115144_10sim_100000timesteps_loadh1h2/"
    dir_loadh1 = "../simulation_data/2018_05_30_135529_10sim_100000timesteps_loadh1/"

    dir_list = [dir_normal, dir_loadfull, dir_loadh1h2, dir_loadh1]
    Y_list = []
    Ys = []

    w = 2000
    start = w
    stop = 100000
    step = 1000
    X_fixed = range(start, stop + step, step)  # List of fixed-timesteps

    for d in dir_list:
        # For each simulation file
        for f in glob.glob(d + "*.csv"):
            if os.stat(f).st_size == 0:
                print('empty file', f)
                continue
            data = pd.read_csv(f)

            # Get data Timestep of Goal-Reached
            X = list(data[data['event'].str.startswith('reached goal')]['timestep'])

            # Get scores for fixed-timesteps
            Y = [score_goalreached(X, e, w) for e in X_fixed]

            # Append to list
            Ys.append(Y)

        # Append to list of list
        Y_list.append(Ys)
        Ys = []


    X_Final = X_fixed
    Y_normal = [sum(e) / len(e) for e in zip(*Y_list[0])]
    Y_loadfull = [sum(e) / len(e) for e in zip(*Y_list[1])]
    Y_loadh1h2 = [sum(e) / len(e) for e in zip(*Y_list[2])]
    Y_loadh1 = [sum(e) / len(e) for e in zip(*Y_list[3])]
    #Y_Final = Y_loadh1

    # Standard deviation
    Y_std = [np.std(np.asarray(e)) for e in zip(*Y_list[0])]
    Y_var = [np.var(np.asarray(e)) for e in zip(*Y_list[0])]

    # Plot fixed-timestep key-values
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(X_Final, Y_normal, label='normal training')
    ax.plot(X_Final, Y_loadfull, label='load full weights')
    ax.plot(X_Final, Y_loadh1h2, label='load h1 h2')
    ax.plot(X_Final, Y_loadh1, label='load h1')

    #ax.fill_between(X_Final, np.array(Y_Final) + np.array(Y_std), np.array(Y_Final) - np.array(Y_std), facecolor='blue', alpha=0.3)

    ax.set_xlim(w, stop)
    ax.set_ylim(0, 4)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Goal reached')
    ax.set_title('Goal reached inside fixed window w = {} timesteps (10 simulations)'.format(w))
    ax.legend()
    ax.grid()
    plt.show()
