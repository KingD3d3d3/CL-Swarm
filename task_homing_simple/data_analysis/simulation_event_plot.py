from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Import dataset
    data = pd.read_csv("../simulation_data/2018_05_18_135051_loadh1_homing_simple.csv")

    # Get data Goal Reached
    X = data[data['event'].str.startswith('reached goal')]['goal_reached']

    # Get data Timestep2Goal
    Y = data[data['event'].str.startswith('reached goal')]['timestep_to_goal']

    plt.plot(X, Y)
    plt.xlim(xmin=1, xmax=25)
    #plt.ylim(ymin=0, ymax=5000)
    plt.xlabel('Goal Reached')
    plt.ylabel('Timesteps to goal')
    plt.title('Timesteps to goal over Goal reached')
    plt.grid(True)
    plt.show()