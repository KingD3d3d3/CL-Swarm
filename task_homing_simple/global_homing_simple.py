import os
import errno
import time

timestep = 1  # 0 # timesteps passed since beginning of simulation
timer = 0.00  # times passed since beginning of simulation
debug = True  # debug mode flag, if debug mode then print event's message in console
record = False  # record flag, if yes record simulation's events in file
fo = None  # file object to open file for recording
writer = None  # writer object to record events
event_count = 0

def fileCreate():
    """
        Create record csv file
        Also create the /simulation_logs/ directory if it doesn't exist
    """
    timestr = time.strftime("%Y_%m_%d_%H%M%S")
    filename = "./simulation_logs/" + timestr + "_homing_simple.csv"  # CSV file

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    return filename
