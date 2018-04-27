import os
import errno
import time

timestep = 0
timer = 0.00
debug = True
record = False
fo = None
writer = None

def fileCreate():
    # Creating file (and directory if it doesn't exist)
    timestr = time.strftime("%Y_%m_%d_%H%M%S")
    #filename = "./simulation_logs/" + timestr + "_homing.txt" # txt file
    filename = "./simulation_logs/" + timestr + "_homing.csv"  # CSV file

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    return filename
