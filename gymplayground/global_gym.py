
import os
import errno
import time

try:
    import Util
    import Global
except NameError as err:
    print(err, "--> our error message")
    from .. import Util
    from .. import Global

# Global fixed
record = False  # record flag, if yes record simulation's events in file
debug = True  # debug mode flag, if debug mode then print event's message in console

# Changed directly
sim_id = 0

# Variables that will be reset
timer = 0.00  # times passed since beginning of simulation
simlogs_fo = None  # file object to open file for recording
header_write = False # Write header of record file only once at the beginning of each simulation
simlogs_writer = None  # writer object to record events
event_count = 0
timestr = Util.getTimeString()
start_time = time.time()

def reset_simulation_global():
    global timer
    global header_write
    global simlogs_fo
    global simlogs_writer
    global event_count
    global timestr
    global start_time

    Global.timestep = 0  # timesteps passed since beginning of simulation
    timer = 0.00  # times passed since beginning of simulation
    header_write = False # Write header of record file only once at the beginning of each simulation
    simlogs_fo = None  # file object to open file for recording
    simlogs_writer = None  # writer object to record events
    event_count = 0
    timestr = Util.getTimeString()
    start_time = time.time()

def fileCreate(dir, suffix="", extension=".csv"):
    """
        Create record csv file
        Also create the directory if it doesn't exist
    """
    time_str = Util.getTimeString()
    filename = dir + suffix + '_' + time_str + extension

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    return filename