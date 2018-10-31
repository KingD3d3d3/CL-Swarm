
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
simulation_id = 1

# Variables that will be reset
timer = 0.00  # times passed since beginning of simulation
simlogs_fo = None  # file object to open file for recording
header_write = False # Write header of record file only once at the beginning of each simulation
simlogs_writer = None  # writer object to record events
sim_event_count = 0
timestr = Util.get_time_string()

def reset_simulation_global():
    global timer
    global header_write
    global simlogs_fo
    global simlogs_writer
    global sim_event_count
    global timestr

    Global.sim_timesteps = 0  # timesteps passed since beginning of simulation
    timer = 0.00  # times passed since beginning of simulation
    header_write = False # Write header of record file only once at the beginning of each simulation
    simlogs_fo = None  # file object to open file for recording
    simlogs_writer = None  # writer object to record events
    event_count = 0
    timestr = Util.get_time_string()

def fileCreate(dir, suffix="", extension=".csv"):
    """
        Create record csv file
        Also create the directory if it doesn't exist
    """
    time_str = Util.get_time_string()
    filename = dir + suffix + '_' + time_str + extension

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    return filename