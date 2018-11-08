"""
    Global variables that are mainly use in the debug_gym for logging simulation event
"""
import time
import res.Util as Util
import Global

# Global fixed
record = False  # record flag, if yes record simulation's events in file
debug = True  # debug mode flag, if debug mode then print event's message in console

# Changed directly
sim_id = 0

# Variables that will be reset
simlogs_fo = None  # file object to open file for recording
header_write = False # Write header of record file only once at the beginning of each simulation
simlogs_writer = None  # writer object to record events
sim_event_count = 0
timestr = Util.get_time_string()

def reset_simulation_global():
    global header_write
    global simlogs_fo
    global simlogs_writer
    global sim_event_count
    global timestr

    # Global.sim_timesteps = 0  # timesteps passed since beginning of simulation
    header_write = False # Write header of record file only once at the beginning of each simulation
    simlogs_fo = None  # file object to open file for recording
    simlogs_writer = None  # writer object to record events
    sim_event_count = 0
    timestr = Util.get_time_string()
    Global.sim_start_time = time.time()
