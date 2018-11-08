"""
    Print log, record simulation event
    Print simulation setup
"""
import os
import errno
import task_race.global_race as global_race
from res.print_colors import *
import res.Util as Util
import Global

header = (
    'agent',
    'episode',
    'timesteps',
    'avg_timestep',
    'd2g',
    'tot_timesteps',
    'sim_t'
)

def print_event(agent, episode, tmstp, avg_tmstp, d2g, tot_tmstp, record=False, debug=True):
    """
        Write doc
    """
    msg_debug = (
        "sim_id: {:3.0f}, ".format(global_race.sim_id) +
        "agent: {:3.0f}, ".format(agent.id) +
        "episode: {:5.0f}, ".format(episode) +
        "tmstp: {:4.0f}, ".format(tmstp) +
        "avg_tmstp: {:6.2f}, ".format(avg_tmstp) +
        "d2g: {:3.0f}, ".format(d2g) +
        "tot_tmstp: {:8.0f}, ".format(tot_tmstp) +
        "training_it: {:8.0f}, ".format(agent.brain.training_it) +
        "sim_t: {}, ".format(Global.get_sim_time()) +
        "global_t: {}, ".format(Global.get_time()) +
        "world_t: {}".format(Util.get_time_string2())
    )

    sim_t = Global.get_sim_time_in_seconds()
    msg_csv = (
        agent.id,
        episode,
        tmstp,
        avg_tmstp,
        d2g,
        tot_tmstp,
        sim_t
    )

    # Record data
    if global_race.record and record:
        global_race.simlogs_writer.writerow(msg_csv)

    # Print in debug mode
    if global_race.debug and debug:
        print(msg_debug)


def xprint(color=PRINT_BLUE, msg=""):
    print_color(color=color, msg="sim_id: {:3.0f}, ".format(global_race.sim_id) +
                                 "{: <35s}, ".format(msg) +
                                 "sim_t: {}, ".format(Global.get_sim_time()) +
                                 "global_t: {}, ".format(Global.get_time()) +
                                 "world_t: {}".format(Util.get_time_string2()))

def create_record_file(dir, suffix=""):
    """
        Create record CSV file and return it
        Also create the directory if it doesn't exist
    """
    time_str = Util.get_time_string()
    filename = dir + time_str + '_' + suffix + "_rec.csv"

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    return filename
