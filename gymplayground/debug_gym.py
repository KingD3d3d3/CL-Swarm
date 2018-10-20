"""
    Print log, record simulation event
    Print simulation setup
"""
import os
import errno

try:
    # Running in PyCharm
    import gymplayground.global_gym as global_gym
    from res.print_colors import *
    import Util
    import Global
except NameError as err:
    print(err, "--> our error message")
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running from command line -> Import libraries as package")
    from . import global_gym
    from ..res.print_colors import *
    from .. import Util
    from .. import Global

header = (
    'agent',
    'episode',
    'score',
    'avg_score',
    'timesteps',
    'tot_timesteps'
)


def print_event(agent, episode, score, avg_score, timesteps, tot_timesteps, record=False, debug=True):
    """
        agent           : agent
        episode         : event's message
        score           : score of the episode
        avg_score       : average score over the last 100 episodes
        timesteps        : number of timesteps passed for the episode
        tot_timesteps   : total number of timesteps passed
    """
    msg_event = (
        "sim_id: {:3.0f}, ".format(global_gym.sim_id) +
        "agent: {:3.0f}, ".format(agent.id) +
        "episode: {:5.0f}, ".format(episode) +
        "score: {:4.0f}, ".format(score) +
        "avg_score: {:8.2f}, ".format(avg_score) +
        "timesteps: {:4.0f}, ".format(timesteps) +
        "tot_timesteps: {:8.0f}, ".format(tot_timesteps) +
        "training_it: {:8.0f}, ".format(agent.brain.training_it) +
        "global_t: {}, ".format(Global.get_time()) +
        "world_t: {}".format(Util.getTimeString2())
    )

    msg_csv = (
        agent.id,
        episode,
        score,
        avg_score,
        timesteps,
        tot_timesteps
    )

    # Record data
    if global_gym.record and record:
        global_gym.simlogs_writer.writerow(msg_csv)

    # Print in debug mode
    if global_gym.debug and debug:
        print(msg_event)


def xprint(color=PRINT_BLUE, msg=""):
    printColor(color=color, msg="sim_id: {:3.0f}, ".format(global_gym.sim_id) +
                                "{: <35s}, ".format(msg) +
                                "sim_tmstp: {:8.0f}, ".format(Global.sim_timesteps) +
                                "global_t: {}, ".format(Global.get_time()) +
                                "world_t: {}".format(Util.getTimeString2()))

def create_record_file(dir, suffix=""):
    """
        Create record CSV file and return it
        Also create the directory if it doesn't exist
    """
    time_str = Util.getTimeString()
    filename = dir + time_str + '_' + suffix + ".csv"

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    return filename
