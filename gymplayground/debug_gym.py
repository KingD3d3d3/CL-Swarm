"""
    Print log, record simulation event
"""
import os
import errno
import gymplayground.global_gym as global_gym
from res.print_colors import *
import res.Util as Util
import Global

header = (
    'environment',
    'agent',
    'episode',
    'score',
    'avg_score',
    'timesteps',
    'tot_timesteps',
    'sim_t'
)

def print_event(env, agent, episode, score, avg_score_100ep, avg_score_10ep, timesteps, tot_timesteps, record=False, debug=True):
    """

        :param env: environment (also called problem)
        :param agent: agent calling the function
        :param episode: current episode
        :param score: score of the episode
        :param avg_score_100ep: average score over the last 100 episodes
        :param avg_score_10ep: average score over the last 10 episodes
        :param timesteps: number of timesteps passed for the episode
        :param tot_timesteps: total number of timesteps passed since beginning of simulation
        :param record: if yes record this log
        :param debug: if yes print this log
    """
    msg_debug = (
        "env: {:<15s}, ".format(env) +
        "sim_id: {:3.0f}, ".format(global_gym.sim_id) +
        "agent: {:3.0f}, ".format(agent.id) +
        "episode: {:5.0f}, ".format(episode) +
        "score: {:4.0f}, ".format(score) +
        "avg_score: {:8.2f}, ".format(avg_score_100ep) +
        "score_10ep: {:8.2f}, ".format(avg_score_10ep) +
        "timesteps: {:4.0f}, ".format(timesteps) +
        "tot_timesteps: {:8.0f}, ".format(tot_timesteps) +
        "training_it: {:8.0f}, ".format(agent.brain.training_it) +
        "sim_t: {}, ".format(Global.get_sim_time()) +
        "global_t: {}, ".format(Global.get_time()) +
        "world_t: {}".format(Util.get_time_string2())
    )

    sim_t = Global.get_sim_time_in_seconds()
    msg_csv = (
        env,
        agent.id,
        episode,
        score,
        avg_score_100ep,
        timesteps,
        tot_timesteps,
        sim_t
    )

    # Record data
    if global_gym.record and record:
        global_gym.simlogs_writer.writerow(msg_csv)

    # Print in debug mode
    if global_gym.debug and debug:
        print(msg_debug)

def xprint(color=PRINT_BLUE, msg=""):
    """
        Custom print with different color and print simulation info
        :param color: color to print with
        :param msg: message to print
    """
    print_color(color=color, msg="sim_id: {:3.0f}, ".format(global_gym.sim_id) +
                                 "{: <35s}, ".format(msg) +
                                 # "sim_tmstp: {:8.0f}, ".format(Global.sim_timesteps) +
                                 "sim_t: {}, ".format(Global.get_sim_time()) +
                                 "global_t: {}, ".format(Global.get_time()) +
                                 "world_t: {}".format(Util.get_time_string2()))


def create_record_file(dir, suffix=""):
    """
        Create record CSV file and return it
        Also create the directory if it doesn't exist
        :param dir: directory to save the file
        :param suffix: suffix to add
        :return: the record file (simulation logs file)
    """
    time_str = Util.get_time_string()
    if suffix:
        filename = dir + time_str + '_' + suffix + "_rec.csv"
    else:
        filename = dir + time_str + "_rec.csv"

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    return filename
