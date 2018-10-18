
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

header = ("agent",
          "event",
          "timestep",
          "goal_reached",
          "timestep_to_goal",
          "learning_score"
          )


def printEvent(color="", agent=None, event_message=""):
    """
        Agent           : agent ID
        Event           : event's message
        tmstp           : timestep passed (since beginning of simulation)
        GR              : goal reached count
        tmstp2G         : timestep to goal
        LS              : learning score of the agent (average of rewards in sliding window)
        t       : time passed (since beginning of simulation)
    """
    global_homing_simple.event_count += 1  # increment the event counter

    # Don't print in non-debug mode
    if not global_homing_simple.debug:
        return

    msg = ("sim_id: {:3.0f}, ".format(global_homing_simple.simulation_id) +
           "agent: {:3.0f}, ".format(agent.id) +
           "{:>25s}".format(event_message) +  # 28
           ", tmstp: {:10.0f}, "
           "training_it: {:10.0f}, "
           "GR: {:5.0f}, "
           "tmstp2G : {:8.0f}, "
           "LS: {:3.4f}, "
           "event_count: {:5.0f}, "
           "t: {}"
           .format(
               Global.timestep,
               agent.training_it(),
               agent.goalReachedCount,
               agent.elapsedTimestep,
               agent.learning_score(),
               global_homing_simple.event_count,
               Global.get_time()
           )
           )

    msg_csv = (agent.id,
               event_message,
               Global.timestep,
               agent.goalReachedCount,
               agent.elapsedTimestep,
               agent.learning_score()
               )

    # Record data
    if global_homing_simple.record:

        # Write header only once at the beginning
        if not global_homing_simple.header_write:
            if len(header) != len(msg_csv):
                sys.exit("Header doesn't match csv data")
            global_homing_simple.simlogs_writer.writerow(header)
            global_homing_simple.header_write = True

        global_homing_simple.simlogs_writer.writerow(msg_csv)

    # Print in debug mode
    if global_homing_simple.debug:
        sys.stdout.write(color)
        print(msg)
        sys.stdout.write(PRINT_RESET)

    return


def xprint(color=PRINT_BLUE, msg=""):
    printColor(color=color, msg="sim_id: {:3.0f}, ".format(global_gym.sim_id) +
                                "{: <37s}, ".format(msg) +
                                "tmstp: {:10.0f}, t: {}, ".format(Global.timestep, Global.get_time()) +
                                "world_t: {}".format(Util.getTimeString2()))
