
import sys

try:
    # Running in PyCharm
    import task_homing.global_homing as global_homing
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
    import task_homing.global_homing as global_homing
    from ..res.print_colors import *
    from .. import Util
    from .. import Global

header = ("agent",
          "event",
          "timestep",
          "goal_reached",
          "timestep_to_goal",
          "collisions_to_goal",
          "collisions",
          "agent_collisions_to_goal",
          "agent_collisions",
          "learning_score"
          )

dico_event = {}

def print_event(color="", agent=None, event_message=""):
    """
        Agent           : agent ID
        Event           : event's message
        tmstp           : timestep passed (since beginning of simulation)
        GR              : goal reached count
        tmstp2G         : timestep to goal
        Col2G           : collision count between 1 goal to another
        Col             : total collision count
        AgentCol2G      : agent collision count between 1 goal to another
        AgentCol        : total agent collision count
        LS              : learning score of the agent (average of rewards in sliding window)
        t       : time passed (since beginning of simulation)
    """
    # global_homing.event_count += 1  # increment the event counter

    # Don't print in non-debug mode
    if not global_homing.debug:
        return

    msg = ("SimID: {:3.0f}, ".format(global_homing.simulation_id) +
           "Agent: {:3.0f}, ".format(agent.id) +
           "{:>25s}".format(event_message) +
           ", tmstp: {:10.0f}, ".format(Global.sim_timesteps) +
           "training_it: {:10.0f}, ".format(agent.training_it()) +
           "GR: {:5.0f}, ".format(agent.goalReachedCount) +
           "tmstp2G : {:8.0f}, ".format(agent.elapsedTimestep) +
           "Col2G: {:3.0f}, Col: {:5.0f}, ".format(agent.t2GCollisionCount, agent.collisionCount) +
           "AgentCol2G: {:3.0f}, AgentCol: {:5.0f}, ".format(agent.t2GAgentCollisionCount, agent.agentCollisionCount) +
           # "LS: {:3.4f}, "
           # "event_count: {:5.0f}, "
           "t: {}".format(Global.get_time())
           )

    msg_csv = (agent.id,
               event_message,
               Global.sim_timesteps,
               agent.goalReachedCount,
               agent.elapsedTimestep,
               agent.t2GCollisionCount,
               agent.collisionCount,
               agent.t2GAgentCollisionCount,
               agent.agentCollisionCount,
               # agent.learning_score()
               )

    # Record data
    if global_homing.record:

        # Write header only once at the beginning
        if not global_homing.header_write:
            if len(header) != len(msg_csv):
                sys.exit("Header doesn't match csv data")
            global_homing.simlogs_writer.writerow(header)
            global_homing.header_write = True

        global_homing.simlogs_writer.writerow(msg_csv)

    # Increment counter in event's dictionary
    if event_message in dico_event:
        dico_event[event_message] += 1
    else: # First time event encounter
        dico_event[event_message] = 1

    # Don't print in non-debug mode
    if global_homing.debug:
        sys.stdout.write(color)
        print(msg)
        sys.stdout.write(PRINT_RESET)

    return


def xprint(color=PRINT_BLUE, msg=""):
    print_color(color=color, msg="{: <37s}".format(msg) +
                                ", tmstp: {:10.0f}, t: {}".format(Global.sim_timesteps, Global.get_time()) +
                                ", world_t: {}".format(Util.get_time_string2()))
