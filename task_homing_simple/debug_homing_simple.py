import sys

try:
    # Running in PyCharm
    import global_homing_simple
    from res.print_colors import printColor
    import Util
except:
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running from command line -> Import libraries as package')
    import global_homing_simple
    from ..res.print_colors import printColor
    from .. import Util

header_write = False
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


def printEvent(agent=None, event_message=""):
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
    global header_write

    global_homing_simple.event_count += 1 # increment the event counter

    msg = ("Agent: {:3.0f}, ".format(agent.id) +
           "{:>25s}".format(event_message) + # 28
           ", tmstp: {:10.0f}, "
           "GR: {:5.0f}, "
           "tmstp2G : {:8.0f}, "
           "Col2G: {:5.0f}, Col: {:5.0f}, "
           "AgentCol2G: {:5.0f}, AgentCol: {:5.0f}, "
           "LS: {:3.2f}, "
           "event_count: {:5.0f}, "
           "t: {}"
           .format(
               global_homing_simple.timestep,
               agent.goalReachedCount,
               agent.elapsedTimestep,
               agent.t2GCollisionCount, agent.collisionCount,
               agent.t2GAgentCollisionCount, agent.agentCollisionCount,
               agent.learning_score(),
               global_homing_simple.event_count,
               Util.get_time()
           )
           )

    msg_csv = (agent.id,
               event_message,
               global_homing_simple.timestep,
               agent.goalReachedCount,
               agent.elapsedTimestep,
               agent.t2GCollisionCount,
               agent.collisionCount,
               agent.t2GAgentCollisionCount,
               agent.agentCollisionCount,
               agent.learning_score()
               )

    # Record data
    if global_homing_simple.record:

        # Write header only once at the beginning
        if not header_write:
            if len(header) != len(msg_csv):
                sys.exit("Header doesn't match csv data")
            global_homing_simple.writer.writerow(header)
            header_write = True

        global_homing_simple.writer.writerow(msg_csv)

    # Don't print in non-debug mode
    if global_homing_simple.debug:
        print(msg)

    return

def xprint(msg=""):
    printColor(msg="{: <37s}".format(msg) +
                   ", tmstp: {:10.0f}, t: {}".format(global_homing_simple.timestep, Util.get_time()))