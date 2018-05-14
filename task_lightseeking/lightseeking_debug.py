import sys

try:
    # Running in PyCharm
    import lightseeking_global
    from res.print_colors import printColor
except:
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running from command line -> Import libraries as package')
    import lightseeking_global
    from ..res.print_colors import printColor

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

    msg = ("Agent: {:3.0f}, ".format(agent.id) +
           "{:>28s}".format(event_message) +
           ", tmstp: {:10.0f}, "
           "GR: {:5.0f}, "
           "tmstp2G : {:10.0f}, "
           "Col2G: {:5.0f}, Col: {:5.0f}, "
           "AgentCol2G: {:5.0f}, AgentCol: {:5.0f}, "
           "LS: {:3.2f}, "
           "t: {:10.3f}"
           .format(
               lightseeking_global.timestep,
               agent.goalReachedCount,
               agent.elapsedTimestep,
               agent.t2GCollisionCount, agent.collisionCount,
               agent.t2GAgentCollisionCount, agent.agentCollisionCount,
               agent.learning_score(),
               lightseeking_global.timer
           )
           )

    msg_csv = (agent.id,
               event_message,
               lightseeking_global.timestep,
               agent.goalReachedCount,
               agent.elapsedTimestep,
               agent.t2GCollisionCount,
               agent.collisionCount,
               agent.t2GAgentCollisionCount,
               agent.agentCollisionCount,
               agent.learning_score()
               )

    # Record data
    if lightseeking_global.record:

        # Write header only once at the beginning
        if not header_write:
            if len(header) != len(msg_csv):
                sys.exit("Header doesn't match csv data")
            lightseeking_global.writer.writerow(header)
            header_write = True

        lightseeking_global.writer.writerow(msg_csv)

    # Don't print in non-debug mode
    if lightseeking_global.debug:
        print(msg)

    return

def xprint(msg=""):
    printColor(msg="{: <40s}".format(msg) +
                   ", tmstp: {:10.0f}, t: {:10.3f}".format(lightseeking_global.timestep, lightseeking_global.timer))