import sys

try:
    # Running in PyCharm
    import homing_global
except:
    # Running in command line
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running from command line -> Import libraries as package')
    import homing_global

header_write = False
header = ("Agent",
          "Event",
          "Timestep",
          "Goal reached",
          "Timestep to goal",
          "Collisions to goal",
          "Collisions",
          "Agent collisions to goal",
          "Agent collisions",
          "Learning score"
          )


def xprint(agent=None, event_message=""):
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
        Learning score  : learning score of the agent
    """
    global header_write

    msg = ("Agent: {:3.0f}, ".format(agent.id) +
           "{:>30s}".format(event_message) +
           ", tmstp: {:10.0f}, "
           "GR: {:5.0f}, "
           "tmstp2G : {:10.0f}, "
           "Col2G: {:5.0f}, Col: {:5.0f}, "
           "AgentCol2G: {:5.0f}, AgentCol: {:5.0f}, "
           "Learning Score: {:3.2f}"
           .format(
               homing_global.timestep,
               agent.goalReachedCount,
               agent.elapsedTimestep,
               agent.t2GCollisionCount, agent.collisionCount,
               agent.t2GAgentCollisionCount, agent.agentCollisionCount,
               agent.score()
           )
           )

    msg_csv = (agent.id,
               event_message,
               homing_global.timestep,
               agent.goalReachedCount,
               agent.elapsedTimestep,
               agent.t2GCollisionCount,
               agent.collisionCount,
               agent.t2GAgentCollisionCount,
               agent.agentCollisionCount,
               agent.score()
               )

    # Record data
    if homing_global.record:

        # Write header only once at the beginning
        if not header_write:
            if len(header) != len(msg_csv):
                sys.exit("Header doesn't match csv data")
            homing_global.writer.writerow(header)
            header_write = True

        homing_global.writer.writerow(msg_csv)

    # Don't print in non-debug mode
    if homing_global.debug:
        print(msg)

    return
