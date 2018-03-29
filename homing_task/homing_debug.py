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


def xprint(agent=None, message="", record=False):
    """
        t2G     : time to goal
        tmstp2G : timestep to goal
        GR      : goal reached count
        t       : time passed (since beginning of simulation)
        tmstp   : timestep passed ( // )
        Col2G   : collision count between 1 goal to another
        Col     : total collision count
        AgentCol2G   : agent collision count between 1 goal to another
        AgentCol     : total agent collision count
    """
    total_msg = ("Agent: {:3.0f}, ".format(agent.id) +
                 "{:>30s}".format(message) +
                 ", t2G: {:10.3f}, tmstp2G : {:10.0f}, "
                 "GR: {:5.0f}, t: {:10.3f}, tmstp: {:10.0f}, "
                 "Col2G: {:5.0f}, Col: {:5.0f}, "
                 "AgentCol2G: {:5.0f}, AgentCol: {:5.0f}"
                 .format(agent.elapsedTime, agent.elapsedTimestep,
                         agent.goalReachedCount, homing_global.timer, homing_global.timestep,
                         agent.t2GCollisionCount, agent.collisionCount,
                         agent.t2GAgentCollisionCount, agent.agentCollisionCount)
                 )

    if homing_global.record:
        homing_global.fo.write(total_msg + '\n')

    # Don't print in non-debug mode
    if homing_global.debug:
        print(total_msg)

    return

# def xprint(agent, *args):
#     # Don't print in non-debug mode
#     if not homing_global.debug:
#         return
#
#     print("Agent: {:3.0f}, ".format(agent.id) +
#           ",".join(map(str, args)) +
#           " time to goal: {:10.3f}, goal reached count: {:3.0f}, simulation time: {:10.3f}, timestep : {:10.0f}"
#           .format(agent.elapsedTime / 1000.0, agent.goalReachedCount, homing_global.timer, homing_global.timestep))
#     pass
