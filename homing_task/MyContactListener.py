from Box2D.b2 import (world, polygonShape, vec2, contactListener)
import pygame, sys

try:
    # Running in PyCharm
    import res.colors as Color
    from Circle import StaticCircle
    from AgentHoming import AgentHoming
    from Border import Wall
    import homing_global
    import homing_debug
except:
    # Running in command line
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running from command line -> Import libraries as package')
    from ..res import colors as Color
    from ..Circle import StaticCircle
    from .AgentHoming import AgentHoming
    from ..Border import Wall
    import homing_global
    import homing_debug


# Event when there is a collision
class MyContactListener(contactListener):
    def __init__(self):
        contactListener.__init__(self)

    def BeginContact(self, contact):
        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body

        objectA = bodyA.userData
        objectB = bodyB.userData

        # Agent to Obstacles collision
        if isinstance(objectA, AgentHoming) and isinstance(objectB, StaticCircle):
            agent = objectA
            obstacle = objectB

            # Change colors
            agent.color = Color.DeepSkyBlue
            agent.raycastSideColor = Color.Magenta
            agent.raycastFrontColor = Color.Magenta

            homing_debug.xprint(agent, "collision {}: {}".format("StaticCircle", obstacle.id))

        if isinstance(objectA, StaticCircle) and isinstance(objectB, AgentHoming):
            agent = objectB
            obstacle = objectA

            # Change colors
            agent.color = Color.DeepSkyBlue
            agent.raycastSideColor = Color.Magenta
            agent.raycastFrontColor = Color.Magenta

            homing_debug.xprint(agent, "collision {}: {}".format("StaticCircle", obstacle.id))

        # Agent to Wall collision
        if isinstance(objectA, AgentHoming) and isinstance(objectB, Wall):
            agent = objectA
            obstacle = objectB

            # Change colors
            agent.color = Color.DeepSkyBlue
            agent.raycastSideColor = Color.Magenta
            agent.raycastFrontColor = Color.Magenta

            homing_debug.xprint(agent, "collision {}: {}".format("Wall", obstacle.id))

        if isinstance(objectA, Wall) and isinstance(objectB, AgentHoming):
            agent = objectB
            obstacle = objectA

            # Change colors
            agent.color = Color.DeepSkyBlue
            agent.raycastSideColor = Color.Magenta
            agent.raycastFrontColor = Color.Magenta

            homing_debug.xprint(agent, "collision {}: {}".format("Wall", obstacle.id))

    def EndContact(self, contact):
        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body

        objectA = bodyA.userData
        objectB = bodyB.userData

        nameA = objectA.__class__.__name__
        nameB = objectB.__class__.__name__

        # Agent to Obstacles collision
        if isinstance(objectA, AgentHoming) and isinstance(objectB, StaticCircle):
            agent = objectA
            agent.color = agent.initial_color
            agent.raycastFrontColor = agent.initial_raycastFrontColor
            agent.raycastSideColor = agent.initial_raycastSideColor
        if isinstance(objectA, StaticCircle) and isinstance(objectB, AgentHoming):
            agent = objectB
            agent.color = agent.initial_color
            agent.raycastFrontColor = agent.initial_raycastFrontColor
            agent.raycastSideColor = agent.initial_raycastSideColor

        # Agent to Wall collision
        if isinstance(objectA, AgentHoming) and isinstance(objectB, Wall):
            agent = objectA
            agent.color = agent.initial_color
            agent.raycastFrontColor = agent.initial_raycastFrontColor
            agent.raycastSideColor = agent.initial_raycastSideColor
        if isinstance(objectA, Wall) and isinstance(objectB, AgentHoming):
            agent = objectB
            agent.color = agent.initial_color
            agent.raycastFrontColor = agent.initial_raycastFrontColor
            agent.raycastSideColor = agent.initial_raycastSideColor


    def PreSolve(self, contact, oldManifold):
        pass

    def PostSolve(self, contact, impulse):
        pass
