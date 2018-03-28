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
        self.objectA = None
        self.objectB = None
        self.timestepCount = 0
        self.epsilonTimestep = 30

        self.startTimestep = 0
        self.elapsedTimestep = 0

    def BeginContact(self, contact):

        self.timestepCount += 1

        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body

        # Check timestep for
        # Agent to Obstacles collision
        if isinstance(self.objectA, AgentHoming) and isinstance(self.objectB, StaticCircle) \
                and isinstance(self.objectA, AgentHoming) and isinstance(self.objectB, StaticCircle):

            agent = self.objectA
            self.elapsedTimestep = homing_global.timestep - self.startTimestep
            if self.elapsedTimestep <= self.epsilonTimestep:  # Check elapsed timestep
                self.startTimestep = homing_global.timestep  # update based on elapsed time

                # Change colors
                agent.color = Color.DeepSkyBlue
                agent.raycastSideColor = Color.Magenta
                agent.raycastFrontColor = Color.Magenta

                return     # was too short for collision, so just return

        if isinstance(self.objectA, StaticCircle) and isinstance(self.objectB, AgentHoming) \
                and isinstance(self.objectA, StaticCircle) and isinstance(self.objectB, AgentHoming):

            agent = self.objectB

            self.elapsedTimestep = homing_global.timestep - self.startTimestep
            if self.elapsedTimestep <= self.epsilonTimestep:  # Check elapsed timestep
                self.startTimestep = homing_global.timestep  # update based on elapsed time

                # Change colors
                agent.color = Color.DeepSkyBlue
                agent.raycastSideColor = Color.Magenta
                agent.raycastFrontColor = Color.Magenta

                return      # was too short for collision, so just return


        self.objectA = bodyA.userData
        self.objectB = bodyB.userData

        # ------------- Agent to Obstacles collision ----------------
        if isinstance(self.objectA, AgentHoming) and isinstance(self.objectB, StaticCircle):
            agent = self.objectA
            obstacle = self.objectB
            agent.elapsedCollisionCount += 1
            agent.collisionCount += 1

            # Change colors
            agent.color = Color.DeepSkyBlue
            agent.raycastSideColor = Color.Magenta
            agent.raycastFrontColor = Color.Magenta

            homing_debug.xprint(agent, "collision {}: {}".format("StaticCircle", obstacle.id))

            self.startTimestep = homing_global.timestep  # Start counting
        if isinstance(self.objectA, StaticCircle) and isinstance(self.objectB, AgentHoming):
            agent = self.objectB
            obstacle = self.objectA
            agent.elapsedCollisionCount += 1
            agent.collisionCount += 1

            # Change colors
            agent.color = Color.DeepSkyBlue
            agent.raycastSideColor = Color.Magenta
            agent.raycastFrontColor = Color.Magenta

            homing_debug.xprint(agent, "collision {}: {}".format("StaticCircle", obstacle.id))

            self.startTimestep = homing_global.timestep  # Start counting
        # ---------------------------------------------------------

        # ------------- Agent to Wall collision ----------------
        if isinstance(self.objectA, AgentHoming) and isinstance(self.objectB, Wall):
            agent = self.objectA
            obstacle = self.objectB
            agent.elapsedCollisionCount += 1
            agent.collisionCount += 1

            # Change colors
            agent.color = Color.DeepSkyBlue
            agent.raycastSideColor = Color.Magenta
            agent.raycastFrontColor = Color.Magenta

            homing_debug.xprint(agent, "collision {}: {}".format("Wall", obstacle.id))

        if isinstance(self.objectA, Wall) and isinstance(self.objectB, AgentHoming):
            agent = self.objectB
            obstacle = self.objectA
            agent.elapsedCollisionCount += 1
            agent.collisionCount += 1

            # Change colors
            agent.color = Color.DeepSkyBlue
            agent.raycastSideColor = Color.Magenta
            agent.raycastFrontColor = Color.Magenta

            homing_debug.xprint(agent, "collision {}: {}".format("Wall", obstacle.id))
        # ---------------------------------------------------------

    def EndContact(self, contact):
        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body

        objectA = bodyA.userData
        objectB = bodyB.userData

        nameA = objectA.__class__.__name__
        nameB = objectB.__class__.__name__

        # ------------- Agent to Obstacles collision -------------
        if isinstance(objectA, AgentHoming) and isinstance(objectB, StaticCircle):
            agent = objectA
            obstacle = objectB

            agent.color = agent.initial_color
            agent.raycastFrontColor = agent.initial_raycastFrontColor
            agent.raycastSideColor = agent.initial_raycastSideColor

            #homing_debug.xprint(agent, "end collision {}: {}".format("StaticCircle", obstacle.id))

        if isinstance(objectA, StaticCircle) and isinstance(objectB, AgentHoming):
            agent = objectB
            obstacle = objectA

            agent.color = agent.initial_color
            agent.raycastFrontColor = agent.initial_raycastFrontColor
            agent.raycastSideColor = agent.initial_raycastSideColor

            #homing_debug.xprint(agent, "end collision {}: {}".format("StaticCircle", obstacle.id))
        # ---------------------------------------------------------

        # ------------- Agent to Wall collision -------------
        if isinstance(objectA, AgentHoming) and isinstance(objectB, Wall):
            agent = objectA
            obstacle = objectB

            agent.color = agent.initial_color
            agent.raycastFrontColor = agent.initial_raycastFrontColor
            agent.raycastSideColor = agent.initial_raycastSideColor

            #homing_debug.xprint(agent, "end collision {}: {}".format("Wall", obstacle.id))

        if isinstance(objectA, Wall) and isinstance(objectB, AgentHoming):
            agent = objectB
            obstacle = objectA

            agent.color = agent.initial_color
            agent.raycastFrontColor = agent.initial_raycastFrontColor
            agent.raycastSideColor = agent.initial_raycastSideColor

            #homing_debug.xprint(agent, "end collision {}: {}".format("Wall", obstacle.id))
        # ---------------------------------------------------------

    def PreSolve(self, contact, oldManifold):
        pass

    def PostSolve(self, contact, impulse):
        pass
