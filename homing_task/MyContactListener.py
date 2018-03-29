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

EPSILON_TIMESTEP = 30


# Event when there is a collision
class MyContactListener(contactListener):

    def __init__(self):
        contactListener.__init__(self)

    def BeginContact(self, contact):

        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body

        objectA = bodyA.userData
        objectB = bodyB.userData

        self.checkAgentObstacleCollision(objectA, objectB)
        self.checkAgentWallCollision(objectA, objectB)

    def EndContact(self, contact):
        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body

        objectA = bodyA.userData
        objectB = bodyB.userData

        nameA = objectA.__class__.__name__
        nameB = objectB.__class__.__name__

        self.checkAgentObstacleEndCollision(objectA, objectB)
        self.checkAgentWallEndCollision(objectA, objectB)

    def PreSolve(self, contact, oldManifold):
        pass

    def PostSolve(self, contact, impulse):
        pass

    @classmethod
    def checkAgentObstacleCollision(cls, objectA, objectB):
        """
            Compute if agent to obstacle collision
        """
        if cls.isTimestepAgentObstacleCollisionShort(objectA, objectB):
            return  # Same agent to obstacle collision in a small time

        if isinstance(objectA, AgentHoming) and isinstance(objectB, StaticCircle):
            agent = objectA
            obstacle = objectB
            agent.lastObjectCollide = obstacle
            agent.t2GCollisionCount += 1
            agent.totalCollisionCount += 1

            agent.collisionColor()

            homing_debug.xprint(agent, "collision {}: {}".format("StaticCircle", obstacle.id))
            agent.startTimestepCollision = homing_global.timestep  # Start counting
            return

        if isinstance(objectA, StaticCircle) and isinstance(objectB, AgentHoming):
            agent = objectB
            obstacle = objectA
            agent.lastObjectCollide = obstacle
            agent.t2GCollisionCount += 1
            agent.totalCollisionCount += 1

            agent.collisionColor()

            homing_debug.xprint(agent, "collision {}: {}".format("StaticCircle", obstacle.id))
            agent.startTimestepCollision = homing_global.timestep  # Start counting
            return

    @classmethod
    def isTimestepAgentObstacleCollisionShort(cls, objectA, objectB):
        """
            Check if time between successive collision between the same agent and same obstacles was too short
        """
        if isinstance(objectA, AgentHoming) and isinstance(objectB, StaticCircle):
            agent = objectA

            if agent.lastObjectCollide == objectB:  # same objects colliding
                agent.elapsedTimestepCollision = homing_global.timestep - agent.startTimestepCollision

                if agent.elapsedTimestepCollision <= EPSILON_TIMESTEP:  # Check elapsed timestep
                    agent.startTimestepCollision = homing_global.timestep  # update based on elapsed time

                    # Keep colors
                    agent.collisionColor()

                    return True  # was too short for collision, so just return

        elif isinstance(objectA, StaticCircle) and isinstance(objectB, AgentHoming):
            agent = objectB

            if agent.lastObjectCollide == objectA:  # same objects colliding
                agent.elapsedTimestepCollision = homing_global.timestep - agent.startTimestepCollision

                if agent.elapsedTimestepCollision <= EPSILON_TIMESTEP:  # Check elapsed timestep
                    agent.startTimestepCollision = homing_global.timestep  # update based on elapsed time

                    # Keep colors
                    agent.collisionColor()

                    return True  # was too short for collision, so just return

        return False

    @classmethod
    def checkAgentWallCollision(cls, objectA, objectB):
        """
            Compute if agent to wall collision
        """
        if isinstance(objectA, AgentHoming) and isinstance(objectB, Wall):
            agent = objectA
            obstacle = objectB
            agent.lastObjectCollide = obstacle
            agent.t2GCollisionCount += 1
            agent.totalCollisionCount += 1

            agent.collisionColor()

            homing_debug.xprint(agent, "collision {}: {}".format("Wall", obstacle.id))
            return

        if isinstance(objectA, Wall) and isinstance(objectB, AgentHoming):
            agent = objectB
            obstacle = objectA
            agent.lastObjectCollide = obstacle
            agent.t2GCollisionCount += 1
            agent.totalCollisionCount += 1

            agent.collisionColor()

            homing_debug.xprint(agent, "collision {}: {}".format("Wall", obstacle.id))
            return


    @classmethod
    def checkAgentAgentCollision(cls, objectA, objectB):
        return

    @classmethod
    def checkAgentObstacleEndCollision(cls, objectA, objectB):
        """
            Compute if agent to wall end of collision
        """
        if isinstance(objectA, AgentHoming) and isinstance(objectB, StaticCircle):
            agent = objectA
            obstacle = objectB

            agent.endCollisionColor()
            #homing_debug.xprint(agent, "end collision {}: {}".format("StaticCircle", obstacle.id))
            return

        if isinstance(objectA, StaticCircle) and isinstance(objectB, AgentHoming):
            agent = objectB
            obstacle = objectA

            agent.endCollisionColor()
            #homing_debug.xprint(agent, "end collision {}: {}".format("StaticCircle", obstacle.id))
            return

    @classmethod
    def checkAgentWallEndCollision(cls, objectA, objectB):
        """
            Compute if agent to wall end of collision
        """
        if isinstance(objectA, AgentHoming) and isinstance(objectB, Wall):
            agent = objectA
            obstacle = objectB

            agent.endCollisionColor()
            # homing_debug.xprint(agent, "end collision {}: {}".format("Wall", obstacle.id))
            return

        if isinstance(objectA, Wall) and isinstance(objectB, AgentHoming):
            agent = objectB
            obstacle = objectA

            agent.endCollisionColor()
            # homing_debug.xprint(agent, "end collision {}: {}".format("Wall", obstacle.id))
            return