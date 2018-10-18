
from Box2D.b2 import (world, polygonShape, vec2, contactListener)
import pygame, sys

try:
    # Running in PyCharm
    import res.colors as Color
    from objects.Circle import StaticCircle
    from AgentHoming import AgentHoming
    from objects.Border import Wall
    import debug_homing
    import global_homing
    import Global
except NameError as err:
    print(err, "--> our error message")
    # Running in command line
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running from command line -> Import libraries as package")
    from ..res import colors as Color
    from ..objects.Circle import StaticCircle
    from .AgentHoming import AgentHoming
    from ..objects.Border import Wall
    import debug_homing
    import global_homing
    from .. import Global

EPSILON_TIMESTEP = 10  # 30


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
        self.checkAgentAgentCollision(objectA, objectB)

    def EndContact(self, contact):
        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body

        objectA = bodyA.userData
        objectB = bodyB.userData

        self.checkAgentObstacleEndCollision(objectA, objectB)
        self.checkAgentWallEndCollision(objectA, objectB)
        self.checkAgentAgentEndCollision(objectA, objectB)

    def PreSolve(self, contact, oldManifold):
        pass

    def PostSolve(self, contact, impulse):
        pass


    # ---------------------- Agent to obstacles ------------------------------------------------------------------------
    @classmethod
    def checkAgentObstacleCollision(cls, objectA, objectB):
        """
            Compute if agent to obstacle collision
        """

        if isinstance(objectA, AgentHoming) and isinstance(objectB, StaticCircle):
            agent = objectA
            obstacle = objectB

            # Same agent to obstacle collision in a small time
            if cls.isTimestepAgentObstacleCollisionShort(agent=agent, obstacle=obstacle):
                return

            agent.lastObstacleCollide = obstacle
            agent.t2GCollisionCount += 1
            agent.collisionCount += 1
            agent.collisionColor()
            debug_homing.printEvent(agent=agent, event_message="collision {}: {}".format("StaticCircle", obstacle.id))

            return

        if isinstance(objectA, StaticCircle) and isinstance(objectB, AgentHoming):
            agent = objectB
            obstacle = objectA

            # Same agent to obstacle collision in a small time
            if cls.isTimestepAgentObstacleCollisionShort(agent=agent, obstacle=obstacle):
                return

            agent.lastObstacleCollide = obstacle
            agent.t2GCollisionCount += 1
            agent.collisionCount += 1
            agent.collisionColor()
            debug_homing.printEvent(agent=agent, event_message="collision {}: {}".format("StaticCircle", obstacle.id))

            return

    @classmethod
    def isTimestepAgentObstacleCollisionShort(cls, agent, obstacle):
        """
            Check if time between successive collision between the same agent and same obstacles was too short
        """
        if agent.lastObstacleCollide == obstacle:  # same objects colliding
            agent.elapsedTimestepObstacleCollision = Global.timestep - agent.startTimestepObstacleCollision

            if agent.elapsedTimestepObstacleCollision <= EPSILON_TIMESTEP:  # Check elapsed timestep
                agent.startTimestepObstacleCollision = Global.timestep  # update based on elapsed time

                # Keep colors
                agent.collisionColor()

                return True  # was too short for collision, so return True

        return False

    @classmethod
    def checkAgentObstacleEndCollision(cls, objectA, objectB):
        """
            Compute if agent to wall end of collision
        """
        if isinstance(objectA, AgentHoming) and isinstance(objectB, StaticCircle):
            agent = objectA
            obstacle = objectB
            agent.lastObstacleCollide = obstacle

            agent.endCollisionColor()
            agent.startTimestepObstacleCollision = Global.timestep  # Start counting
            return

        if isinstance(objectA, StaticCircle) and isinstance(objectB, AgentHoming):
            agent = objectB
            obstacle = objectA
            agent.lastObstacleCollide = obstacle

            agent.endCollisionColor()
            agent.startTimestepObstacleCollision = Global.timestep  # Start counting
            return

    # ------------------------------------------------------------------------------------------------------------------


    # ---------------------- Agent to Wall -----------------------------------------------------------------------------
    @classmethod
    def checkAgentWallCollision(cls, objectA, objectB):
        """
            Compute if agent to wall collision
        """
        if isinstance(objectA, AgentHoming) and isinstance(objectB, Wall):
            agent = objectA
            wall = objectB

            # Same agent to wall collision in a small time
            if cls.isTimestepAgentObstacleCollisionShort(agent=agent, obstacle=wall):
                return

            agent.lastObstacleCollide = wall
            agent.t2GCollisionCount += 1
            agent.collisionCount += 1

            agent.collisionColor()

            debug_homing.printEvent(agent=agent, event_message="collision {}: {}".format("Wall", wall.id))

            return

        if isinstance(objectA, Wall) and isinstance(objectB, AgentHoming):
            agent = objectB
            wall = objectA

            # Same agent to wall collision in a small time
            if cls.isTimestepAgentObstacleCollisionShort(agent=agent, obstacle=wall):
                return

            agent.lastObstacleCollide = wall
            agent.t2GCollisionCount += 1
            agent.collisionCount += 1

            agent.collisionColor()

            debug_homing.printEvent(agent=agent, event_message="collision {}: {}".format("Wall", wall.id))

            return

    @classmethod
    def checkAgentWallEndCollision(cls, objectA, objectB):
        """
            Compute if agent to wall end of collision
        """
        if isinstance(objectA, AgentHoming) and isinstance(objectB, Wall):
            agent = objectA
            wall = objectB
            agent.lastObstacleCollide = wall

            agent.endCollisionColor()
            return

        if isinstance(objectA, Wall) and isinstance(objectB, AgentHoming):
            agent = objectB
            wall = objectA
            agent.lastObstacleCollide = wall

            agent.endCollisionColor()
            return
    # ------------------------------------------------------------------------------------------------------------------


    # ---------------------- Agent to Agent ----------------------------------------------------------------------------
    @classmethod
    def checkAgentAgentCollision(cls, objectA, objectB):
        """
            Compute if agent to agent collision
        """
        if isinstance(objectA, AgentHoming) and isinstance(objectB, AgentHoming):
            agentA = objectA
            agentB = objectB

            # Same agent to agent collision in a small time
            if cls.isTimestepAgentAgentCollisionShort(agentA=agentA, agentB=agentB):
                return

            # Agent A
            agentA.t2GAgentCollisionCount += 1
            agentA.agentCollisionCount += 1
            agentA.collisionColor()
            agentA.lastObstacleCollide = agentB

            # Agent B
            agentB.t2GAgentCollisionCount += 1
            agentB.agentCollisionCount += 1
            agentB.collisionColor()
            agentB.lastObstacleCollide = agentA

            debug_homing.printEvent(agent=agentA, event_message="collision {}: {}".format("Agent", agentB.id))
            debug_homing.printEvent(agent=agentB, event_message="collision {}: {}".format("Agent", agentA.id))

            return

    @classmethod
    def isTimestepAgentAgentCollisionShort(cls, agentA, agentB):
        """
            Check if time between successive collision between the same agent and same obstacles was too short
        """
        idA = agentA.id
        idB = agentB.id

        agentA.elapsedTimestepAgentCollision[idB] = Global.timestep - agentA.startTimestepAgentCollision[idB]
        agentB.elapsedTimestepAgentCollision[idA] = Global.timestep - agentB.startTimestepAgentCollision[idA]

        if agentA.elapsedTimestepAgentCollision[idB] <= EPSILON_TIMESTEP \
                and agentB.elapsedTimestepAgentCollision[idA] <= EPSILON_TIMESTEP:  # Check elapsed timestep

            agentA.startTimestepAgentCollision[idB] = Global.timestep  # update based on elapsed time
            agentB.startTimestepAgentCollision[idA] = Global.timestep  # update based on elapsed time

            # Keep colors
            agentA.collisionColor()
            agentB.collisionColor()

            return True  # was too short for collision, so return True

        return False

    @classmethod
    def checkAgentAgentEndCollision(cls, objectA, objectB):
        """
            Compute if agent to agent end of collision
        """
        if isinstance(objectA, AgentHoming) and isinstance(objectB, AgentHoming):
            agentA = objectA
            agentB = objectB

            idA = agentA.id
            idB = agentB.id

            agentA.endCollisionColor()
            agentB.endCollisionColor()

            agentA.startTimestepAgentCollision[idB] = Global.timestep  # Start counting
            agentB.startTimestepAgentCollision[idA] = Global.timestep  # Start counting
            return
    # ------------------------------------------------------------------------------------------------------------------
