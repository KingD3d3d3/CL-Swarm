from __future__ import division
from Box2D.b2 import (world, polygonShape, vec2, contactListener)
import pygame, sys

try:
    # Running in PyCharm
    import res.colors as Color
    from Circle import StaticCircle
    from AgentHomingSimple import AgentHomingSimple
    from Border import Wall
    import debug_homing_simple
    import global_homing_simple
except:
    # Running in command line
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running from command line -> Import libraries as package')
    from ..res import colors as Color
    from ..Circle import StaticCircle
    from .AgentHomingSimple import AgentHomingSimple
    from ..Border import Wall
    import debug_homing_simple
    import global_homing_simple

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

        if isinstance(objectA, AgentHomingSimple) and isinstance(objectB, StaticCircle):
            agent = objectA
            obstacle = objectB

            if cls.isTimestepAgentObstacleCollisionShort(agent=agent, obstacle=obstacle):
                return  # Same agent to obstacle collision in a small time

            agent.t2GCollisionCount += 1
            agent.collisionCount += 1

            agent.collisionColor()


            # Log event only when collision avoidance behavior
            if agent.collision_avoidance:
                debug_homing_simple.printEvent(agent, "collision {}: {}".format("StaticCircle", obstacle.id))

            return

        if isinstance(objectA, StaticCircle) and isinstance(objectB, AgentHomingSimple):
            agent = objectB
            obstacle = objectA

            if cls.isTimestepAgentObstacleCollisionShort(agent=agent, obstacle=obstacle):
                return  # Same agent to obstacle collision in a small time

            agent.t2GCollisionCount += 1
            agent.collisionCount += 1

            agent.collisionColor()

            # Log event only when collision avoidance behavior
            if agent.collision_avoidance:
                debug_homing_simple.printEvent(agent, "collision {}: {}".format("StaticCircle", obstacle.id))

            return

    @classmethod
    def isTimestepAgentObstacleCollisionShort(cls, agent, obstacle):
        """
            Check if time between successive collision between the same agent and same obstacles was too short
        """
        if agent.lastObstacleCollide == obstacle:  # same objects colliding
            agent.elapsedTimestepObstacleCollision = global_homing_simple.timestep - agent.startTimestepObstacleCollision

            if agent.elapsedTimestepObstacleCollision <= EPSILON_TIMESTEP:  # Check elapsed timestep
                agent.startTimestepObstacleCollision = global_homing_simple.timestep  # update based on elapsed time

                # Keep colors
                agent.collisionColor()

                return True  # was too short for collision, so return True

        return False

    @classmethod
    def checkAgentObstacleEndCollision(cls, objectA, objectB):
        """
            Compute if agent to wall end of collision
        """
        if isinstance(objectA, AgentHomingSimple) and isinstance(objectB, StaticCircle):
            agent = objectA
            obstacle = objectB
            agent.lastObstacleCollide = obstacle

            agent.endCollisionColor()
            agent.startTimestepObstacleCollision = global_homing_simple.timestep  # Start counting
            return

        if isinstance(objectA, StaticCircle) and isinstance(objectB, AgentHomingSimple):
            agent = objectB
            obstacle = objectA
            agent.lastObstacleCollide = obstacle

            agent.endCollisionColor()
            agent.startTimestepObstacleCollision = global_homing_simple.timestep  # Start counting
            return

    # ------------------------------------------------------------------------------------------------------------------


    # ---------------------- Agent to Wall -----------------------------------------------------------------------------
    @classmethod
    def checkAgentWallCollision(cls, objectA, objectB):
        """
            Compute if agent to wall collision
        """
        if isinstance(objectA, AgentHomingSimple) and isinstance(objectB, Wall):
            agent = objectA
            obstacle = objectB
            agent.lastObstacleCollide = obstacle
            agent.t2GCollisionCount += 1
            agent.collisionCount += 1

            agent.collisionColor()

            # Log event only when collision avoidance behavior
            if agent.collision_avoidance:
                debug_homing_simple.printEvent(agent, "collision {}: {}".format("Wall", obstacle.id))

            return

        if isinstance(objectA, Wall) and isinstance(objectB, AgentHomingSimple):
            agent = objectB
            obstacle = objectA
            agent.lastObstacleCollide = obstacle
            agent.t2GCollisionCount += 1
            agent.collisionCount += 1

            agent.collisionColor()

            # Log event only when collision avoidance behavior
            if agent.collision_avoidance:
                debug_homing_simple.printEvent(agent, "collision {}: {}".format("Wall", obstacle.id))

            return

    @classmethod
    def checkAgentWallEndCollision(cls, objectA, objectB):
        """
            Compute if agent to wall end of collision
        """
        if isinstance(objectA, AgentHomingSimple) and isinstance(objectB, Wall):
            agent = objectA

            agent.endCollisionColor()
            return

        if isinstance(objectA, Wall) and isinstance(objectB, AgentHomingSimple):
            agent = objectB

            agent.endCollisionColor()
            return
    # ------------------------------------------------------------------------------------------------------------------


    # ---------------------- Agent to Agent ----------------------------------------------------------------------------
    @classmethod
    def checkAgentAgentCollision(cls, objectA, objectB):
        """
            Compute if agent to agent collision
        """
        if isinstance(objectA, AgentHomingSimple) and isinstance(objectB, AgentHomingSimple):
            agentA = objectA
            agentB = objectB

            if cls.isTimestepAgentAgentCollisionShort(agentA=agentA, agentB=agentB):
                return  # Same agent to obstacle collision in a small time

            # Agent A
            agentA.t2GAgentCollisionCount += 1
            agentA.agentCollisionCount += 1
            agentA.collisionColor()

            # Agent B
            agentB.t2GAgentCollisionCount += 1
            agentB.agentCollisionCount += 1
            agentB.collisionColor()

            # Log event only when collision avoidance behavior
            if agentA.collision_avoidance:
                debug_homing_simple.printEvent(agentA, "collision {}: {}".format("Agent", agentB.id))

            if agentB.collision_avoidance:
                debug_homing_simple.printEvent(agentB, "collision {}: {}".format("Agent", agentA.id))

            return

    @classmethod
    def isTimestepAgentAgentCollisionShort(cls, agentA, agentB):
        """
            Check if time between successive collision between the same agent and same obstacles was too short
        """
        idA = agentA.id
        idB = agentB.id

        agentA.elapsedTimestepAgentCollision[idB] = global_homing_simple.timestep - agentA.startTimestepAgentCollision[idB]
        agentB.elapsedTimestepAgentCollision[idA] = global_homing_simple.timestep - agentB.startTimestepAgentCollision[idA]

        if agentA.elapsedTimestepAgentCollision[idB] <= EPSILON_TIMESTEP \
                and agentB.elapsedTimestepAgentCollision[idA] <= EPSILON_TIMESTEP:  # Check elapsed timestep

            agentA.startTimestepAgentCollision[idB] = global_homing_simple.timestep  # update based on elapsed time
            agentB.startTimestepAgentCollision[idA] = global_homing_simple.timestep  # update based on elapsed time

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
        if isinstance(objectA, AgentHomingSimple) and isinstance(objectB, AgentHomingSimple):
            agentA = objectA
            agentB = objectB

            idA = agentA.id
            idB = agentB.id

            agentA.endCollisionColor()
            agentB.endCollisionColor()
            # debug_homing_simple.printEvent(agent, "end collision {}: {}".format("Wall", obstacle.id))

            agentA.startTimestepAgentCollision[idB] = global_homing_simple.timestep  # Start counting
            agentB.startTimestepAgentCollision[idA] = global_homing_simple.timestep  # Start counting
            return
    # ------------------------------------------------------------------------------------------------------------------
