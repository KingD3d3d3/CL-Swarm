
from Box2D.b2 import (world, polygonShape, vec2, contactListener)
import pygame, sys

try:
    # Running in PyCharm
    from task_race.AgentRace import AgentRace
except NameError as err:
    print(err, "--> our error message")
    # Running in command line
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Running from command line -> Import libraries as package")
    from task_race.AgentRace import AgentRace


# Event when there is a collision
class RaceContactListener(contactListener):

    def __init__(self):
        contactListener.__init__(self)

    def BeginContact(self, contact):

        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body

        objectA = bodyA.userData
        objectB = bodyB.userData

        if isinstance(objectA, AgentRace):
            agent = objectA
            agent.done = True
            agent.collision = True
            return

        if isinstance(objectB, AgentRace):
            agent = objectB
            agent.done = True
            agent.collision = True
            return

    def EndContact(self, contact):
        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body

    def PreSolve(self, contact, oldManifold):
        pass

    def PostSolve(self, contact, impulse):
        pass
