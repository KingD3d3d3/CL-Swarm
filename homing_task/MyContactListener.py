from Box2D.b2 import (world, polygonShape, vec2, contactListener)
import res.colors as Color
from Circle import Circle, StaticCircle
from Agent import Agent
import pygame, sys
from Border import Wall

# Event when there is a collision
class MyContactListener(contactListener):
    def __init__(self):
        contactListener.__init__(self)

    def BeginContact(self, contact):
        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body

        objectA = bodyA.userData
        objectB = bodyB.userData

        nameA = objectA.__class__.__name__
        nameB = objectB.__class__.__name__

        # Agent to Obstacles collision
        if isinstance(objectA, Agent) and isinstance(objectB, StaticCircle):
            print(nameA + " " + str(objectA.id)
                  + " collision " + nameB + " " + str(objectB.id)
                  + " at " + str(pygame.time.get_ticks() / 1000.0) + " s")
        if isinstance(objectA, StaticCircle) and isinstance(objectB, Agent):
            print(nameB + " " + str(objectB.id)
                  + " collision " + nameA + " " + str(objectA.id)
                  + " at " + str(pygame.time.get_ticks() / 1000.0) + " s")

        # Agent to Wall collision
        if isinstance(objectA, Agent) and isinstance(objectB, Wall):
            print(nameA + " " + str(objectA.id)
                  + " collision " + nameB + " " + str(objectB.id)
                  + " at " + str(pygame.time.get_ticks() / 1000.0) + " s")
        if isinstance(objectA, Wall) and isinstance(objectB, Agent):
            print(nameB + " " + str(objectB.id)
                  + " collision " + nameA + " " + str(objectA.id)
                  + " at " + str(pygame.time.get_ticks() / 1000.0) + " s")

        pass

    def EndContact(self, contact):
        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body

        objectA = bodyA.userData
        objectB = bodyB.userData

        nameA = objectA.__class__.__name__
        nameB = objectB.__class__.__name__

        # # Agent to Obstacles collision
        # if isinstance(objectA, Agent) and isinstance(objectB, StaticCircle):
        #     print(nameA + " " + str(objectA.id)
        #           + " end collision " + nameB + " " + str(objectB.id)
        #           + " at " + str(pygame.time.get_ticks() / 1000.0) + " s")
        # if isinstance(objectA, StaticCircle) and isinstance(objectB, Agent):
        #     print(nameB + " " + str(objectB.id)
        #           + " end collision " + nameA + " " + str(objectA.id)
        #           + " at " + str(pygame.time.get_ticks() / 1000.0) + " s")
        #
        # # Agent to Wall collision
        # if isinstance(objectA, Agent) and isinstance(objectB, Wall):
        #     print(nameA + " " + str(objectA.id)
        #           + " end collision " + nameB + " " + str(objectB.id)
        #           + " at " + str(pygame.time.get_ticks() / 1000.0) + " s")
        # if isinstance(objectA, Wall) and isinstance(objectB, Agent):
        #     print(nameB + " " + str(objectB.id)
        #           + " end collision " + nameA + " " + str(objectA.id)
        #           + " at " + str(pygame.time.get_ticks() / 1000.0) + " s")

        pass

    def PreSolve(self, contact, oldManifold):
        pass

    def PostSolve(self, contact, impulse):
        pass
