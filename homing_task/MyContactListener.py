from Box2D.b2 import (world, polygonShape, vec2, contactListener)
import res.colors as Color
from Circle import Circle, StaticCircle
from Agent import Agent
from AgentHoming import AgentHoming
import pygame, sys
from Border import Wall
from Box import StaticBox

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

            print("Agent: {}, collision {}: {}, time to goal: {:5.3f}, goal reached count: {:3.0f}, time: {:5.3f}"
                  .format(agent.id, "StaticCircle", obstacle.id, agent.elapsedTime / 1000.0, agent.goalReachedCount,
                          pygame.time.get_ticks() / 1000.0))
        if isinstance(objectA, StaticCircle) and isinstance(objectB, AgentHoming):
            agent = objectB
            obstacle = objectA

            # Change colors
            agent.color = Color.DeepSkyBlue
            agent.raycastSideColor = Color.Magenta
            agent.raycastFrontColor = Color.Magenta

            print("Agent: {}, collision {}: {}, time to goal: {:5.3f}, goal reached count: {:3.0f}, time: {:5.3f}"
                  .format(agent.id, "StaticCircle", obstacle.id, agent.elapsedTime / 1000.0, agent.goalReachedCount,
                          pygame.time.get_ticks() / 1000.0))

        # Agent to Wall collision
        if isinstance(objectA, AgentHoming) and isinstance(objectB, Wall):
            agent = objectA
            obstacle = objectB
            print("Agent: {}, collision {}: {}, time to goal: {:5.3f}, goal reached count: {:3.0f}, time: {:5.3f}"
                  .format(agent.id, "Wall", obstacle.id, agent.elapsedTime / 1000.0, agent.goalReachedCount,
                          pygame.time.get_ticks() / 1000.0))
        if isinstance(objectA, Wall) and isinstance(objectB, AgentHoming):
            agent = objectB
            obstacle = objectA
            print("Agent: {}, collision {}: {}, time to goal: {:5.3f}, goal reached count: {:3.0f}, time: {:5.3f}"
                  .format(agent.id, "Wall", obstacle.id, agent.elapsedTime / 1000.0, agent.goalReachedCount,
                          pygame.time.get_ticks() / 1000.0))
        pass

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
