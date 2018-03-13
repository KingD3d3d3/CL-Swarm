from Box2D.b2 import (vec2)
from Util import *

class EntityManager(object):

    def __init__(self):
        self.entities = []

    def updateCurrentState(self):
        for i in xrange(len(self.entities)):
            bodyPosition = self.entities[i].body.position
            bodyAngle = self.entities[i].body.angle

            self.entities[i].previousPosition = bodyPosition
            self.entities[i].anglePrevious = bodyAngle

    def interpolate(self, alpha):
        for i in xrange(len(self.entities)):

            bodyPosition = self.entities[i].body.position
            bodyAngle = self.entities[i].body.angle

            prev_pos = self.entities[i].previousPosition
            prev_angle = self.entities[i].previousAngle

            newPos = vec2(bodyPosition.x * alpha + prev_pos.x * (1.0 - alpha),
                          bodyPosition.y * alpha + prev_pos.y * (1.0 - alpha))
            newAngle = bodyAngle * alpha + prev_angle * (1.0 - alpha)

            self.entities[i].position = newPos
            self.entities[i].angle = newAngle

            #print('self.body.transform.position', self.body.transform.position)
            #print('self.body.position', self.body.position)

    def addEntity(self, entity):
        self.entities.append(entity)

    def removeEntity(self, entity):
        self.entities.remove(entity)

    def clearList(self):
        self.entities = []
