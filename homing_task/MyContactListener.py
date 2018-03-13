from Box2D.b2 import (world, polygonShape, vec2, contactListener)
import res.colors as Color
from Circle import Circle


# Event when there is a collision
class MyContactListener(contactListener):
    def __init__(self):
        contactListener.__init__(self)

    def BeginContact(self, contact):
        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body

        o1 = bodyA.userData
        o2 = bodyB.userData

        if isinstance(o1, Circle) or isinstance(o2, Circle):
            o2.color = Color.Blue # change color

        # Circle and Ground collision
        # if(isinstance(o1, Ground) and isinstance(o2,Circle)):
        #     o2.color = (255, 255, 0, 255)
        pass

    def EndContact(self, contact):
        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body

        o1 = bodyA.userData
        o2 = bodyB.userData

        if (isinstance(o1, Circle) or isinstance(o2, Circle)):
            o2.color = Color.Lime

        pass

    def PreSolve(self, contact, oldManifold):
        pass

    def PostSolve(self, contact, impulse):
        pass
