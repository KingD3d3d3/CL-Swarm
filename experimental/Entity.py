class Entity(object):

    def __init__(self, entityManager, gameObject):
        self.body = gameObject.body
        self.position = self.body.position
        self.angle = self.body.angle

        self.previousPosition = self.body.position
        self.previousAngle = self.body.angle

        # Add the new entity to the EntityManager
        entityManager.addEntity(self)
