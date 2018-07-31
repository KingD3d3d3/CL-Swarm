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
    import Global
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
    from .. import Global

# Event when there is a collision
class MyContactListener(contactListener):

    def __init__(self):
        contactListener.__init__(self)

    def BeginContact(self, contact):
        pass

    def EndContact(self, contact):
        pass

    def PreSolve(self, contact, oldManifold):
        pass

    def PostSolve(self, contact, impulse):
        pass

