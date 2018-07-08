from __future__ import division
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (vec2)
import math
import numpy as np
import time

from datetime import datetime
try:
    from Setup import *
except:
    # Running in command line
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Running from command line -> Import libraries as package')
    from .Setup import *


def PrintFPS(screen, font, text, color=(255, 0, 0, 255)):  # red
    """
        Draw fps text at the top left
    """
    screen.blit(font.render(
        text, True, color), (10, 3))

def worldToPixels(vector):
    return vector.x * PPM, SCREEN_HEIGHT - vector.y * PPM


def pixelsToWorld((a, b)):
    return vec2(a / PPM, (SCREEN_HEIGHT - b) / PPM)


def normalize(vector):
    length = vector.length
    if length == 0.:
        invLength = 0.
    else:
        invLength = 1.0 / length
    return vec2(vector.x * invLength, vector.y * invLength)


def degToRad(degree):
    return degree * (math.pi / 180.0)


def radToDeg(radian):
    return radian * (180.0 / math.pi)


def angle(vec1, vec2):
    '''Computes the angle between a and b, and returns the angle in
    degrees.
    pytorch vector angle implementation
    >>> Vector(100, 0).angle((0, 100))
    -90.0
    >>> Vector(87, 23).angle((-77, 10))
    -157.7920283010705
    >>> Vector(0, 1).angle((1, 0))
    90.0
    '''
    angle = -(180 / math.pi) * math.atan2(
        vec1.x * vec2.y - vec1.y * vec2.x,
        vec1.x * vec2.x + vec1.y * vec2.y)
    return angle


def megaSlowFunction():
    for i in xrange(500000):
        a = math.sqrt(9123456)

def minMaxNormalization(x, _min, _max, new_min, new_max):
    """
        Normalize input x to new range [new_min, new_max]
        Using MinMax normalization
    """
    if _min == _max:
        return (new_min + new_max) / 2
    else:
        return (((x - _min) / (_max - _min)) * (new_max - new_min)) + new_min

def minMaxNormalization_0_1(x, _min, _max):
    """
        Normalize input x to range [0,1]
        Using MinMax normalization
    """
    return minMaxNormalization(x, _min, _max, 0., 1.)

def minMaxNormalization_m1_1(x, _min, _max):
    """
        Normalize input x to range [-1,1]
        Using MinMax normalization and scaling
    """
    return minMaxNormalization(x, _min, _max, -1., 1.)




#
# def minMaxNormalization_0_1(x, minX, maxX):
#     """
#         Normalize input x to range [0,1]
#         Using MinMax normalization
#     """
#     if minX == maxX:
#         return 0.5
#     else:
#         return (x - minX) / (maxX - minX)
#
# def minMaxNormalization_m1_1(x, minX, maxX):
#     """
#         Normalize input x to range [-1,1]
#         Using MinMax normalization and scaling
#     """
#     if minX == maxX:
#         return 0
#     else:
#         return 2 * (x - minX) / (maxX - minX) - 1

def getTimeString():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S_%f")

def getTimeString2():
    """
        Human readable format
    """
    now = datetime.now()
    return now.strftime("%Hh%Mm%Ss %Y/%m/%d")

def distance(v1, v2):
    distance = np.sqrt((v1.x - v2.x) ** 2 +
                       (v1.y - v2.y) ** 2)
    return distance