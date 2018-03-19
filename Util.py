# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (vec2)
import math

try:
    from Setup import *
except:
    # Running in command line
    print('Not in Pycharm -> Import as package')
    from .Setup import *

# TODO Describe this file


def worldToPixels(vector):
    return vector.x * PPM, SCREEN_HEIGHT - vector.y * PPM


def pixelsToWorld((a, b)):
    return vec2(a / PPM, (SCREEN_HEIGHT - b) / PPM)


def normalize(vector):
    length = vector.length
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