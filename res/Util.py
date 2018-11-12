
from Box2D.b2 import (vec2)
import math
import numpy as np
import argparse
import os
import errno
from datetime import datetime

def world_to_pixels(vector, screen_height, ppm):
    return vector.x * ppm, screen_height - vector.y * ppm

def pixels_to_world(p, screen_height, ppm):
    return vec2(p[0] / ppm, (screen_height - p[1]) / ppm)

def print_fps(screen, font, text, color=(255, 0, 0, 255)):  # red
    """
        Draw fps text at the top left
    """
    screen.blit(font.render(
        text, True, color), (10, 3))

def normalize(vector):
    length = vector.length
    if length == 0.:
        invLength = 0.
    else:
        invLength = 1.0 / length
    return vec2(vector.x * invLength, vector.y * invLength)

def deg_to_rad(degree):
    return degree * (math.pi / 180.0)

def rad_to_deg(radian):
    return radian * (180.0 / math.pi)

def angle_indirect(vec1, vec2):
    """
        Clock-wise (angle augmente dans le sens trigonometrique indirect)

        Computes the angle between a and b, and returns the angle in degrees.
        Kivy vector angle implementation
        -> Vector(100, 0).angle((0, 100))
            -90.0
        -> Vector(87, 23).angle((-77, 10))
            -157.7920283010705
        -> Vector(0, 1).angle((1, 0))
            90.0
    """
    ang = -(180 / math.pi) * math.atan2(
        vec1.x * vec2.y - vec1.y * vec2.x,
        vec1.x * vec2.x + vec1.y * vec2.y)
    return ang

def angle_direct(vec1, vec2):
    """
        Counterclock-wise (angle augmente dans le sens trigonometrique direct)

        Computes the angle between a and b, and returns the angle in degrees.
        Kivy vector angle implementation
        -> Vector(100, 0).angle((0, 100))
            90.0
        -> Vector(87, 23).angle((-77, 10))
            157.7920283010705
        -> Vector(0, 1).angle((1, 0))
            -90.0
    """
    ang = (180 / math.pi) * math.atan2(
        vec1.x * vec2.y - vec1.y * vec2.x,
        vec1.x * vec2.x + vec1.y * vec2.y)
    return ang

def rotate(vec, angle):
    """
        Rotate the vector with an angle_indirect in degrees.
        -> v = Vector(100, 0)
        -> v.rotate(45)
        [70.71067811865476, 70.71067811865474]
    """
    angle = math.radians(angle)
    return vec2(
        (vec.x * math.cos(angle)) - (vec.y * math.sin(angle)),
        (vec.y * math.cos(angle)) + (vec.x * math.sin(angle)))

def mega_slow_function():
    for i in range(500000):
        math.sqrt(9123456)

def min_max_normalization(val, _min, _max, new_min, new_max):
    """
        Normalize input x to new range [new_min, new_max]
        Using MinMax normalization
    """
    if _min == _max:
        return (new_min + new_max) / 2
    elif val <= _min:
        return new_min
    elif val >= _max:
        return new_max
    else:
        return (((val - _min) / (_max - _min)) * (new_max - new_min)) + new_min

def min_max_normalization_0_1(val, _min, _max):
    """
        Normalize input x to range [0,1]
        Using MinMax normalization
    """
    return min_max_normalization(val, _min, _max, 0., 1.)

def min_max_normalization_m1_1(val, _min, _max):
    """
        Normalize input x to range [-1,1]
        Using MinMax normalization and scaling
    """
    return min_max_normalization(val, _min, _max, -1., 1.)

def get_time_string():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S_%f")

def get_time_string2():
    """
        Human readable format
    """
    now = datetime.now()
    return now.strftime("%Hh%Mm%Ss %Y/%m/%d")

def distance(v1, v2):
    d = np.sqrt((v1.x - v2.x) ** 2 +
                (v1.y - v2.y) ** 2)
    return d

def sqr_distance(v1, v2):
    d = (v1.x - v2.x) ** 2 + (v1.y - v2.y) ** 2
    return d

def interquartile_mean(x):
    """
        Return the Interquartile Mean of given collections x
    """
    x = sorted(x)
    quart = int(len(x) / 4)
    x = x[quart:len(x) - quart]

    return np.mean(x)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_dir(dir):
    """
        Create the given directory if it doesn't exist
    """
    if not os.path.exists(os.path.dirname(dir)):
        try:
            os.makedirs(os.path.dirname(dir))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def remove_blank(s):
    """
        Remove leading spaces, trailing spaces, successive spaces, newline characters and tab characters
    """
    return " ".join(s.split())

def str_to_int(x):
    if x == 'None':
        return None
    else:
        return int(x)