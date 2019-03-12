
from Box2D.b2 import (vec2)
import math
import numpy as np
import argparse
import os
import errno
from datetime import datetime
import scipy.stats

def world_to_pixels(vector, screen_height, ppm):
    """
        Convert box2D coordinates to pygame pixels coordinates
        :param vector: box2d vector
        :param screen_height: height of the screen
        :param ppm: pixel per meter
        :return: pygame pixel
    """
    return vector.x * ppm, screen_height - vector.y * ppm

def pixels_to_world(p, screen_height, ppm):
    """
        Convert pygame pixels coordinates to box2d coordinates
        :param p: pygame pixels
        :param screen_height: height of the screen
        :param ppm: pixel per meter
        :return: box2d vector
    """
    return vec2(p[0] / ppm, (screen_height - p[1]) / ppm)

def print_fps(screen, font, text, color=(255, 0, 0, 255)):
    """
        Draw fps text at the top left
        :param screen: pygame screen
        :param font: font
        :param text: fps text to write
        :param color: color of the text (default red)
    """
    screen.blit(font.render(
        text, True, color), (10, 3))

def normalize(vector):
    """
        Normalize a vector
        :param vector: box2d vector
        :return: normalized vector
    """
    length = vector.length
    if length == 0.:
        invLength = 0.
    else:
        invLength = 1.0 / length
    return vec2(vector.x * invLength, vector.y * invLength)

def deg_to_rad(degree):
    """
        Convert angle degree to radian
        :param degree: angle in degree
        :return: angle in radian
    """
    return degree * (math.pi / 180.0)

def rad_to_deg(radian):
    """
        Convert angle radian to degree
        :param radian: angle in radian
        :return: angle in degree
    """
    return radian * (180.0 / math.pi)

def angle_indirect(vec1, vec2):
    """
        Clock-wise (angle increase in the clockwise way) (sens trigonometrique indirect)

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
        Counterclock-wise (angle increase in the counterclock-wise way) (ens trigonometrique direct)

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
        Rotate the vector in degrees. (clockwise base)
        -> v = Vector(100, 0)
        -> v.rotate(45)
        [70.71067811865476, 70.71067811865474]
    """
    angle = math.radians(angle)
    return vec2(
        (vec.x * math.cos(angle)) - (vec.y * math.sin(angle)),
        (vec.y * math.cos(angle)) + (vec.x * math.sin(angle)))

def mega_slow_function():
    """
        Simply a function performing a lot of square root calculation
    """
    for i in range(500000):
        math.sqrt(9123456)

def min_max_normalization(val, _min, _max, new_min, new_max):
    """
        Normalize input x to new range [new_min, new_max]
        Using MinMax normalization
        :param val: input to normalize
        :param _min: input minimum range value
        :param _max: input maximum range value
        :param new_min: new minimum range value
        :param new_max: new maximum range value
        :return: normalized value in the range [new_min, new_max]
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
        :param val: input to normalize
        :param _min: input minimum range value
        :param _max: input maximum range value
        :return: normalized value in the range [0,1]
    """
    return min_max_normalization(val, _min, _max, 0., 1.)

def min_max_normalization_m1_1(val, _min, _max):
    """
        Normalize input x to range [-1,1]
        Using MinMax normalization and scaling
        :param val: input to normalize
        :param _min: input minimum range value
        :param _max: input maximum range value
        :return: normalized value in the range [-1,1]
    """
    return min_max_normalization(val, _min, _max, -1., 1.)

def get_time_string():
    """
        Get the time in string format 'Ymd_HMSf'
        :return: time in 'Ymd_HMSf' string format
    """
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S_%f")

def get_time_string2():
    """
        Get the time in a more human readable format e.g. 13h45m10s 2019/3/8
        :return: time in a more human readable format
    """
    now = datetime.now()
    return now.strftime("%Hh%Mm%Ss %Y/%m/%d")

def distance(v1, v2):
    """
        Distance between two box2d vectors
        :param v1: first vector
        :param v2: second vector
        :return: distance between the two vectors
    """
    d = np.sqrt((v1.x - v2.x) ** 2 +
                (v1.y - v2.y) ** 2)
    return d

def sqr_distance(v1, v2):
    """
        Squared distance between two box2d vectors
        :param v1: first vector
        :param v2: second vector
        :return: squared distance between the two vectors
    """
    d = (v1.x - v2.x) ** 2 + (v1.y - v2.y) ** 2
    return d

def interquartile_mean(x):
    """
        Return the interquartile mean of given collections x
        :param x: a collection (list, array)
        :return: interquartile mean
    """
    x = sorted(x)
    quart = int(len(x) / 4)
    x = x[quart:len(x) - quart]

    return np.mean(x)

def str2bool(v):
    """
        Convert a string (boolean format) to boolean
        Boolean format e.g.
        'yes', 'true', 't', 'y', or '1' -> True
        'no', 'false', 'f', 'n', or '0' -> False
        :param v: input string
        :return: boolean
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_dir(dir):
    """
        Create the given directory if it doesn't exist
        :param dir: directory name
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
        :param s: input string
    """
    return " ".join(s.split())

def str_to_int(x):
    """
        Convert a string to integer
        :param x: input string
        :return: integer value
    """
    if x == 'None':
        return None
    else:
        return int(x)

def str_to_intlist(l):
    """
        Convert a string to a list of integers
        The string contains numerical value separated by comma ','
        :param l: input string
        :return: list of integers
    """
    if l == 'None':
        return None
    else:
        return [int(e) for e in l.split(',')]

# def confidence_interval_95_10trials(data):
#     x = 1.0 * np.array(data)
#     n = len(x)
#     mean = np.mean(x)
#     std = np.std(x, ddof=1) # sample standard deviation
#     t = 2.262 # inf -> 1.96
#
#     low = mean - t * (std / np.sqrt(n))
#     high = mean + t * (std / np.sqrt(n))
#
#     return low, high

def mean_confidence_interval(data, confidence=0.95):
    """
        Compute the mean with confidence interval for given data
        :param data: input data
        :param confidence: percentage of confidence
        :return: mean and the confidence range
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def weighted_average_10(data):
    """
        Weighted average for 10 values
        Weights are as follows: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        It gives increasing importance to most recent values
        :param data: input data
        :return: weighted average
    """
    w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    if len(data) < len(w):
        return np.average(data, weights=w[:len(data)])
    else:
        return np.average(data, weights=w)