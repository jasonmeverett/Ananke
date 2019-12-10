# -*- coding: utf-8 -*-

from numpy import *
from scipy.linalg import norm


def unit(v):
    v = array(v)
    v = v/norm(v)
    return v


def d2r(d):
    return d*pi/180

def r2d(r):
    return r*180/pi