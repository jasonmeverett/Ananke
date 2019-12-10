#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:23:04 2019

@author: jasonmeverett
"""

from scipy.spatial.transform import Rotation as R
from numpy import *

# Construct a DCM that represents the transformation from a planetary inertial
# frame to an Up-East-North frame. Expects lat and lon in radians
def DCM_I_UEN(lon,lat,degrees=False):
    """
    Convert a latitude and a longitude to a UEN DCM.
    X - Up 
    Y - East
    Z - North
    """
    
    if degrees == True:
        lon = lon*pi/180
        lat = lat*pi/180
    
    # First rotation is longitude along the Z-axis.
    R1 = R.from_dcm([
        [cos(lon),sin(lon),0],
        [-sin(lon),cos(lon),0],
        [0,0,1]])
    
    # Second rotation is negative latitude along the new Y-axis.
    R2 = R.from_dcm([
        [cos(-lat),0,-sin(-lat)],
        [0,1,0],
        [sin(-lat),0,cos(-lat)]])
    
    # Combine rotations
    return R2*R1

    
    
    
    
