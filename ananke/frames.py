#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:23:04 2019

@author: jasonmeverett
"""

from scipy.spatial.transform import Rotation as R
from numpy import *
from ananke.planets import *

def Rot_I_Perifocal(Om, i, om, degrees=True):
    
    if degrees:
        Om = Om * pi/180
        i  = i  * pi/180
        om = om * pi/180
        
    R3 = R.from_dcm([
        [cos(-om),sin(-om),0],
        [-sin(-om),cos(-om),0],
        [0,0,1]
        ]).inv()
    
    R2 = R.from_dcm([
        [1,0,0],
        [0,cos(-i),sin(-i)],
        [0,-sin(-i),cos(-i)]
        ]).inv()
    
    R1 = R.from_dcm([
        [cos(-Om),sin(-Om),0],
        [-sin(-Om),cos(-Om),0],
        [0,0,1]
        ]).inv()
    
    return R3 * R2 * R1
    

# Get the inertial position of a landing site.
def Pos_LS(lon,lat,alt,R_eq=1738e3,degrees=False):
    """
    Get the inertial position of a landing site based on planetary location.
    Does not yet incorporate planetary rotation.
    """

    if degrees == True:
        lon = lon*pi/180
        lat = lat*pi/180
    
    # Grab the planetary rotation.
    R_I_UEN = DCM_I_UEN(lon,lat)
    
    # Based on altitude
    X_UEN = array([R_eq + alt, 0, 0])
    
    # Convert to inertial
    X_I = R_I_UEN.inv().apply(X_UEN)
    
    return X_I
    

# Construct a DCM that represents the transformation from a planetary inertial
# frame to an Up-East-North frame. Expects lat and lon in radians
def Rot_PF_UEN(lon,lat,degrees=False):
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
        [cos(lat),0,sin(lat)],
        [0,1,0],
        [-sin(lat),0,cos(lat)]])
    
    # Combine rotations
    return R2*R1


# Construct a DCM that represents the transformation from a planetary inertial
# frame to a planetary-fixed frame. Rotation around the Z-axis.
def Rot_I_PF(Om, ep, t,degrees=False):
    """
    Convert a latitude and a longitude to a UEN DCM.
    X - Meridian
    Z - North Pole
    Y - Z x X
    """
    
    if degrees == True:
        Om = Om*pi/180
    
    # Calculate total rotation angle.
    th = Om*(t-ep)
    
    # First rotation is longitude along the Z-axis.
    R1 = R.from_dcm([
        [cos(th),sin(th),0],
        [-sin(th),cos(th),0],
        [0,0,1]])

    
    # Combine rotations
    return R1   
    
