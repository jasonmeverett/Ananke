#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:51:36 2019

@author: jasonmeverett
"""


from numpy import *
from ananke.util import unit
from scipy.linalg import norm
from scipy.spatial.transform import Rotation as R

def calc_o_odot(a,e,nu,mu,degrees=False):
    if degrees:
        nu = nu*pi/180
        
    # Get the distance from the central body
    E = 2*arctan(sqrt((1-e)/(1+e)) * tan(nu/2) )
    r = a*(1-e*cos(E))
    
    # Obtain the position and velocity vector in the perifocal frame
    o = r*array([cos(nu),sin(nu),0])
    odot = sqrt(mu*a)/r*array([-sin(E), sqrt(1-e**2.0)*cos(E), 0])
    
    return (o, odot)

# https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
def elts_to_rv(a,e,i,Om,om,nu,mu,degrees=False):
    
    if degrees:
        i  = i  * pi/180
        Om = Om * pi/180
        om = om * pi/180
        nu = nu * pi/180
        
    o, odot = calc_o_odot(a, e, nu, mu, degrees=False)
    
    # Constrict transformation matrices
    # NOTE: No inverse required here because construction occurs from dcm 
    # directly. if using euler angles, then an inv() call is neeeded.
    R1 = R.from_dcm([[cos(-om),sin(-om),0],[-sin(-om),cos(-om),0],[0,0,1]])
    R2 = R.from_dcm([[1,0,0],[0,cos(-i),sin(-i)],[0,-sin(-i),cos(-i)]])
    R3 = R.from_dcm([[cos(-Om),sin(-Om),0],[-sin(-Om),cos(-Om),0],[0,0,1]])
    R_p_i = R3 * R2 * R1
    
    r = R_p_i.apply(o)
    v = R_p_i.apply(odot)
    return (r,v)


# https://downloads.rene-schwarz.com/download/M002-Cartesian_State_Vectors_to_Keplerian_Orbit_Elements.pdf
def rv_to_elts(r, v, mu):
    
    # Cast to array.
    r = array(r)
    v = array(v)
    
    # Double check array shapes.
    if not (r.shape == (3,1) or r.shape == (3,)) or not (v.shape == (3,1) or v.shape == (3,)):
        print("ERROR: Incorrect vector shape.")
        return None
    
    # Calculate orbital momentum vector
    h = cross(r,v)
    
    # Obtain eccentricity vector
    evec = cross(v,h)/mu - unit(r)
    
    # Determine the ascending node vector
    n = cross([0,0,1],h)
    
    # True anomaly
    tmp = arccos(dot(evec,r)/(norm(evec)*norm(r)))
    if dot(r, v) >= 0.0:
        nu = tmp
    else:
        nu = 2*pi-tmp
        
    # Calculate orbital inclination
    i = arccos(h[2]/norm(h))
    
    # Orbital eccentricity and eccentric anomaly
    e = norm(evec)
    E = 2*arctan(sqrt((1-e)/(1+e)) * tan(nu/2) )
    
    # Longitude of ascending node, and argument of periapsis
    tmp = arccos(n[0]/norm(n))
    if n[1] >= 0.0:
        Om = tmp
    else:
        Om = 2*pi-tmp
        
    tmp = arccos(dot(n,evec)/(norm(n)*norm(evec)))
    if evec[2] >= 0.0:
        om = tmp
    else:
        om = 2*pi-tmp
        
    # Calculate semi-major axis
    a = 1/(2/norm(r) - norm(v)**2.0/mu)
    

    return (a,e,i,Om,om,nu)
    
    