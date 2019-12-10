# -*- coding: utf-8 -*-

from numpy import *
from scipy.linalg import norm


def unit(v):
    v = array(v)
    v = v/norm(v)
    return v

def get_alt(r, R_eq):
    return norm(r)-R_eq

def d2r(d):
    return d*pi/180

def r2d(r):
    return r*180/pi


def recast_pts(data,npts):
    
    dnew = data[0]
    
    npts_original = len(data)
    t_orig = linspace(0.0, 1.0, npts_original)
    t_new = linspace(0.0, 1.0, npts)
    
    # Recast new points
    jj = 0
    for ii in range(0,npts):
        if t_new[ii] > t_orig[jj]:
            print("---> ", ii, " , ", jj)
            jj = jj + 1
        
    return


def get_init_guess(r0,v0,rt,vt,TOF,npts):
    
    tgo = TOF
    dt = TOF/npts
    r = array(r0)
    v = array(v0)
    X = [r[0]]
    Y = [r[1]]
    Z = [r[2]]
    Vx = [v[0]]
    Vy = [v[1]]
    Vz = [v[2]]
    Ax = []
    Ay = []
    Az = []
    for ii in range(0,npts-1):
        
        # Compute commanded accleration
        c0 = -4/tgo*array(v-vt) - 6/tgo*array(vt) - 6/tgo**2.0*array(r-rt)
        
        # Append acceleration
        Ax = Ax + [c0[0] + 1.625]
        Ay = Ay + [c0[1]]
        Az = Az + [c0[2]]

        # Propagate
        Vx = Vx + [Vx[-1] + Ax[-1]*dt - 1.625*dt]
        Vy = Vy + [Vy[-1] + Ay[-1]*dt]
        Vz = Vz + [Vz[-1] + Az[-1]*dt]
        X = X + [X[-1] + Vx[-1]*dt]
        Y = Y + [Y[-1] + Vy[-1]*dt]
        Z = Z + [Z[-1] + Vz[-1]*dt]
        
        r = array([X[-1],Y[-1],Z[-1]])
        v = array([Vx[-1],Vy[-1],Vz[-1]])
        
    # Append last row of accels
    Ax = Ax + [0]
    Ay = Ay + [0]
    Az = Az + [1.625]
    
    return (X,Y,Z,Vx,Vy,Vz,Ax,Ay,Az)

        
    