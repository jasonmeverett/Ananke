"""
======
ANANKE
======

LanderOpt
---------
Solves a 3D Lunar Lander Optimization problem.

@author: jasonmeverett
"""

from ananke.opt import *
from ananke.examples import *
from ananke.orbit import *
from ananke.frames import *
from ananke.util import *
from ananke.planets import *
import numpy as np
from numpy import *
from scipy.linalg import norm
import pygmo as pg
import json

# =============================================================================
#                                                      DYNAMICS AND CONSTRAINTS
# =============================================================================

# -----------------------------------------------------------
# System dynamics for legs.
# -----------------------------------------------------------
def grav(R, mu):
    R = array(R)
    return -mu/norm(R)**3.0*R

# Equations of motion
def f(X, U, T, params):
    mu  = params[0]
    Tm  = params[1]
    isp = params[2]
    g0 = 9.81
    Xdot = np.zeros(7)
    Xdot[0:3] = X[3:6]
    Xdot[3:6] = grav(X[0:3],mu) + Tm*U[3]/X[6]*array(U[0:3])
    Xdot[6] = -Tm*U[3]/(g0*isp)
    return Xdot

def dfdX(X, U, T, params):
    mu  = params[0]
    Tm  = params[1]
    isp = params[2]
    Rk = X[0:3]
    Vk = X[3:6]
    m = X[6]
    u = U[0:3]
    eta = U[3]
    I = eye(3)
    A = zeros((7,7))
    A[0:3,3:6] = I
    A[3:6,6] = -Tm*eta/(m**2.0)*u
    A[3:6,0:3] = -mu/norm(Rk)**3.0*I + 3*mu/norm(Rk)**5.0 * outer(Rk,Rk)
    return A

def dfdU(X, U, T, params):
    mu  = params[0]
    Tm  = params[1]
    isp = params[2]
    m = X[6]
    u = U[0:3]
    eta = U[3]
    g0 = 9.81
    I = eye(3)
    A = zeros((7,4))
    A[3:6,0:3] = Tm*eta/m*I
    A[3:6,3] = Tm/m*u
    A[6,3] = -Tm/(g0*isp)
    return A

# -----------------------------------------------------------
# Objective function - minimum control
# -----------------------------------------------------------   
def Jctrl(X, U, T, params):
    Tm  = params[0]
    J = (Tm * U[3] / X[6])**2.0
    return J

def dJctrl(X, U, T, params):
    Tm  = params[0]
    dJ = np.zeros((1, 11))
    dJ[0, 6] = -2 * (Tm * U[3])**2.0 * (X[6]**(-3.0))
    dJ[0, 10] = 2 * (Tm * U[3] / X[6]) * (Tm / X[6]) 
    return dJ

# -----------------------------------------------------------
# Objective function - minimum fuel
# -----------------------------------------------------------   
def Jfuel(X, U, T, params):
    Tm = params[0]
    J = U[3]
    return J

def dJfuel(X, U, T, params):
    Tm  = params[0]
    dJ = np.zeros((1, 11))
    dJ[0, 10] = 1.0
    return dJ

# -----------------------------------------------------------
# Boundary constraint: starting state
# -----------------------------------------------------------
def g_X0(X, U, T, params):
    g = np.zeros(7)
    g[0] = X[0] - params[0]
    g[1] = X[1] - params[1]
    g[2] = X[2] - params[2]
    g[3] = X[3] - params[3]
    g[4] = X[4] - params[4]
    g[5] = X[5] - params[5]
    g[6] = X[6] - params[6]
    return g

def dg_X0(X, U, T, params):
    dg = np.zeros((7, 11))
    dg[0:7, 0:7] = np.eye(7)
    return dg

# -----------------------------------------------------------
# Boundary eq constraint: ending position magnitude
# -----------------------------------------------------------
def g_Xf(X, U, T, params):
    g = np.zeros(6)
    plOm = params[0]
    Omvec = np.array([0, 0, plOm])
    R_eq = params[1]
    LS_lat = params[2]
    LS_lon = params[3]
    LS_alt = params[4]
    rt_LS = np.array(params[5:8])
    vt_LS = np.array(params[8:11])
    R_I_PF = Rot_I_PF(plOm, 0.0, T)
    R_UEN_PF = R.from_dcm(np.reshape(params[11:20], (3,3)))
    R_UEN_I = R_I_PF.inv() * R_UEN_PF
    rLS_I = R_UEN_I.apply([R_eq + LS_alt,0,0])
    vLS_I = cross([0,0,plOm],rLS_I)
    rt_I = rLS_I + R_UEN_I.apply(rt_LS)
    vt_I = R_UEN_I.apply(vt_LS) + cross([0,0,plOm],rt_I)
    g[0:3] = X[0:3] - rt_I
    g[3:6] = X[3:6] - vt_I
    return g
    
def dg_Xf(X, U, T, params):
    dg = np.zeros((6,11))
    dt = np.zeros(6)
    plOm = params[0]
    Omvec = np.array([0, 0, plOm])
    R_eq = params[1]
    LS_lat = params[2]
    LS_lon = params[3]
    LS_alt = params[4]
    rt_LS = np.array(params[5:8])
    vt_LS = np.array(params[8:11])
    R_I_PF = Rot_I_PF(plOm, 0.0, T)
    R_UEN_PF = R.from_dcm(np.reshape(params[11:20], (3,3)))
    R_UEN_I = R_I_PF.inv() * R_UEN_PF
    rLS_I = R_UEN_I.apply([R_eq + LS_alt,0,0])
    vLS_I = cross([0,0,plOm],rLS_I)
    rt_I = rLS_I + R_UEN_I.apply(rt_LS)
    vt_I = R_UEN_I.apply(vt_LS) + cross([0,0,plOm],rt_I)
    dg[0:6, 0:6] = np.eye(6)
    dt[0:3] = -cross(Omvec, rt_I)
    dt[3:6] = -cross(Omvec, vt_I)
    return dg, dt
    
# -----------------------------------------------------------
# Linking function.
# -----------------------------------------------------------
def l_12(X1, U1, X2, U2, T, params):
    l = np.zeros(7)
    l = X1 - X2
    return l

def dl_12(X1, U1, X2, U2, T, params):
    dl1 = np.zeros((7, 11))
    dl2 = np.zeros((7, 11))
    dl1[0:7, 0:7] = np.eye(7)
    dl2[0:7, 0:7] = -1.0 * np.eye(7)
    return dl1, dl2

# -----------------------------------------------------------
# Path eq constraint: control vector norm of unity
# -----------------------------------------------------------
def g_conU(X, U, T, params):
    g = np.zeros(1)
    g[0] = U[0]**2.0 + U[1]**2.0 + U[2]**2.0 - 1.0
    return g

def dg_conU(X, U, T, params):
    dg = np.zeros((1, 11))
    dg[0, 7] = 2 * U[0]
    dg[0, 8] = 2 * U[1]
    dg[0, 9] = 2 * U[2]
    return dg

# -----------------------------------------------------------
# Path ineq constraint: control throttle bounds
# -----------------------------------------------------------
def g_conEtaLB(X, U, T, params):
    etaLB = params[0]
    g = np.zeros(1)
    g[0] = etaLB - U[3]
    return g
def dg_conEtaLB(X, U, T, params):
    dg = np.zeros((1, 11))
    dg[0, 10] = -1.0
    return dg

def g_conEtaUB(X, U, T, params):
    etaUB = params[0]
    g = np.zeros(1)
    g[0] = U[3] - etaUB
    return g
def dg_conEtaUB(X, U, T, params):
    dg = np.zeros((1, 11))
    dg[0, 10] = 1.0
    return dg
















