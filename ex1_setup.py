"""
======
ANANKE
======

Example 1
---------
Solves a 1D control problem.

@author: jasonmeverett
"""

from ananke.opt import *
from ananke.examples import *
import numpy as np
import pygmo as pg
import json

# =============================================================================
#                                                      DYNAMICS AND CONSTRAINTS
# =============================================================================

# -----------------------------------------------------------
# System dynamics for legs.
# -----------------------------------------------------------
def f_1D(X, U, T, params):
    mult = params[0]
    dX = np.zeros(2)
    dX[0] = X[1]
    dX[1] = U[0] * mult
    return dX

def dfdX_1D(X, U, T, params):
    dX = np.zeros((2, 2))
    dX[0,1] = 1.0
    return dX

def dfdU_1D(X, U, T, params):
    dX = np.zeros((2,1))
    mult = params[0]
    dX[1] = mult
    return dX

# -----------------------------------------------------------
# Objective function - minimum control
# -----------------------------------------------------------   
def Jctrl(X, U, T, params):
    J = U[0]**2.0
    return J

def dJctrl(X, U, T, params):
    dJ = np.zeros((1, 3))
    dJ[0, 2] = 2.0*U[0]
    return dJ

# -----------------------------------------------------------
# Objective function - minimum fuel
# -----------------------------------------------------------   
def Jfuel(X, U, T, params):
    J = np.sqrt(U[0]**2.0)
    return J

def dJfuel(X, U, T, params):
    dJ = np.zeros((1, 3))
    dJ[0, 2] = U[0]/np.sqrt(U[0]**2.0)
    return dJ

# -----------------------------------------------------------
# Constraint 1: starting state
# -----------------------------------------------------------
def g1(X, U, T, params):
    g = np.zeros(2)
    g[0] = X[0]
    g[1] = X[1]
    return g

def dg1(X, U, T, params):
    dg = np.zeros((2, 3))
    dg[0, 0] = 1.0
    dg[1, 1] = 1.0
    return dg

# -----------------------------------------------------------
# Constraint 2: ending state
# -----------------------------------------------------------
def g2(X, U, T, params):
    g = np.zeros(2)
    g[0] = X[0] - 1.0
    g[1] = X[1]
    return g

def dg2(X, U, T, params):
    dg = np.zeros((2, 3))
    dg[0, 0] = 1.0
    dg[1, 1] = 1.0
    return dg

# -----------------------------------------------------------
# Constraint 3: controls path constraint limit
# -----------------------------------------------------------
def g3(X, U, T, params):
    ulim = params[0]
    g = np.zeros(1)
    g[0] = np.sqrt(U[0]**2.0) - ulim
    return g

def dg3(X, U, T, params):
    dg = np.zeros((1, 3))
    dg[0, 2] = U[0]/np.sqrt(U[0]**2.0)
    return dg



