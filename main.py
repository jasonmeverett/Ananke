#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:38:41 2019

@author: jasonmeverett
"""

from ananke.frames import *
from ananke.orbit import *
from ananke.planets import Moon, Earth
from ananke.util import *
from ananke.examples import *

# R1 = DCM_I_UEN(0,-90,degrees=True)
# print(R1.apply([1,0,0]))

# elts0 = rv_to_elts([0,1,2],[0,3,4], Moon.mu)

# r,v = elts_to_rv(Earth.R_eq + 100000, 0.0, d2r(25), d2r(15), d2r(50), 0, Earth.mu)
# print(r)
# print(v)

run_problem2()