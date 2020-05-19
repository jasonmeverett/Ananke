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
import ex1_setup as ex1
import numpy as np
import pygmo as pg
import json


# =============================================================================
#                                                               PROBLEM SET UP
# =============================================================================

# Configure Ananke optimizer. This Python class directly inherits the functions
# of a PyGMO problem() class, and can be used as such in code.
ao = Ananke_Config()

# Configure a trajectory leg.
num_nodes = 70
tl1 = TrajLeg(num_nodes)
tl1.set_len_X_U(2, 1)
tl1.set_dynamics(ex1.f_1D, ex1.dfdX_1D, ex1.dfdU_1D, [1.0])
tl1.add_eq(ex1.g1, ex1.dg1, 2, RegionFlags.FRONT)
tl1.add_eq(ex1.g2, ex1.dg2, 2, RegionFlags.BACK)
tl1.add_ineq(ex1.g3, ex1.dg3, 1, RegionFlags.PATH, [6.0])
# tl1.set_obj(ex1.Jctrl, ex1.dJctrl, ObjectiveFlags.LAGRANGE, [])
tl1.set_obj(ex1.Jfuel, ex1.dJfuel, ObjectiveFlags.LAGRANGE, [])
tl1.set_TOF(1.0, 1.0)
bnds_min = [-100.0, -100.0, -100.0]
bnds_max = [ 100.0,  100.0,  100.0]
tl1.set_bounds(bnds_min, bnds_max)

# Add a trajectory leg.
ao.add_leg(tl1)
ao.set_TOF(0.0, 1.0)

prob = pg.problem(ao)
prob.c_tol = 1e-5
algo = pg.algorithm(pg.nlopt('slsqp'))
algo.set_verbosity(20)
algo.extract(pg.nlopt).xtol_rel = 0.0
algo.extract(pg.nlopt).ftol_rel = 0.0
algo.extract(pg.nlopt).maxeval = 500

X0 = [1.0]
for ii in range(0, num_nodes):
    X0 = X0 + [float(ii)/float(num_nodes), 1.0, 1.0 - float(ii)/float(num_nodes)] 

pop = pg.population(prob)
pop.push_back(X0)
pop = algo.evolve(pop)

# Grab first leg data.
outdata = ao.get_array_data(pop.champion_x)[0]

# Plot information.
fig, axs = plt.subplots(3,1,sharex=True,squeeze=True)
axs[0].plot(outdata[:,0],outdata[:,1],marker='*')
axs[1].plot(outdata[:,0],outdata[:,2],marker='*')
axs[2].plot(outdata[:,0],outdata[:,3],marker='*')
axs[0].grid(which='both')
axs[0].minorticks_on()
axs[1].grid(which='both')
axs[1].minorticks_on()
axs[2].grid(which='both')
axs[2].minorticks_on()
axs[0].set_ylabel('Position')
axs[1].set_ylabel('Velocity')
axs[2].set_ylabel('Control')
plt.show()



