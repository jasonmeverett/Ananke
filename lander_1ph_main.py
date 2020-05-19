"""
======
ANANKE
======

Example 1
---------
Solves a 3D lander control problem

@author: jasonmeverett
"""

from ananke.opt import *
from ananke.examples import *
from ananke.orbit import *
from ananke.frames import *
from ananke.util import *
from ananke.planets import *
import lander_1ph_setup as lo
from scipy.linalg import norm
import numpy as np
import pygmo as pg
import json


# =============================================================================
#                                                               PROBLEM SET UP
# =============================================================================

# Parameters
mu = 4902799000000.0
R_eq = 1738000.0
Omega = 0.0000026616665
Tmax = 100000.0
isp = 450.0
etaLB = 0.2
etaUB = 0.9

# Configure starting state
m0 = 30000.0
alta = 100000.0
altp = 15000.0
ra = alta + R_eq
rp = altp + R_eq
sma = 0.5*(ra+rp)
ecc = (ra-rp)/(ra+rp)
inc = 0.0
raan = 0.0
argper = -20.0
ta = 0.0
r0_I,v0_I = elts_to_rv(sma,ecc,inc,raan,argper,ta,mu,degrees=True)

LSlat = 0.0
LSlon = 0.0
LSalt = 0.0
rf = [200.0, 0.0, 0.0]
vf = [-15.0, 0.0, 0.0]
R_PF_UEN = Rot_PF_UEN(LSlon, LSlat, False)
R_UEN_PF = R_PF_UEN.inv()
parLand = [Omega, R_eq, LSlat, LSlon, LSalt] + rf + vf + np.reshape(R_UEN_PF.as_dcm(), (9,)).tolist()

# Configure Ananke optimizer. This Python class directly inherits the functions
# of a PyGMO problem() class, and can be used as such in code.
ao = Ananke_Config()

# Configure a trajectory leg.
num_nodes = 10
tl1 = TrajLeg(num_nodes)

# Dynamics and objective
tl1.set_len_X_U(7, 4)
tl1.set_dynamics(lo.f, lo.dfdX, lo.dfdU, [mu, Tmax, isp])
tl1.set_obj(lo.Jfuel, lo.dJfuel, ObjectiveFlags.LAGRANGE, [Tmax])
# tl1.set_obj(lo.Jctrl, lo.dJctrl, ObjectiveFlags.LAGRANGE, [Tmax])

# Boundary constraints
tl1.add_eq(lo.g_X0, lo.dg_X0, 7, RegionFlags.FRONT, params=(r0_I.tolist() + v0_I.tolist() + [m0]))
tl1.add_eq(lo.g_Xf, lo.dg_Xf, 6, RegionFlags.BACK, params=parLand, td=True)

# Path constraints
tl1.add_eq(lo.g_conU, lo.dg_conU, 1, RegionFlags.PATH, params=[])
tl1.add_ineq(lo.g_conEtaLB, lo.dg_conEtaLB, 1, RegionFlags.PATH, params=[etaLB])
tl1.add_ineq(lo.g_conEtaUB, lo.dg_conEtaUB, 1, RegionFlags.PATH, params=[etaUB])

# Set time of flights, and bounds
tl1.set_TOF(100.0, 1000.0)
bnds_min = 11 * [-2000000]
bnds_max = 11 * [ 2000000]
tl1.set_bounds(bnds_min, bnds_max)

# Add a trajectory leg.
ao.add_leg(tl1)
ao.set_TOF(100.0, 1000.0)

# Configure problem.
prob = pg.problem(ao)
prob.c_tol = 1e-6
algo = pg.algorithm(pg.nlopt('slsqp'))
algo.set_verbosity(50)
algo.extract(pg.nlopt).xtol_rel = 0.0
algo.extract(pg.nlopt).ftol_rel = 0.0
algo.extract(pg.nlopt).maxeval = 500

# Set up initial guess.
X0 = [r0_I[0], r0_I[1], r0_I[2], v0_I[0], v0_I[1], v0_I[2], m0, 0.0, -1.0, 0.0, 0.9]
Xf = [R_eq, 0, 0, 0, 0, 0, 0.5*X0[6], 0.0, -1.0, 0.0, 0.2]
Xinit = [800]
for ii in range(0, num_nodes):
    pc = float(ii)/float(num_nodes)
    Xinit = Xinit +  ( np.array(X0) + pc * (np.array(Xf) - np.array(X0)) ).tolist()

# Xinit = np.load('champion_x.npy')

pop = pg.population(prob)
pop.push_back(Xinit)
pop = algo.evolve(pop)

# Grab first leg data.
outdata = ao.get_array_data(pop.champion_x)[0]
np.save('champion_x', pop.champion_x)

# Plot information.
fig, axs = plt.subplots(3,1,sharex=True,squeeze=True)
axs[0].plot(outdata[:,0],outdata[:,7],marker='*')
axs[1].plot(outdata[:,0],norm(outdata[:,1:4],axis=1) - R_eq,marker='*')
axs[2].plot(outdata[:,0],outdata[:,11],marker='*')
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



