#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 09:48:16 2019

3-Dimensional full gravity optimization problem.
This should match optimality!!!!

@author: jasonmeverett
"""

from numpy import *
import pykep as pk
import matplotlib.pyplot as plt
import pygmo as pg
from scipy.linalg import norm
from ananke.orbit import *
from ananke.frames import *
from ananke.util import *
from ananke.planets import *
from numba import jit, jitclass
from numba.extending import overload

class prob_3D_lander(object):
    
    # Function initialization
    def __init__(self,npts=30,tof=800):
        
        # Planet definition
        self.mu = 4902.799e9
        self.R_eq = 1738e3
        
        # Collocation information
        self.npts = npts
        
        # Initial state
        ra = self.R_eq+100000
        rp = self.R_eq+15000
        sma = 0.5*(ra+rp)
        ecc = (ra-rp)/(ra+rp)
        r,v = elts_to_rv(sma,ecc,0.0,0.0,d2r(-10.0),0.0,self.mu)
        self.r0_I = r
        self.v0_I = v
        
        # Final state
        self.LS_lat = 0.0
        self.LS_lon = 0.0
        self.LS_alt = 200.0
        self.R_I_UEN = DCM_I_UEN(self.LS_lon, self.LS_lat, degrees=True)
        self.rt_I = Pos_LS(self.LS_lon, self.LS_lat, self.LS_alt, self.R_eq, degrees=True)
        self.vt_I = self.R_I_UEN.inv().apply([-15.0, 0.0, 0.0])
        
        # Convert to LS frame
        self.r0_LS = self.R_I_UEN.apply(self.r0_I - self.rt_I)
        self.v0_LS = self.R_I_UEN.apply(self.v0_I)
        self.rt_LS = [200,0,0]
        self.vt_LS = [-15,0,0]
        
        # Time of flight
        self.tof = tof
        
        return
    
    def get_nic(self):
        return 0
    
    # X, Vx init 
    # Y, Vy init 
    # Z, Vz init
    # X, Vx final
    # Y, Vy final
    # Z, Vz final
    # X constraints
    # Vx constraints
    # Y constraints
    # Vy constraints
    # Z constraints
    # Vz constraints
    def get_nec(self):
        return 12 + 6*(self.npts-1)
    
    # TODO: Set these bounds in a smarter fashion.
    def get_bounds(self):
        
        # Set up here.
        sf_r = 1.0
        sf_v = 2.0
        r_lb = [-sf_r*norm(self.r0_LS)]*3*self.npts
        r_ub = [ sf_r*norm(self.r0_LS)]*3*self.npts
        v_lb = [-sf_v*norm(self.v0_LS)]*3*self.npts
        v_ub = [ sf_v*norm(self.v0_LS)]*3*self.npts
        u_lb = [-20]*3*self.npts
        u_ub = [ 20]*3*self.npts
        
        LB = \
            r_lb + \
            v_lb + \
            u_lb
        
        UB = \
            r_ub + \
            v_ub + \
            u_ub
        
        return (LB, UB)
    
    def fitness(self, x):
        return self.run_traj(x, plot_traj=0)
    
    def summary(self, x):
        return self.run_traj(x, plot_traj=1)
    
    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x, 1e-3)
    
    def run_traj(self, x, plot_traj=0):
        
        # Initialize return lists
        OBJVAL = []
        CONSTR_EQ = []
        CONSTR_INEQ = []
        
        # X Vector will be:
        # -----------------
        # X pts
        # Y pts
        # Z pts
        # Vx pts
        # Vy pts
        # Vz pts
        # Ax pts
        # Ay pts
        # Ax pts
        
        # Collocation timestep
        dt = self.tof/self.npts
        
        # Extract state information
        i0 = 0
        X       = x[i0                   :(i0+self.npts)      ]
        Y       = x[(i0+self.npts)       :(i0+2*self.npts)    ]
        Z       = x[(i0+2*self.npts)     :(i0+3*self.npts)    ]
        Vx      = x[(i0+3*self.npts)     :(i0+4*self.npts)    ]
        Vy      = x[(i0+4*self.npts)     :(i0+5*self.npts)    ]
        Vz      = x[(i0+5*self.npts)     :(i0+6*self.npts)    ]
        Ux      = x[(i0+6*self.npts)     :(i0+7*self.npts)    ]
        Uy      = x[(i0+7*self.npts)     :(i0+8*self.npts)    ]
        Uz      = x[(i0+8*self.npts)     :(i0+9*self.npts)    ]

        # Starting and ending constraints
        # Unscaled, because want within 1 meter for each
        r0 = array([X[0],Y[0],Z[0]])
        v0 = array([Vx[0],Vy[0],Vz[0]])
        rf = array([X[-1],Y[-1],Z[-1]])
        vf = array([Vx[-1],Vy[-1],Vz[-1]])
        CONSTR_EQ = \
            list(r0-self.r0_LS) + \
            list(v0-self.v0_LS) + \
            list(rf-self.rt_LS) + \
            list(vf-self.vt_LS)
            
        # Path equality constraints
        for ii in range(0, self.npts-1):
            
            # State vectors
            rk = array([X[ii],Y[ii],Z[ii]])
            vk = array([Vx[ii],Vy[ii],Vz[ii]])
            uk = array([Ux[ii],Uy[ii],Uz[ii]])
            rkp1 = array([X[ii+1],Y[ii+1],Z[ii+1]])
            vkp1 = array([Vx[ii+1],Vy[ii+1],Vz[ii+1]])
            ukp1 = array([Ux[ii+1],Uy[ii+1],Uz[ii+1]])
            
            # Convert positions to inertial.
            rkI = self.rt_I + self.R_I_UEN.inv().apply(rk)
            rkp1I = self.rt_I + self.R_I_UEN.inv().apply(rkp1)
            
            # Gravity calculation at each timestep.
            gk = -self.mu/norm(rkI)**3.0 * rkI
            gkp1 = -self.mu/norm(rkp1I)**3.0 * rkp1I
            gk = self.R_I_UEN.apply(gk)
            gkp1 = self.R_I_UEN.apply(gkp1)
            
            # Path constraints from dynamics
            CONSTR_EQ = CONSTR_EQ + list( (rkp1-rk) - 0.5*dt*(vk + vkp1) )
            CONSTR_EQ = CONSTR_EQ + list( (vkp1-vk) - 0.5*dt*((uk + ukp1) + (gk + gkp1)) )

        # Other objective - minimize control effort
        objvals = [ 0.5*dt*( (Ux[ii]**2.0 + Uy[ii]**2.0 + Uz[ii]**2.0) + (Ux[ii+1]**2.0 + Uy[ii+1]**2.0 + Uz[ii+1]**2.0) ) for ii in range(0, self.npts-1) ]
        OBJVAL = [ sum(objvals) ]

        # Plot, if enabled
        if plot_traj == 1:
            
            # Time array
            t_arr = linspace(0.0, self.tof, self.npts)
            
            plt.figure(1)
            plt.plot(t_arr, Ux, '*-r')
            plt.plot(t_arr, Uy, '*-b')
            plt.plot(t_arr, Uz, '*-g')
            plt.xlabel('Time (sec)')
            plt.ylabel('Accels (m/s)')
            
            # Altitude array
            alt = [0.001*get_alt( self.rt_I + self.R_I_UEN.apply([X[ii],Y[ii],Z[ii]]) ,self.R_eq) for ii in range(0,self.npts)]
            
            plt.figure(2)
            plt.plot(t_arr, alt, '*-b')
            plt.xlabel('Time (sec)')
            plt.ylabel('Alt (km)')
            
            plt.show()

        # Return everything
        return OBJVAL + CONSTR_EQ + CONSTR_INEQ


def run_problem3():
    """
    Solves the minimum control problem of a 2-D lander under a 
    uniform gravity field. Employs a trapezoidal collocation method
    """
    
    x_optimal_output = 'x_optimal_new.txt'
    
    npts = 10
    tof = 600
    
    udp = prob_3D_lander(npts,tof)
    prob = pg.problem(udp)
    prob.c_tol = 1.0
    
    algo = pg.algorithm(pg.nlopt('slsqp'))
    algo.set_verbosity(1)
    algo.extract(pg.nlopt).xtol_rel = 0
    algo.extract(pg.nlopt).ftol_rel = 0
    algo.extract(pg.nlopt).maxeval = 100
    
    r0_LS = array([ -11832.0089696, -304405.25545013,  0.0 ])
    v0_LS = array([ 293.82000729, 1666.3360656,  0.0  ])
    (X,Y,Z,Vx,Vy,Vz,Ux,Uy,Uz) = get_init_guess(r0_LS,v0_LS,[200,0,0],[-15,0,0],tof,npts)
    X0 = X + Y + Z + Vx + Vy + Vz + Ux + Uy + Uz
    pop = pg.population(prob)
    pop.push_back(X0)
    
    # Initial guess from file.
    # content = []
    # with open(x_optimal_output) as f:
    #     content = f.readlines() 
    # x_opt = [float(item) for item in content]
    # pop = pg.population(prob)
    # pop.push_back(x_opt)
    
    # Uncomment for random population.
    # print("Constructing population...")
    # pop = pg.population(prob, 200)
   
    # Evolve
    print("Evolving...")
    pop = algo.evolve(pop)
    
    is_feas = prob.feasibility_x(pop.champion_x)
    if is_feas:
        print("===================")
        print("FEASIBLE TRAJECTORY")
        print("===================")
        
        fout = open(x_optimal_output,'w+')
        champion_vec = pop.champion_x
        for item in champion_vec:
            fout.write("%30.15f\n"%(item))
        fout.close()

    content = []
    with open(x_optimal_output) as f:
        content = f.readlines() 
    x_opt = [float(item) for item in content]
    udp.summary(x_opt)


