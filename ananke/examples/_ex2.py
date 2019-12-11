#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 09:48:16 2019

2-dimensional optimization problem.
Coordinate frame:
X   : horizontal
Y   : vertical

This should match optimality!!!!

@author: jasonmeverett
"""

from numpy import *
import matplotlib.pyplot as plt
import pygmo as pg
from scipy.linalg import norm

class prob_2D_lander:
    
    # Function initialization
    def __init__(self, npts=30, tof=520.0, X_initial=[ -330000, 15000.0, 1200, 0.0 ], X_target = [0.0, 200.0, 0, -10.0]):
        self.X_initial=X_initial
        self.X_target = X_target
        self.npts = npts
        self.tof = tof
        return
    
    def get_nic(self):
        return 0
    
    # X, Vx init 
    # Y, Vy init 
    # X, Vx final
    # Y, Vy final
    # X constraints
    # Vx constraints
    # Y constraints
    # Vy constraints
    def get_nec(self):
        return 8 + 4*(self.npts-1)
    
    def get_bounds(self):
        
        # Set up here.
        X_lb = [-400000]*self.npts
        X_ub = [0]*self.npts 
        Vx_lb = [0]*self.npts
        Vx_ub = [1300]*self.npts
        Ax_lb = [-10]*self.npts
        Ax_ub = [10]*self.npts 
        Y_lb = [0]*self.npts 
        Y_ub = [18000]*self.npts
        Vy_lb = [-1000]*self.npts
        Vy_ub = [1000]*self.npts
        Ay_lb = [-10]*self.npts
        Ay_ub = [10]*self.npts
        
        LB = \
            X_lb + \
            Vx_lb + \
            Ax_lb + \
            Y_lb + \
            Vy_lb + \
            Ay_lb
        
        UB = \
            X_ub + \
            Vx_ub + \
            Ax_ub + \
            Y_ub + \
            Vy_ub + \
            Ay_ub
        
        return (LB,UB)
    
    def fitness(self, x):
        return self.run_traj(x, plot_traj=0)
    
    def summary(self, x):
        return self.run_traj(x, plot_traj=1)
    
    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x, 1e-7)
    
    def run_traj(self, x, plot_traj=0):
        
        # Initialize return lists
        OBJVAL = []
        CONSTR_EQ = []
        CONSTR_INEQ = []
        
        # X Vector will be:
        # -----------------
        # X pts
        # Vx pts
        # Ax pts
        # Y pts
        # Vy pts
        # Ay pts
        
        # Collocation timestep
        dt = self.tof/float(self.npts)
        
        # Extract state information
        i0 = 0
        X       = x[i0                   :(i0+self.npts)      ]
        Vx      = x[(i0+self.npts)       :(i0+2*self.npts)    ]
        Ax      = x[(i0+2*self.npts)     :(i0+3*self.npts)    ]
        Y       = x[(i0+3*self.npts)     :(i0+4*self.npts)    ]
        Vy      = x[(i0+4*self.npts)     :(i0+5*self.npts)    ]
        Ay      = x[(i0+5*self.npts)     :(i0+6*self.npts)    ]
        
        # Constant gravity vector
        gvec = [0.0, -1.625]
        
        # Starting and ending constraints
        # Unscaled, because want within 1 meter for each
        con_X_0 = X[0] - self.X_initial[0]
        con_X_f = X[-1] - self.X_target[0]
        con_Y_0 = Y[0] - self.X_initial[1]
        con_Y_f = Y[-1] - self.X_target[1]
        con_Vx_0 = Vx[0] - self.X_initial[2]
        con_Vx_f = Vx[-1] - self.X_target[2]
        con_Vy_0 = Vy[0] - self.X_initial[3]
        con_Vy_f = Vy[-1] - self.X_target[3]
        CONSTR_EQ = \
            [con_X_0] + \
            [con_X_f] + \
            [con_Y_0] + \
            [con_Y_f] + \
            [con_Vx_0] + \
            [con_Vx_f] + \
            [con_Vy_0] + \
            [con_Vy_f]
            
        # Path equality constraints
        for ii in range(0, self.npts-1):
            # Positional constraints
            CONSTR_EQ.append( (X[ii+1]-X[ii]) - 0.5*dt*(Vx[ii+1] + Vx[ii]) )
            CONSTR_EQ.append( (Y[ii+1]-Y[ii]) - 0.5*dt*(Vy[ii+1] + Vy[ii]) )
            
        for ii in range(0, self.npts-1):
            # Velocity constraints
            CONSTR_EQ.append( (Vx[ii+1]-Vx[ii]) - (dt*gvec[0] + 0.5*dt*(Ax[ii+1] + Ax[ii])) )
            CONSTR_EQ.append( (Vy[ii+1]-Vy[ii]) - (dt*gvec[1] + 0.5*dt*(Ay[ii+1] + Ay[ii])) )   
            
        # Other objective - minimize control effort
        OBJVAL = [ sum( [ 0.5*dt*( (Ax[ii]**2.0+Ay[ii]**2.0) + (Ax[ii+1]**2.0+Ay[ii+1]**2.0) ) for ii in range(0, self.npts-1) ] ) ]

        
        # Plot results.
        if plot_traj == 1:
            
            t_arr = linspace(0.0,self.tof,self.npts)
            
            plt.figure(1)
            plt.plot(0.001*X,0.001*Y,'*-r',linewidth=2.0)
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5')
            plt.grid(which='minor', linestyle=':', linewidth='0.5')
            plt.xlabel('Downrange (km)')
            plt.ylabel('Altitude (km)')
            
            plt.figure(2)
            plt.plot(t_arr,Ax, '*-b',linewidth=2.0,label='Accel X')
            plt.plot(t_arr,Ay, '*-r',linewidth=2.0,label='Accel Y')
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5')
            plt.grid(which='minor', linestyle=':', linewidth='0.5')
            plt.legend()
            plt.xlabel('Time (sec)')
            plt.ylabel('Accel (m/s)')
            
            plt.show()

        # Return everything
        return OBJVAL + CONSTR_EQ + CONSTR_INEQ


def run_problem2():
    """
    Solves the minimum control problem of a 2-D lander under a 
    uniform gravity field. Employs a trapezoidal collocation method
    """
    
    # Initial points
    X_initial=[ -330000, 15000.0, 1200, 0.0 ]
    X_target = [0.0, 200.0, 0, -10.0]
    npts = 15
    tof = 550.0
    
    udp = prob_2D_lander(npts=npts,tof=tof,X_initial=X_initial,X_target=X_target)
    prob = pg.problem(udp)
    prob.c_tol = [10]*4 + [1]*4 + [10]*2*(npts-1) + [1]*2*(npts-1)   
    
    algo = pg.algorithm(pg.nlopt('slsqp'))
    algo.set_verbosity(20)
    algo.extract(pg.nlopt).xtol_rel = 0
    algo.extract(pg.nlopt).ftol_rel = 0
    algo.extract(pg.nlopt).maxeval = 500
    
    # Uncomment this for a good initial guess.
    dt = tof/npts
    X_g = list(linspace(X_initial[0], X_target[0], npts))
    Y_g = list(linspace(X_initial[1], X_target[1], npts))
    Vx_g = list(linspace(X_initial[2], X_target[2], npts))
    Vy_g = list(linspace(X_initial[3], X_target[3], npts))
    Ux_g = list([(Vx_g[ii+1] - Vx_g[ii])/dt for ii in range(0,npts-1)]) + [1.0]
    Uy_g = list([(Vy_g[ii+1] - Vy_g[ii])/dt for ii in range(0,npts-1)]) + [1.0]
    X0 = X_g + Vx_g + Ux_g + Y_g + Vy_g + Uy_g
    
    # Create population of 1.
    pop = pg.population(prob)
    pop.push_back(X0)
    
    # Evolve
    pop = algo.evolve(pop)
    
    is_feas = prob.feasibility_x(pop.champion_x)
    if is_feas:
        print("===================")
        print("FEASIBLE TRAJECTORY")
        print("===================")
        udp.summary(pop.champion_x)


