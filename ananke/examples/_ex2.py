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
import time


class prob_2D_lander:
    
    # Function initialization
    def __init__(self, npts=30, tof=520.0, X_initial=[ -330000, 15000.0, 1200, 0.0 ], X_target = [0.0, 200.0, 0, -10.0]):
        self.X_initial=X_initial
        self.X_target = X_target
        self.npts = npts
        self.tof = tof
        self.lg = [0.0, -1.625] # Gravity accel vector
        return
    
    # Thrust limiting
    def get_nic(self):
        return self.npts
    
    def get_nec(self):
        return 8 + 4*(self.npts-1)
    
    def get_bounds(self):
        
        # Set up here.
        X_lb = [-400000]*self.npts
        X_ub = [0]*self.npts 
        Vx_lb = [-200]*self.npts
        Vx_ub = [1700]*self.npts
        Ax_lb = [-20]*self.npts
        Ax_ub = [20]*self.npts 
        Y_lb = [0]*self.npts 
        Y_ub = [18000]*self.npts
        Vy_lb = [-1000]*self.npts
        Vy_ub = [1000]*self.npts
        Ay_lb = [-20]*self.npts
        Ay_ub = [20]*self.npts
        LB = X_lb + Vx_lb + Ax_lb + Y_lb + Vy_lb + Ay_lb + [300]
        UB = X_ub + Vx_ub + Ax_ub + Y_ub + Vy_ub + Ay_ub + [800]
        return (LB,UB)
    
    def fitness(self, x):
        return self.run_traj(x, plot_traj=0)
    
    def summary(self, x):
        return self.run_traj(x, plot_traj=1)

    def run_traj(self, x, plot_traj=0):

        nps         = self.npts
        tof         = x[6*nps]
        X_initial   = self.X_initial
        X_target    = self.X_target
        
        # Collocation timestep
        dt = tof/float(nps)
        
        # Extract state information
        i0 = 0
        X       = x[i0             :(i0+nps)      ]
        Vx      = x[(i0+nps)       :(i0+2*nps)    ]
        Ax      = x[(i0+2*nps)     :(i0+3*nps)    ]
        Y       = x[(i0+3*nps)     :(i0+4*nps)    ]
        Vy      = x[(i0+4*nps)     :(i0+5*nps)    ]
        Ay      = x[(i0+5*nps)     :(i0+6*nps)    ]
        
        # Starting and ending constraints
        # Unscaled, because want within 1 meter for each
        con_X_0     = X[0]      - X_initial[0]
        con_Y_0     = Y[0]      - X_initial[1]
        con_Vx_0    = Vx[0]     - X_initial[2]
        con_Vy_0    = Vy[0]     - X_initial[3]
        con_X_f     = X[-1]     - X_target[0]
        con_Y_f     = Y[-1]     - X_target[1]
        con_Vx_f    = Vx[-1]    - X_target[2]
        con_Vy_f    = Vy[-1]    - X_target[3]
        CONSTR_EQ   =   [con_X_0] + [con_Y_0] + [con_Vx_0] + [con_Vy_0] + \
                        [con_X_f] + [con_Y_f] + [con_Vx_f] + [con_Vy_f]
            
        # Path equality constraints
        CONSTR_EQ = CONSTR_EQ + [(X[ii+1]-X[ii]) - 0.5*dt*(Vx[ii+1] + Vx[ii]) for ii in range(0,nps-1)]
        CONSTR_EQ = CONSTR_EQ + [(Y[ii+1]-Y[ii]) - 0.5*dt*(Vy[ii+1] + Vy[ii]) for ii in range(0,nps-1)]
        CONSTR_EQ = CONSTR_EQ + [(Vx[ii+1]-Vx[ii]) - dt*self.lg[0] - 0.5*dt*(Ax[ii+1] + Ax[ii]) for ii in range(0,nps-1)]
        CONSTR_EQ = CONSTR_EQ + [(Vy[ii+1]-Vy[ii]) - dt*self.lg[1] - 0.5*dt*(Ay[ii+1] + Ay[ii]) for ii in range(0,nps-1)]
            
        # Other objective - minimize control effort
        OBJVAL = [ sum( [ 0.5*dt*( (Ax[ii]**2.0+Ay[ii]**2.0) + (Ax[ii+1]**2.0+Ay[ii+1]**2.0) ) for ii in range(0, nps-1) ] ) ]
        
        # Thrust limiting constraints
        CONSTR_INEQ = [ Ax[ii]**2.0 + Ay[ii]**2.0 - 6.0**2.0 for ii in range(0,nps) ]

        # Plot results.
        if plot_traj == 1:
                
            mass0 = 20000
            g0 = 9.81
            isp = 300
            m_arr = [mass0]
            for ii in range(0,nps-1):
                A = 0.5*(sqrt(Ax[ii]**2.0 + Ay[ii]**2.0) + sqrt(Ax[ii+1]**2.0 + Ay[ii+1]**2.0))
                T = m_arr[ii]*A
                mdot = T/(g0*isp)
                m_arr = m_arr + [m_arr[ii]-mdot*dt]
                
            print("FINAL MASS: ", m_arr[-1])
            
            t_arr = linspace(0.0,tof,nps)
            sf = 0.8
            
            # Trajectory arc
            plt.figure(1)
            plt.plot(0.001*X,0.001*Y,'*-b',linewidth=2.0)
            for ii in range(0,nps):
                Xs = [0.001*X[ii], 0.001*X[ii] + Ax[ii]*sf]
                Ys = [0.001*Y[ii], 0.001*Y[ii] + Ay[ii]*sf]
                plt.plot(Xs,Ys,'r',linewidth=1.0)
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5')
            plt.grid(which='minor', linestyle=':', linewidth='0.5')
            plt.xlabel('Downrange (km)')
            plt.ylabel('Altitude (km)')

            # Accelerations
            plt.figure(2)
            plt.plot(t_arr,Ax, '*-b',linewidth=2.0,label='Accel X')
            plt.plot(t_arr,Ay, '*-r',linewidth=2.0,label='Accel Y')
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5')
            plt.grid(which='minor', linestyle=':', linewidth='0.5')
            plt.legend()
            plt.xlabel('Time (sec)')
            plt.ylabel('Accel (m/s^2)')
            
            plt.show()
        
        # Return everything
        return OBJVAL + CONSTR_EQ + CONSTR_INEQ

    def gradient(self, x):

        nps = self.npts
        tof = x[6*nps]
        
        # Calculate dt
        dt = tof/float(nps)
    
        # Entire array zeroed out
        grad = zeros((9 + 4*(nps-1) + nps, 6*nps+1))
        
        # TOF Changes
        J = sum( [ 0.5*dt*( (x[2*nps+ii]**2.0+x[5*nps+ii]**2.0)+ (x[2*nps+ii+1]**2.0+x[5*nps+ii+1]**2.0) ) for ii in range(0, nps-1) ] )
        npf = float(nps)
        grad[0,6*nps] = J*1/x[6*nps]
        grad[(9+0*(nps-1)):(9+1*(nps-1)),6*nps] = -0.5/npf*array([ (x[1*nps+ii+1]+x[1*nps+ii]) for ii in range(0,nps-1) ])
        grad[(9+1*(nps-1)):(9+2*(nps-1)),6*nps] = -0.5/npf*array([ (x[4*nps+ii+1]+x[4*nps+ii]) for ii in range(0,nps-1) ])
        grad[(9+2*(nps-1)):(9+3*(nps-1)),6*nps] = -0.5/npf*array([ (x[2*nps+ii+1]+x[2*nps+ii]) for ii in range(0,nps-1) ]) - self.lg[0]/npf*ones((nps-1,))
        grad[(9+3*(nps-1)):(9+4*(nps-1)),6*nps] = -0.5/npf*array([ (x[5*nps+ii+1]+x[5*nps+ii]) for ii in range(0,nps-1) ]) - self.lg[1]/npf*ones((nps-1,))
        
        # This is for the minimum control problem.
        # Middle parameters are counted twice. Outer parameters are not.
        # Think of cost function as being expanded...
        grad[0,(2*nps):(3*nps)] = dt*x[(2*nps):(3*nps)]
        grad[0,(2*nps+1):(3*nps-1)] = grad[0,(2*nps+1):(3*nps-1)] + dt*x[(2*nps+1):(3*nps-1)]
        grad[0,(5*nps):(6*nps)] = dt*x[(5*nps):(6*nps)]
        grad[0,(5*nps+1):(6*nps-1)] = grad[0,(5*nps+1):(6*nps-1)] + dt*x[(5*nps+1):(6*nps-1)]
        
        # Initial/final constraints wrt. state
        grad[1,0] = 1.0
        grad[2,3*nps] = 1.0
        grad[3,1*nps] = 1.0
        grad[4,4*nps] = 1.0
        grad[5,1*nps-1] = 1.0
        grad[6,4*nps-1] = 1.0
        grad[7,2*nps-1] = 1.0
        grad[8,5*nps-1] = 1.0
        
        # Path constraints
        grad[9              :(9+(nps-1)),       0*nps   :1*nps  ] = -1.0*eye(nps-1, M = nps, k = 0) + eye(nps-1, M = nps, k = 1)
        grad[9              :(9+(nps-1)),       1*nps   :2*nps  ] = -0.5*dt*(eye(nps-1, M = nps, k = 0) + eye(nps-1, M = nps, k = 1))
        grad[(9+nps-1)      :(9+2*(nps-1)),     3*nps   :4*nps  ] = -1.0*eye(nps-1, M = nps, k = 0) + eye(nps-1, M = nps, k = 1)
        grad[(9+nps-1)      :(9+2*(nps-1)),     4*nps   :5*nps  ] = -0.5*dt*(eye(nps-1, M = nps, k = 0) + eye(nps-1, M = nps, k = 1))
        grad[(9+2*(nps-1))  :(9+3*(nps-1)),     1*nps   :2*nps  ] = -1.0*eye(nps-1, M = nps, k = 0) + eye(nps-1, M = nps, k = 1)
        grad[(9+2*(nps-1))  :(9+3*(nps-1)),     2*nps   :3*nps  ] = -0.5*dt*(eye(nps-1, M = nps, k = 0) + eye(nps-1, M = nps, k = 1))
        grad[(9+3*(nps-1))  :(9+4*(nps-1)),     4*nps   :5*nps  ] = -1.0*eye(nps-1, M = nps, k = 0) + eye(nps-1, M = nps, k = 1)
        grad[(9+3*(nps-1))  :(9+4*(nps-1)),     5*nps   :6*nps  ] = -0.5*dt*(eye(nps-1, M = nps, k = 0) + eye(nps-1, M = nps, k = 1))
        
        # Thrust limiting constraints
        grad[(9+4*(nps-1)+0*nps):(9+4*(nps-1)+1*nps),2*nps:3*nps] = 2.0*diag(x[2*nps:3*nps])
        grad[(9+4*(nps-1)+0*nps):(9+4*(nps-1)+1*nps),5*nps:6*nps] = 2.0*diag(x[5*nps:6*nps])
        
        # Reshape
        grad = grad.reshape(( (9 + 4*(nps-1) + nps) * (6*nps+1) , ))
        
        return grad


def run_problem2(npts=20,tof=520.05,X_initial=[ -370000, 15000.0, 1500, 0.0 ],X_target = [0.0, 200.0, 0, -10.0]):
    """
    Solves the minimum control problem of a 2-D lander under a 
    uniform gravity field. Employs a trapezoidal collocation method
    """
    
    udp = prob_2D_lander(npts=npts,tof=tof,X_initial=X_initial,X_target=X_target)
    prob = pg.problem(udp)
    prob.c_tol = [1]*4 + [0.1]*4 + [1]*2*(npts-1) + [0.1]*2*(npts-1) + [0.01]*npts
    
    algo = pg.algorithm(pg.nlopt('slsqp'))
    algo.set_verbosity(20)
    algo.extract(pg.nlopt).xtol_rel = 0
    algo.extract(pg.nlopt).ftol_rel = 0
    algo.extract(pg.nlopt).maxeval = 1000
    
    # Uncomment this for a good initial guess.
    dt = tof/npts
    X_g = list(linspace(X_initial[0], X_target[0], npts))
    Y_g = list(linspace(X_initial[1], X_target[1], npts))
    Vx_g = list(linspace(X_initial[2], X_target[2], npts))
    Vy_g = list(linspace(X_initial[3], X_target[3], npts))
    Ux_g = list([(Vx_g[ii+1] - Vx_g[ii])/dt for ii in range(0,npts-1)]) + [1.0]
    Uy_g = list([(Vy_g[ii+1] - Vy_g[ii])/dt for ii in range(0,npts-1)]) + [1.0]
    X0 = X_g + Vx_g + Ux_g + Y_g + Vy_g + Uy_g + [tof]
    
    # Create population of 1.
    pop = pg.population(prob)
    pop.push_back(X0)
    
    # Evolve
    t1 = time.clock()
    pop = algo.evolve(pop)
    t2 = time.clock()
    print("Solved time: %.8f"%(t2-t1))
    print("TOF: %.3f\n"%(pop.champion_x[-1]))
    
    is_feas = prob.feasibility_x(pop.champion_x)
    if is_feas:
        print("===================")
        print("FEASIBLE TRAJECTORY")
        print("===================")
        
    udp.summary(pop.champion_x)
        


