#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:51:02 2019

@author: jasonmeverett
"""

from numpy import *
import pygmo as pg

class prob_1D_helloCollocation():
    
    def __init__(self,npts=20,tof=1.0):
        self.npts = npts
        self.tof = tof
        return
    
    # Starting position, velocity 
    # Final position, velocity
    # Collocation constraints (x2)
    def get_nec(self):
        return 4 + 2*(self.npts-1)
    
    def get_nic(self):
        return 0
    
    # Decision vector is X, V, U
    def get_bounds(self):
        LB = \
            [0] * self.npts + \
            [0] * self.npts + \
            [-30] * self.npts 
        UB = \
            [1] * self.npts + \
            [10] * self.npts + \
            [30] * self.npts
        return (LB, UB)
    
    def fitness(self,x):
        return self.run_traj(x,plot_traj=0)
    
    def summary(self,x):
        return self.run_traj(x,plot_traj=1)
    
    def gradient(self,x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x, 1e-8)
    
    def run_traj(self,x,plot_traj=0):
        import matplotlib.pyplot as plt
        
        # Initialize Return lists
        OBJVAL = []
        CONSTR_EQ = []
        CONSTR_INEQ = []
        
        # Collocation timestep
        dt = self.tof/self.npts
        
        # Collect parameters
        X = x[0:self.npts]
        V = x[self.npts:2*self.npts]
        U = x[2*self.npts:3*self.npts]
        
        # Starting and ending constraints
        CONSTR_EQ = \
            [X[0]] + \
            [X[-1]-1.0] + \
            [V[0]] + \
            [V[-1]]
            
        # Path constraints
        for ii in range(0, self.npts-1):
            CONSTR_EQ.append( (X[ii+1] - X[ii]) - 0.5*dt*(V[ii+1] + V[ii])  )
            CONSTR_EQ.append( (V[ii+1] - V[ii]) - 0.5*dt*(U[ii+1] + U[ii])  )
        
        # Objective value
        OBJVAL = [sum([0.5*dt*(U[ii+1]**2.0 + U[ii]**2.0) for ii in range(0, self.npts-1)])]
        
        # Plot trajectory
        if plot_traj == 1:
            t_arr = linspace(0.0,self.tof,self.npts)
            plt.figure(1)
            
            plt.subplot(3,1,1)
            plt.plot(t_arr,X,"*-r")
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5')
            plt.grid(which='minor', linestyle=':', linewidth='0.5')
            
            plt.subplot(3,1,2)
            plt.plot(t_arr,V,"*-b")
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5')
            plt.grid(which='minor', linestyle=':', linewidth='0.5')
            
            plt.subplot(3,1,3)
            plt.plot(t_arr,U,"*-g")
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5')
            plt.grid(which='minor', linestyle=':', linewidth='0.5')
            
            plt.show()

        # Return parameters
        return OBJVAL + CONSTR_EQ + CONSTR_INEQ
    

def run_problem1():
    """
    Example of a collocation method 1-D problem. Employs a trapezoidal
    collocation method.
    """    
    
    # Problem definition
    udp1 = prob_1D_helloCollocation(50,1.0)
    prob = pg.problem(udp1)
    prob.c_tol = 1e-7

    # Algorithm definition
    uda = pg.nlopt('slsqp')
    uda.xtol_rel = 0
    uda.ftol_rel = 0
    uda.maxeval = 100
    algo = pg.algorithm(uda)
    algo.set_verbosity(10)
    
    # Evolve population
    pop = pg.population(prob, 1000)
    pop = algo.evolve(pop)
    
    # Plot results
    udp1.summary(pop.champion_x)
    return
