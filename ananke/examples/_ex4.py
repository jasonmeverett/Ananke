#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 09:48:16 2019

2-dimensional optimization problem, with mass involved
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


class prob_2D_lander_mass:
    
    # Function initialization
    def __init__(self, npts=30, tof=520.0, X_initial=[ -330000, 15000.0, 1200, 0.0 ], X_target = [0.0, 200.0, 0, -10.0], \
                 mass0=20000, tmax=66000, isp=300.0, min_control=0):
        self.X_initial=X_initial
        self.X_target = X_target
        self.npts = npts
        self.tof = tof
        self.mass0 = mass0
        self.mc = min_control
        self.tmax = tmax
        self.g0 = 9.81
        self.isp = isp
        self.tfmin = 0.2
        self.tfmax = 0.8
        self.lg = [0.0, -1.625] # Gravity accel vector
        return
    
    # Thrust min, thrust max
    def get_nic(self):
        return self.npts + self.npts
    
    # 2d state (x, vx, y, vy), mass, umag=1
    def get_nec(self):
        return 9 + 5*(self.npts-1) + self.npts
    
    def get_bounds(self):
        
        # Set up here.
        X_lb = [-400000]*self.npts
        X_ub = [2000]*self.npts 
        Vx_lb = [-3000]*self.npts
        Vx_ub = [3000]*self.npts
        Ux_lb = [-3]*self.npts
        Ux_ub = [3]*self.npts 
        Y_lb = [0]*self.npts 
        Y_ub = [18000]*self.npts
        Vy_lb = [-3000]*self.npts
        Vy_ub = [3000]*self.npts
        Uy_lb = [-3]*self.npts
        Uy_ub = [3]*self.npts
        m_lb = [0.1*self.mass0]*self.npts
        m_ub = [self.mass0]*self.npts
        T_lb = [0.0]*self.npts
        T_ub = [1.2]*self.npts
        LB = X_lb + Vx_lb + Ux_lb + Y_lb + Vy_lb + Uy_lb + m_lb + T_lb + [300]
        UB = X_ub + Vx_ub + Ux_ub + Y_ub + Vy_ub + Uy_ub + m_ub + T_ub + [800]
        return (LB,UB)
    
    def fitness(self, x):
        return self.run_traj(x, plot_traj=0)
    
    def summary(self, x):
        return self.run_traj(x, plot_traj=1)

    def run_traj(self, x, plot_traj=0):
        
        # Class variables
        nps         = self.npts
        npf         = float(nps)
        X_initial   = self.X_initial
        X_target    = self.X_target
        em          = self.tmax
        m0          = self.mass0
        g0          = self.g0
        isp         = self.isp
        lg          = self.lg
        tof         = x[(8*nps)]
        
        # Extract decision vector information
        X       = x[(0*nps)     :(1*nps)    ]
        Vx      = x[(1*nps)     :(2*nps)    ]
        Ux      = x[(2*nps)     :(3*nps)    ]
        Y       = x[(3*nps)     :(4*nps)    ]
        Vy      = x[(4*nps)     :(5*nps)    ]
        Uy      = x[(5*nps)     :(6*nps)    ]
        m       = x[(6*nps)     :(7*nps)    ]
        T       = x[(7*nps)     :(8*nps)    ]
        
        # Collocation timestep
        dt = tof/npf
        
        # Starting and ending constraints
        # Unscaled, because want within 1 meter for each
        con_X_0     = X[0]      - X_initial[0]
        con_Y_0     = Y[0]      - X_initial[1]
        con_Vx_0    = Vx[0]     - X_initial[2]
        con_Vy_0    = Vy[0]     - X_initial[3]
        con_m_0     = m[0]      - m0    # Final mass constrained by bound
        con_X_f     = X[-1]     - X_target[0]
        con_Y_f     = Y[-1]     - X_target[1]
        con_Vx_f    = Vx[-1]    - X_target[2]
        con_Vy_f    = Vy[-1]    - X_target[3]
        CONSTR_EQ   =   [con_X_0] + [con_Y_0] + [con_Vx_0] + [con_Vy_0] + [con_m_0] + \
                        [con_X_f] + [con_Y_f] + [con_Vx_f] + [con_Vy_f]
            
        # Path equality constraints
        lgx = lg[0]
        lgy = lg[1]
        CONSTR_EQ = CONSTR_EQ + [(X[ii+1]-X[ii]) - 0.5*dt*(Vx[ii+1] + Vx[ii]) for ii in range(0,nps-1)]
        CONSTR_EQ = CONSTR_EQ + [(Y[ii+1]-Y[ii]) - 0.5*dt*(Vy[ii+1] + Vy[ii]) for ii in range(0,nps-1)]
        CONSTR_EQ = CONSTR_EQ + [(Vx[ii+1]-Vx[ii]) - 0.5*dt*(em*T[ii+1]*Ux[ii+1]/m[ii+1] + em*T[ii]*Ux[ii]/m[ii] + 2*lgx) for ii in range(0,nps-1)]
        CONSTR_EQ = CONSTR_EQ + [(Vy[ii+1]-Vy[ii]) - 0.5*dt*(em*T[ii+1]*Uy[ii+1]/m[ii+1] + em*T[ii]*Uy[ii]/m[ii] + 2*lgy) for ii in range(0,nps-1)]
        CONSTR_EQ = CONSTR_EQ + [(m[ii]-m[ii+1]) - 0.5*dt*em/(g0*isp)*( T[ii+1] + T[ii] ) for ii in range(0,nps-1) ]    
        
        # Other objective
        if self.mc == 0:
            OBJVAL = [ sum( [ 0.5*dt*( T[ii+1] + T[ii] ) for ii in range(0, nps-1) ] ) ]
        else:
            OBJVAL = [ sum( [ 0.5*dt*( (em*T[ii+1]/m[ii+1])**2.0 + (em*T[ii]/m[ii])**2.0 ) for ii in range(0, nps-1) ] ) ]
            
        # Thrust unit vector constraints
        CONSTR_EQ = CONSTR_EQ + [ Ux[ii]**2.0 + Uy[ii]**2.0 - 1.0 for ii in range(0,nps) ]
        
        # Thrust magnitude constraints
        CONSTR_INEQ = [ T[ii] - self.tfmax for ii in range(0,nps) ]
        CONSTR_INEQ = CONSTR_INEQ + [ self.tfmin - T[ii] for ii in range(0,nps) ]
        
        # Plot results.
        if plot_traj == 1:
            
            t_arr = linspace(0.0,tof,nps)
            sf = 4.0
            
            # Trajectory arc
            plt.figure(1)
            plt.subplot(3,1,1)
            plt.plot(0.001*X,0.001*Y,'*-b',linewidth=2.0)
            for ii in range(0,nps):
                Xs = [0.001*X[ii], 0.001*X[ii] + Ux[ii]*sf*T[ii]]
                Ys = [0.001*Y[ii], 0.001*Y[ii] + Uy[ii]*sf*T[ii]]
                plt.plot(Xs,Ys,'r',linewidth=1.0)
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5')
            plt.grid(which='minor', linestyle=':', linewidth='0.5')
            plt.xlabel('Downrange (km)')
            plt.ylabel('Altitude (km)')
            
            # Thrust profile
            plt.subplot(3,1,2)
            plt.plot(t_arr, T, '*-b')
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5')
            plt.grid(which='minor', linestyle=':', linewidth='0.5')
            plt.xlabel('Time (sec)')
            plt.ylabel('Thrust Magnitude (-)')
            
            # Mass profile
            plt.subplot(3,1,3)
            plt.plot(t_arr, m, '*-b')
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5')
            plt.grid(which='minor', linestyle=':', linewidth='0.5')
            plt.xlabel('Time (sec)')
            plt.ylabel('Mass (kg))')
            
            if self.mc == 0:
                titlestr = "Objective: Minimum-Fuel    mf: %.2f kg"%(m[-1])
            else:
                titlestr = "Objective: Minimum-Control    mf: %.2f kg"%(m[-1])
                
            plt.suptitle(titlestr)
            # plt.tight_layout()
            plt.show()
        
        # Return everything
        return OBJVAL + CONSTR_EQ + CONSTR_INEQ

    def gradient(self, x):
        
        # Constants
        nps = self.npts
        npf = float(nps)
        em  = self.tmax
        m0 = self.mass0
        g0 = self.g0
        isp = self.isp
        lg = self.lg
        t = x[(8*nps)]
        
        # Decision vector information
        X       = x[(0*nps)     :(1*nps)    ]
        Vx      = x[(1*nps)     :(2*nps)    ]
        Ux      = x[(2*nps)     :(3*nps)    ]
        Y       = x[(3*nps)     :(4*nps)    ]
        Vy      = x[(4*nps)     :(5*nps)    ]
        Uy      = x[(5*nps)     :(6*nps)    ]
        m       = x[(6*nps)     :(7*nps)    ]
        Eta     = x[(7*nps)     :(8*nps)    ]
        
        # Calculate dt
        dt = t/npf
        
        # Set array shape and zero out gradient vector
        arr_shape = (10+5*(nps-1)+3*(nps), 1+8*(nps))
        grad = zeros(arr_shape)
        
        # Gradient estimator compare
        # arr = pg.estimate_gradient_h(lambda x: self.fitness(x), x, 1e-3)
        # arr_r = arr.reshape(arr_shape)
        
        # Cost function 
        if self.mc == 0:
            dJdEta = 0.5*dt*ones((nps))
            dJdEta[1:-1] = [2.0*a for a in dJdEta[1:-1]]
            J = sum( [ 0.5*dt*( Eta[ii+1] + Eta[ii] ) for ii in range(0, nps-1) ] )
            dJdt = J/t
            dJdm = zeros((nps,))
        else:
            J = sum( [ 0.5*dt*( (em*Eta[ii+1]/m[ii+1])**2.0 + (em*Eta[ii]/m[ii])**2.0 ) for ii in range(0, nps-1) ] )
            dJdt = J/t
            dJdEta = [ dt*em**2.0/(m[ii]**2.0) * Eta[ii] for ii in range(0,nps) ]
            dJdEta[1:-1] = [2.0*a for a in dJdEta[1:-1]]
            dJdm = [ -dt*em**2.0*Eta[ii]**2.0 / m[ii]**3.0 for ii in range(0,nps)]
            dJdm[1:-1] = [2.0*a for a in dJdm[1:-1]]
            
        grad[0,6*nps:7*nps] = dJdm
        grad[0,7*nps:8*nps] = dJdEta
        grad[0,8*nps] = dJdt
        
        # Initial/Final state constraints
        grad[1,0] = 1.0
        grad[2,3*nps] = 1.0
        grad[3,1*nps] = 1.0
        grad[4,4*nps] = 1.0
        grad[5,6*nps] = 1.0
        grad[6,1*nps-1] = 1.0
        grad[7,4*nps-1] = 1.0
        grad[8,2*nps-1] = 1.0
        grad[9,5*nps-1] = 1.0
        # print(norm(grad[0:9,:]-arr_r[0:9,:]))
        
        # Path constraints
        dXp_dX = -1.0*eye(nps-1, M = nps, k = 0) + eye(nps-1, M = nps, k = 1)
        dXp_dVx = -0.5*dt*(eye(nps-1, M = nps, k = 0) + eye(nps-1, M = nps, k = 1))
        dXp_dt = -0.5/npf*array([ (Vx[ii+1]+Vx[ii]) for ii in range(0,nps-1) ])
        grad[10:(10+nps-1),0:nps] = dXp_dX        
        grad[10:(10+nps-1),nps:2*nps] = dXp_dVx 
        grad[10:(10+nps-1),8*nps] = dXp_dt
        # print(norm(grad[10:(10+nps-1),:]-arr_r[10:(10+nps-1),:]))
        
        dYp_dY = -1.0*eye(nps-1, M = nps, k = 0) + eye(nps-1, M = nps, k = 1) 
        dYp_dVy = -0.5*dt*(eye(nps-1, M = nps, k = 0) + eye(nps-1, M = nps, k = 1))
        dYp_dt = -0.5/npf*array([ (Vy[ii+1]+Vy[ii]) for ii in range(0,nps-1) ])
        grad[(10+(nps-1)):(10+2*(nps-1)),3*nps:4*nps] = dYp_dY        
        grad[(10+(nps-1)):(10+2*(nps-1)),4*nps:5*nps] = dYp_dVy 
        grad[(10+(nps-1)):(10+2*(nps-1)),8*nps] = dYp_dt
        # print(norm(grad[(10+(nps-1)):(10+2*(nps-1)),:]-arr_r[(10+(nps-1)):(10+2*(nps-1)),:]))
        
        dVxp_dVx = -1.0*eye(nps-1, M = nps, k = 0) + eye(nps-1, M = nps, k = 1) 
        dVxp_dUx = zeros((nps-1,nps))
        for ii in range(0,nps-1):
            dVxp_dUx[ii,ii] = -(em*dt*Eta[ii]/2)*(1/m[ii])
            dVxp_dUx[ii,ii+1] = -(em*dt*Eta[ii+1]/2)*(1/m[ii+1])
        dVxp_dm = zeros((nps-1,nps))
        for ii in range(0,nps-1):
            dVxp_dm[ii,ii] = 0.5*dt*em*Eta[ii]*Ux[ii]/m[ii]**2.0
            dVxp_dm[ii,ii+1] = 0.5*dt*em*Eta[ii+1]*Ux[ii+1]/m[ii+1]**2.0
        dVxp_dEta = zeros((nps-1,nps))
        for ii in range(0,nps-1):
            dVxp_dEta[ii,ii] = -0.5*dt*em*Ux[ii]/m[ii]
            dVxp_dEta[ii,ii+1] = -0.5*dt*em*Ux[ii+1]/m[ii+1]
        dVxp_dt = [-0.5/npf*em*(Eta[ii+1]*Ux[ii+1]/m[ii+1] + Eta[ii]*Ux[ii]/m[ii]) for ii in range(0,nps-1)]
        grad[(10+2*(nps-1)):(10+3*(nps-1)),1*nps:2*nps] = dVxp_dVx        
        grad[(10+2*(nps-1)):(10+3*(nps-1)),2*nps:3*nps] = dVxp_dUx 
        grad[(10+2*(nps-1)):(10+3*(nps-1)),6*nps:7*nps] = dVxp_dm 
        grad[(10+2*(nps-1)):(10+3*(nps-1)),7*nps:8*nps] = dVxp_dEta 
        grad[(10+2*(nps-1)):(10+3*(nps-1)),8*nps] = dVxp_dt
        # print(norm(grad[(10+2*(nps-1)):(10+3*(nps-1)),:]-arr_r[(10+2*(nps-1)):(10+3*(nps-1)),:]))
        
        dVyp_dVy = -1.0*eye(nps-1, M = nps, k = 0) + eye(nps-1, M = nps, k = 1) 
        dVyp_dUy = zeros((nps-1,nps))
        for ii in range(0,nps-1):
            dVyp_dUy[ii,ii] = -(em*dt*Eta[ii]/2)*(1/m[ii])
            dVyp_dUy[ii,ii+1] = -(em*dt*Eta[ii+1]/2)*(1/m[ii+1])
        dVyp_dm = zeros((nps-1,nps))
        for ii in range(0,nps-1):
            dVyp_dm[ii,ii] = 0.5*dt*em*Eta[ii]*Uy[ii]/m[ii]**2.0
            dVyp_dm[ii,ii+1] = 0.5*dt*em*Eta[ii+1]*Uy[ii+1]/m[ii+1]**2.0
        dVyp_dEta = zeros((nps-1,nps))
        for ii in range(0,nps-1):
            dVyp_dEta[ii,ii] = -0.5*dt*em*Uy[ii]/m[ii]
            dVyp_dEta[ii,ii+1] = -0.5*dt*em*Uy[ii+1]/m[ii+1]
        dVyp_dt = [-0.5/npf*(em*Eta[ii+1]*Uy[ii+1]/m[ii+1] + em*Eta[ii]*Uy[ii]/m[ii] + 2*lg[1]) for ii in range(0,nps-1)]
        grad[(10+3*(nps-1)):(10+4*(nps-1)),4*nps:5*nps] = dVyp_dVy        
        grad[(10+3*(nps-1)):(10+4*(nps-1)),5*nps:6*nps] = dVyp_dUy 
        grad[(10+3*(nps-1)):(10+4*(nps-1)),6*nps:7*nps] = dVyp_dm 
        grad[(10+3*(nps-1)):(10+4*(nps-1)),7*nps:8*nps] = dVyp_dEta 
        grad[(10+3*(nps-1)):(10+4*(nps-1)),8*nps] = dVyp_dt
        # print(norm(grad[(10+3*(nps-1)):(10+4*(nps-1)),:]-arr_r[(10+3*(nps-1)):(10+4*(nps-1)),:]))
        
        dmp_dm = eye(nps-1, M = nps, k = 0) - 1.0*eye(nps-1, M = nps, k = 1) 
        dmp_dEta = -0.5*dt*em/(g0*isp)*(eye(nps-1, M = nps, k = 0) + eye(nps-1, M = nps, k = 1)) 
        dmp_dt = [ -0.5/npf*em/(g0*isp)*(Eta[ii+1] + Eta[ii])  for ii in range(0,nps-1)]
        grad[(10+4*(nps-1)):(10+5*(nps-1)),6*nps:7*nps] = dmp_dm
        grad[(10+4*(nps-1)):(10+5*(nps-1)),7*nps:8*nps] = dmp_dEta
        grad[(10+4*(nps-1)):(10+5*(nps-1)),8*nps] = dmp_dt
        # print(norm(grad[(10+4*(nps-1)):(10+5*(nps-1)),:]-arr_r[(10+4*(nps-1)):(10+5*(nps-1)),:]))
        
        # Constraint for thrust unit vector equalling zero exactly
        dUmag_dUx = 2.0*diag(Ux)
        dUmag_dUy = 2.0*diag(Uy)
        grad[(10+5*(nps-1)):(10+5*(nps-1)+nps),2*nps:3*nps] = dUmag_dUx
        grad[(10+5*(nps-1)):(10+5*(nps-1)+nps),5*nps:6*nps] = dUmag_dUy
        # print(norm(grad[(10+5*(nps-1)):(10+5*(nps-1)+nps),:]-arr_r[(10+5*(nps-1)):(10+5*(nps-1)+nps),:]))
        
        # Thrust minimums and maximums
        dEtaMax_dEta = 1.0*eye(nps)
        dEtaMin_dEta = -1.0*eye(nps)
        grad[(10+5*(nps-1)+nps):(10+5*(nps-1)+2*nps),7*nps:8*nps] = dEtaMax_dEta
        grad[(10+5*(nps-1)+2*nps):(10+5*(nps-1)+3*nps),7*nps:8*nps] = dEtaMin_dEta
        
        # idx = unravel_index(argmin(abs(grad[nonzero(grad)])),grad.shape)
        # print(idx, ", ", min(abs(grad[nonzero(grad)])))
        # print(grad[idx[1],idx[0]])
        # print(min(abs(grad[nonzero(grad)])))
        # print(where( grad==min(grad[nonzero(grad)])))
        
        grad = grad.reshape((arr_shape[0]*arr_shape[1],))
          
        return grad


def run_problem4(npts=30,tof=550.0,X_initial=[ -370000, 15000.0, 1500, 0.0 ],X_target = [0.0, 200.0, 0, -10.0],mass0=20000,tmax=1.3*66000,isp=300):
    """
    Solves the minimum control problem of a 2-D lander under a 
    uniform gravity field. Employs a trapezoidal collocation method
    """
    
    udp = prob_2D_lander_mass(npts=npts,tof=tof,X_initial=X_initial,X_target=X_target,mass0=mass0,tmax=tmax,isp=isp,min_control=0)
    prob = pg.problem(udp)
    prob.c_tol = [1e-1]*9 + [1e-1]*5*(npts-1) + [1e-2]*3*npts
    
    uda = pg.algorithm(pg.nlopt('slsqp'))
    uda.set_verbosity(50)
    uda.extract(pg.nlopt).xtol_rel = 0
    uda.extract(pg.nlopt).ftol_rel = 0
    uda.extract(pg.nlopt).maxeval = 2000
    uda2 = pg.mbh(uda, stop=2, perturb=0.1)
    algo = pg.algorithm(uda2)
    
    # Uncomment this for a good initial guess.
    dt = tof/npts
    X_g = list(linspace(X_initial[0], X_target[0], npts))
    Vx_g = list(linspace(X_initial[2], X_target[2], npts))
    Ux_g = [-1.0]*npts
    Y_g = list(linspace(X_initial[1], X_target[1], npts))
    Vy_g = list(linspace(X_initial[3], X_target[3], npts))
    Uy_g = [0.0]*npts
    m = list(linspace(mass0,0.6*mass0,npts))
    T = [0.2]*5 + [0.8]*25
    # T = list(linspace(0.8,0.2,npts))
    
    X0 = X_g + Vx_g + Ux_g + Y_g + Vy_g + Uy_g + m + T + [tof]
    
    # Create population of 1.
    pop = pg.population(prob)
    pop.push_back(X0)
    
    # Evolve
    t1 = time.clock()
    pop = uda.evolve(pop)
    t2 = time.clock()
    print("Solved time: %.8f"%(t2-t1))
    print("TOF: %.3f\n"%(pop.champion_x[-1]))
    print("Final Mass: %.5f\n"%(pop.champion_x[7*npts-1]))
    
    is_feas = prob.feasibility_x(pop.champion_x)
    if is_feas:
        print("===================")
        print("FEASIBLE TRAJECTORY")
        print("===================")
        
    udp.summary(pop.champion_x)
        


