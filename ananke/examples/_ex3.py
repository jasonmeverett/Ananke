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
from mpl_toolkits.mplot3d import Axes3D

class prob_3D_lander(object):
    
    # Function initialization
    def __init__(self,config):
        self.npts = config['npts']
        self.tof = config['tof']
        self.r0_I = config['r0_I']
        self.v0_I = config['v0_I']
        self.rt_I = config['rt_I']
        self.vt_I = config['vt_I']
        self.g0 = 9.81
        self.isp = config['isp']
        self.Tmax = config['Tmax']
        self.mass0 = config['mass0']
        self.Eta_lb = config['Eta_lb']
        self.Eta_ub = config['Eta_ub']
        self.mu = config['mu']
        self.R_eq = config['R_eq']
        
        return
    
    def get_nic(self):
        return 2*self.npts

    def get_nec(self):
        return 13 + 7*(self.npts-1) + self.npts
    
    # TODO: Set these bounds in a smarter fashion.
    def get_bounds(self):
        
        sf_r = 1.0
        sf_v = 2.0
        r_lb = [-sf_r*norm(self.r0_I)]*3*self.npts
        r_ub = [ sf_r*norm(self.r0_I)]*3*self.npts
        v_lb = [-sf_v*norm(self.v0_I)]*3*self.npts
        v_ub = [ sf_v*norm(self.v0_I)]*3*self.npts
        u_lb = [-5]*3*self.npts
        u_ub = [ 5]*3*self.npts
        m_lb = [0.1*self.mass0]*self.npts
        m_ub = [1.1*self.mass0]*self.npts
        Eta_lb = [0.0]*self.npts
        Eta_ub = [1.1]*self.npts
        T_lb = [300]
        T_ub = [800]
        
        LB = r_lb + v_lb + u_lb + m_lb + Eta_lb + T_lb
        UB = r_ub + v_ub + u_ub + m_ub + Eta_ub + T_ub
        
        return (LB, UB)
    
    def fitness(self, x):
        return self.run_traj(x, plot_traj=0)
    
    def summary(self, x):
        return self.run_traj(x, plot_traj=1)
    
    def gradient(self, x):
        
        n = self.npts
        nf = float(n)
        Tm = self.Tmax
        mu = self.mu
        isp = self.isp
        g0 = self.g0
        Eta_lb = self.Eta_lb
        Eta_ub = self.Eta_ub
        
        # Extract state information
        Rx      = x[(0*n)  :(1*n) ]
        Ry      = x[(1*n)  :(2*n) ]
        Rz      = x[(2*n)  :(3*n) ]
        Vx      = x[(3*n)  :(4*n) ]
        Vy      = x[(4*n)  :(5*n) ]
        Vz      = x[(5*n)  :(6*n) ]
        Ux      = x[(6*n)  :(7*n) ]
        Uy      = x[(7*n)  :(8*n) ]
        Uz      = x[(8*n)  :(9*n) ]
        m       = x[(9*n)  :(10*n)]
        Eta     = x[(10*n) :(11*n)]
        tof     = x[11*n]

        # Collocation timestep
        dt = tof/nf
        
        # Zero gradient
        arr_shape = (1 + 13 + 7*(n-1) + n + 2*n, len(x))
        grad = zeros(arr_shape)
        
        # Gradient estimation
        # grade = pg.estimate_gradient(lambda x: self.fitness(x), x, 1e-5)
        # arre = grade.reshape(arr_shape)
        
        # Matrices to use later
        dbeyeneg = -1.0*eye(n-1, M = n, k = 0) + eye(n-1, M = n, k = 1)
        dbeyepos =  1.0*eye(n-1, M = n, k = 0) + eye(n-1, M = n, k = 1)
        
        # Cost function partials
        J = sum([ 0.5*dt*( (Tm*Eta[k]/m[k])**2.0 + (Tm*Eta[k+1]/m[k+1])**2.0 ) for k in range(0,n-1) ])
        dJ_dm = [-dt*Tm**2.0*Eta[k]**2.0/m[k]**3.0 for k in range(0,n)]
        dJ_dm[1:-1] = [2*a for a in dJ_dm[1:-1]]
        dJ_dEta = [dt*Tm**2.0/m[k]**2.0*Eta[k] for k in range(0,n)]
        dJ_dEta[1:-1] = [2*a for a in dJ_dEta[1:-1]]
        dJ_dT = J/tof
        grad[0,( 9*n):(10*n)] = dJ_dm
        grad[0,(10*n):(11*n)] = dJ_dEta 
        grad[0,(11*n)] = dJ_dT

        # Initial/Final state constraints
        grad[1,0*n] = 1.0
        grad[2,1*n] = 1.0
        grad[3,2*n] = 1.0
        grad[4,3*n] = 1.0
        grad[5,4*n] = 1.0
        grad[6,5*n] = 1.0
        grad[7,9*n] = 1.0
        grad[ 8,1*n-1] = 1.0
        grad[ 9,2*n-1] = 1.0
        grad[10,3*n-1] = 1.0
        grad[11,4*n-1] = 1.0
        grad[12,5*n-1] = 1.0
        grad[13,6*n-1] = 1.0

        # Path equality constraints - Position
        dRxp_dRx = dbeyeneg
        dRyp_dRy = dbeyeneg
        dRzp_dRz = dbeyeneg
        dRxp_dVx = -0.5*dt*dbeyepos
        dRyp_dVy = -0.5*dt*dbeyepos
        dRzp_dVz = -0.5*dt*dbeyepos
        dRxp_dT = [-0.5/nf*(Vx[k] + Vx[k+1]) for k in range(0,n-1)]
        dRyp_dT = [-0.5/nf*(Vy[k] + Vy[k+1]) for k in range(0,n-1)]
        dRzp_dT = [-0.5/nf*(Vz[k] + Vz[k+1]) for k in range(0,n-1)]
        grad[(14+0*(n-1)):(14+1*(n-1)),(0*n):(1*n)] = dRxp_dRx
        grad[(14+1*(n-1)):(14+2*(n-1)),(1*n):(2*n)] = dRyp_dRy
        grad[(14+2*(n-1)):(14+3*(n-1)),(2*n):(3*n)] = dRzp_dRz
        grad[(14+0*(n-1)):(14+1*(n-1)),(3*n):(4*n)] = dRxp_dVx
        grad[(14+1*(n-1)):(14+2*(n-1)),(4*n):(5*n)] = dRyp_dVy
        grad[(14+2*(n-1)):(14+3*(n-1)),(5*n):(6*n)] = dRzp_dVz
        grad[(14+0*(n-1)):(14+1*(n-1)),(11*n)] = dRxp_dT
        grad[(14+1*(n-1)):(14+2*(n-1)),(11*n)] = dRyp_dT
        grad[(14+2*(n-1)):(14+3*(n-1)),(11*n)] = dRzp_dT
        
        # Path equality constraints - Velocity
        dVxp_dVx = dbeyeneg
        dVyp_dVy = dbeyeneg
        dVzp_dVz = dbeyeneg
        dVxp_dUx = zeros((n-1,n))
        dVyp_dUy = zeros((n-1,n))
        dVzp_dUz = zeros((n-1,n))
        dVxp_dm = zeros((n-1,n))
        dVyp_dm = zeros((n-1,n))
        dVzp_dm = zeros((n-1,n))
        dVxp_dEta = zeros((n-1,n))
        dVyp_dEta = zeros((n-1,n))
        dVzp_dEta = zeros((n-1,n))
        dVxp_dT = zeros((n-1,))
        dVyp_dT = zeros((n-1,))
        dVzp_dT = zeros((n-1,))
        dVxp_dRx = zeros((n-1,n))
        dVxp_dRy = zeros((n-1,n))
        dVxp_dRz = zeros((n-1,n))
        dVyp_dRx = zeros((n-1,n))
        dVyp_dRy = zeros((n-1,n))
        dVyp_dRz = zeros((n-1,n))
        dVzp_dRx = zeros((n-1,n))
        dVzp_dRy = zeros((n-1,n))
        dVzp_dRz = zeros((n-1,n))
        for k in range(0, n-1):
            dVxp_dUx[k,k] = -0.5*dt*Tm*Eta[k]/m[k]
            dVyp_dUy[k,k] = -0.5*dt*Tm*Eta[k]/m[k]
            dVzp_dUz[k,k] = -0.5*dt*Tm*Eta[k]/m[k]
            dVxp_dUx[k,k+1] = -0.5*dt*Tm*Eta[k+1]/m[k+1]
            dVyp_dUy[k,k+1] = -0.5*dt*Tm*Eta[k+1]/m[k+1]
            dVzp_dUz[k,k+1] = -0.5*dt*Tm*Eta[k+1]/m[k+1]
            dVxp_dm[k,k] = 0.5*dt*Tm*Eta[k]*Ux[k]/m[k]**2.0
            dVyp_dm[k,k] = 0.5*dt*Tm*Eta[k]*Uy[k]/m[k]**2.0
            dVzp_dm[k,k] = 0.5*dt*Tm*Eta[k]*Uz[k]/m[k]**2.0
            dVxp_dm[k,k+1] = 0.5*dt*Tm*Eta[k+1]*Ux[k+1]/m[k+1]**2.0
            dVyp_dm[k,k+1] = 0.5*dt*Tm*Eta[k+1]*Uy[k+1]/m[k+1]**2.0
            dVzp_dm[k,k+1] = 0.5*dt*Tm*Eta[k+1]*Uz[k+1]/m[k+1]**2.0
            dVxp_dEta[k,k] = -0.5*dt*Tm*Ux[k]/m[k]
            dVyp_dEta[k,k] = -0.5*dt*Tm*Uy[k]/m[k]
            dVzp_dEta[k,k] = -0.5*dt*Tm*Uz[k]/m[k]
            dVxp_dEta[k,k+1] = -0.5*dt*Tm*Ux[k+1]/m[k+1]
            dVyp_dEta[k,k+1] = -0.5*dt*Tm*Uy[k+1]/m[k+1]
            dVzp_dEta[k,k+1] = -0.5*dt*Tm*Uz[k+1]/m[k+1]
            Rk = (Rx[k]**2.0 + Ry[k]**2.0 + Rz[k]**2.0)**0.5
            Rkp1 = (Rx[k+1]**2.0 + Ry[k+1]**2.0 + Rz[k+1]**2.0)**0.5
            dVxp_dT[k] = - 0.5/nf*( (Tm*Eta[k]*Ux[k]/m[k]) + (Tm*Eta[k+1]*Ux[k+1]/m[k+1]) + (-mu/Rk**3.0)*Rx[k] + (-mu/Rkp1**3.0)*Rx[k+1] )
            dVyp_dT[k] = - 0.5/nf*( (Tm*Eta[k]*Uy[k]/m[k]) + (Tm*Eta[k+1]*Uy[k+1]/m[k+1]) + (-mu/Rk**3.0)*Ry[k] + (-mu/Rkp1**3.0)*Ry[k+1] )
            dVzp_dT[k] = - 0.5/nf*( (Tm*Eta[k]*Uz[k]/m[k]) + (Tm*Eta[k+1]*Uz[k+1]/m[k+1]) + (-mu/Rk**3.0)*Rz[k] + (-mu/Rkp1**3.0)*Rz[k+1] )
            muoRk5 = mu/Rk**5.0
            muoRk3 = mu/Rk**3.0
            dVxp_dRx[k,k] = 3*muoRk5*Rx[k]*Rx[k] - muoRk3
            dVxp_dRy[k,k] = 3*muoRk5*Rx[k]*Ry[k]
            dVxp_dRz[k,k] = 3*muoRk5*Rx[k]*Rz[k]
            dVyp_dRx[k,k] = 3*muoRk5*Ry[k]*Rx[k]
            dVyp_dRy[k,k] = 3*muoRk5*Ry[k]*Ry[k] - muoRk3
            dVyp_dRz[k,k] = 3*muoRk5*Ry[k]*Rz[k]
            dVzp_dRx[k,k] = 3*muoRk5*Rz[k]*Rx[k]
            dVzp_dRy[k,k] = 3*muoRk5*Rz[k]*Ry[k]
            dVzp_dRz[k,k] = 3*muoRk5*Rz[k]*Rz[k] - muoRk3
            muoRkp15 = mu/Rk**5.0
            muoRkp13 = mu/Rk**3.0
            dVxp_dRx[k,k+1] = 3*muoRkp15*Rx[k+1]*Rx[k+1] - muoRkp13
            dVxp_dRy[k,k+1] = 3*muoRkp15*Rx[k+1]*Ry[k+1]
            dVxp_dRz[k,k+1] = 3*muoRkp15*Rx[k+1]*Rz[k+1]
            dVyp_dRx[k,k+1] = 3*muoRkp15*Ry[k+1]*Rx[k+1]
            dVyp_dRy[k,k+1] = 3*muoRkp15*Ry[k+1]*Ry[k+1] - muoRkp13
            dVyp_dRz[k,k+1] = 3*muoRkp15*Ry[k+1]*Rz[k+1]
            dVzp_dRx[k,k+1] = 3*muoRkp15*Rz[k+1]*Rx[k+1]
            dVzp_dRy[k,k+1] = 3*muoRkp15*Rz[k+1]*Ry[k+1]
            dVzp_dRz[k,k+1] = 3*muoRkp15*Rz[k+1]*Rz[k+1] - muoRkp13
        
        grad[(14+3*(n-1)):(14+4*(n-1)),(0*n):(1*n)] = dVxp_dRx
        grad[(14+3*(n-1)):(14+4*(n-1)),(1*n):(2*n)] = dVxp_dRy
        grad[(14+3*(n-1)):(14+4*(n-1)),(2*n):(3*n)] = dVxp_dRz
        grad[(14+4*(n-1)):(14+5*(n-1)),(0*n):(1*n)] = dVyp_dRx
        grad[(14+4*(n-1)):(14+5*(n-1)),(1*n):(2*n)] = dVyp_dRy
        grad[(14+4*(n-1)):(14+5*(n-1)),(2*n):(3*n)] = dVyp_dRz
        grad[(14+5*(n-1)):(14+6*(n-1)),(0*n):(1*n)] = dVzp_dRx
        grad[(14+5*(n-1)):(14+6*(n-1)),(1*n):(2*n)] = dVzp_dRy
        grad[(14+5*(n-1)):(14+6*(n-1)),(2*n):(3*n)] = dVzp_dRz
        grad[(14+3*(n-1)):(14+4*(n-1)),(3*n):(4*n)] = dVxp_dVx
        grad[(14+4*(n-1)):(14+5*(n-1)),(4*n):(5*n)] = dVyp_dVy
        grad[(14+5*(n-1)):(14+6*(n-1)),(5*n):(6*n)] = dVzp_dVz
        grad[(14+3*(n-1)):(14+4*(n-1)),(6*n):(7*n)] = dVxp_dUx
        grad[(14+4*(n-1)):(14+5*(n-1)),(7*n):(8*n)] = dVyp_dUy
        grad[(14+5*(n-1)):(14+6*(n-1)),(8*n):(9*n)] = dVzp_dUz
        grad[(14+3*(n-1)):(14+4*(n-1)),( 9*n):(10*n)] = dVxp_dm
        grad[(14+4*(n-1)):(14+5*(n-1)),( 9*n):(10*n)] = dVyp_dm
        grad[(14+5*(n-1)):(14+6*(n-1)),( 9*n):(10*n)] = dVzp_dm
        grad[(14+3*(n-1)):(14+4*(n-1)),(10*n):(11*n)] = dVxp_dEta
        grad[(14+4*(n-1)):(14+5*(n-1)),(10*n):(11*n)] = dVyp_dEta
        grad[(14+5*(n-1)):(14+6*(n-1)),(10*n):(11*n)] = dVzp_dEta
        grad[(14+3*(n-1)):(14+4*(n-1)),(11*n)] = dVxp_dT
        grad[(14+4*(n-1)):(14+5*(n-1)),(11*n)] = dVyp_dT
        grad[(14+5*(n-1)):(14+6*(n-1)),(11*n)] = dVzp_dT
        
        # Mass path partials
        dmp_dm = -dbeyeneg
        dmp_dEta = -0.5*dt*Tm/(g0*isp)*dbeyepos
        dmp_dT = [-0.5/nf*Tm/(g0*isp)*(Eta[k]+Eta[k+1]) for k in range(0,n-1)]
        grad[(14+6*(n-1)):(14+7*(n-1)),( 9*n):(10*n)] = dmp_dm
        grad[(14+6*(n-1)):(14+7*(n-1)),(10*n):(11*n)] = dmp_dEta
        grad[(14+6*(n-1)):(14+7*(n-1)),(11*n)] = dmp_dT
        
        # Thrust unit vector magnitudes
        dUm_dUx = 2*diag(Ux)
        dUm_dUy = 2*diag(Uy)
        dUm_dUz = 2*diag(Uz)
        grad[(14+7*(n-1)):(14+7*(n-1)+1*n),(6*n):(7*n)] = dUm_dUx
        grad[(14+7*(n-1)):(14+7*(n-1)+1*n),(7*n):(8*n)] = dUm_dUy
        grad[(14+7*(n-1)):(14+7*(n-1)+1*n),(8*n):(9*n)] = dUm_dUz
        
        # Inequality thrust magnitude constraints
        dElb_dEta = -eye(n)
        dEub_dEta = eye(n)
        grad[(14+7*(n-1)+1*n):(14+7*(n-1)+2*n),(10*n):(11*n)] = dElb_dEta
        grad[(14+7*(n-1)+2*n):(14+7*(n-1)+3*n),(10*n):(11*n)] = dEub_dEta
        
        # a1 = grad
        # a2 = arre
        # print(norm(a1-a2))
        # quit()
        
        grad = grad.reshape((arr_shape[0]*arr_shape[1],))
        return grad
    
    def run_traj(self, x, plot_traj=0):
        
        n = self.npts
        nf = float(n)
        Tm = self.Tmax
        mu = self.mu
        isp = self.isp
        g0 = self.g0
        Eta_lb = self.Eta_lb
        Eta_ub = self.Eta_ub
        
        # Extract state information
        Rx      = x[(0*n)  :(1*n) ]
        Ry      = x[(1*n)  :(2*n) ]
        Rz      = x[(2*n)  :(3*n) ]
        Vx      = x[(3*n)  :(4*n) ]
        Vy      = x[(4*n)  :(5*n) ]
        Vz      = x[(5*n)  :(6*n) ]
        Ux      = x[(6*n)  :(7*n) ]
        Uy      = x[(7*n)  :(8*n) ]
        Uz      = x[(8*n)  :(9*n) ]
        m       = x[(9*n)  :(10*n)]
        Eta     = x[(10*n) :(11*n)]
        tof     = x[11*n]

        # Collocation timestep
        dt = tof/nf

        # Starting and ending constraints
        r0 = array([Rx[0],Ry[0],Rz[0]])
        v0 = array([Vx[0],Vy[0],Vz[0]])
        rf = array([Rx[-1],Ry[-1],Rz[-1]])
        vf = array([Vx[-1],Vy[-1],Vz[-1]])
        CONSTR_EQ = \
            list(r0-self.r0_I) + \
            list(v0-self.v0_I) + \
            [m[0] - self.mass0] + \
            list(rf-self.rt_I) + \
            list(vf-self.vt_I)
            
        # Equality path constraints
        Rxp = [(Rx[k+1]-Rx[k]) - 0.5*dt*(Vx[k]+Vx[k+1]) for k in range(0,n-1)]
        Ryp = [(Ry[k+1]-Ry[k]) - 0.5*dt*(Vy[k]+Vy[k+1]) for k in range(0,n-1)]
        Rzp = [(Rz[k+1]-Rz[k]) - 0.5*dt*(Vz[k]+Vz[k+1]) for k in range(0,n-1)]
        Vxp = []
        Vyp = []
        Vzp = []
        for k in range(0,n-1):
            Rk = (Rx[k]**2.0 + Ry[k]**2.0 + Rz[k]**2.0)**0.5
            Rkp1 = (Rx[k+1]**2.0 + Ry[k+1]**2.0 + Rz[k+1]**2.0)**0.5
            Vxp = Vxp + [ (Vx[k+1]-Vx[k]) - 0.5*dt*( (Tm*Eta[k]*Ux[k]/m[k]) + (Tm*Eta[k+1]*Ux[k+1]/m[k+1]) + (-mu/Rk**3.0)*Rx[k] + (-mu/Rkp1**3.0)*Rx[k+1] ) ]
            Vyp = Vyp + [ (Vy[k+1]-Vy[k]) - 0.5*dt*( (Tm*Eta[k]*Uy[k]/m[k]) + (Tm*Eta[k+1]*Uy[k+1]/m[k+1]) + (-mu/Rk**3.0)*Ry[k] + (-mu/Rkp1**3.0)*Ry[k+1] ) ]
            Vzp = Vzp + [ (Vz[k+1]-Vz[k]) - 0.5*dt*( (Tm*Eta[k]*Uz[k]/m[k]) + (Tm*Eta[k+1]*Uz[k+1]/m[k+1]) + (-mu/Rk**3.0)*Rz[k] + (-mu/Rkp1**3.0)*Rz[k+1] ) ]
        mp = [ (m[k] - m[k+1]) - 0.5*dt*Tm/(g0*isp)*( Eta[k] + Eta[k+1] ) for k in range(0,n-1) ]
        Um = [ Ux[k]**2.0 + Uy[k]**2.0 + Uz[k]**2.0 - 1.0 for k in range(0,n) ]
        CONSTR_EQ = CONSTR_EQ + Rxp + Ryp + Rzp + Vxp + Vyp + Vzp + mp + Um

        # Inequality throttle constraints
        Eta_bound_0 = [Eta_lb - Eta[k] for k in range(0,n)]
        Eta_bound_f = [Eta[k] - Eta_ub for k in range(0,n)]
        CONSTR_INEQ = Eta_bound_0 + Eta_bound_f
        
        # Other objective - minimize control effort
        OBJVAL = [ sum([ 0.5*dt*( (Tm*Eta[k]/m[k])**2.0 + (Tm*Eta[k+1]/m[k+1])**2.0 ) for k in range(0,n-1) ]) ]

        # Plot, if enabled
        if plot_traj == 1:
            t_arr = linspace(0,tof,n)
            
            fig = plt.figure(1)
            ax = fig.gca(projection='3d')
            plt.title('Position (Landing Site)')
            
            # Landing site position
            Rx_LS = Rx - self.rt_I[0]
            Ry_LS = Ry - self.rt_I[1]
            Rz_LS = Rz - self.rt_I[2]
            ax.plot(Ry_LS, Rz_LS, Rx_LS,'*-b')
            
            # Rx_LS = Rx
            # Ry_LS = Ry
            # Rz_LS = Rz
            # ax.plot(Rx, Ry, Rz,'*-b')
            
            for ii in range(0,n):
                fac = 30000
                Xs = [Rx[ii]- self.rt_I[0], Rx[ii]- self.rt_I[0] + fac*Ux[ii]*Eta[ii]]
                Ys = [Ry[ii]- self.rt_I[1], Ry[ii]- self.rt_I[1] + fac*Uy[ii]*Eta[ii]]
                Zs = [Rz[ii]- self.rt_I[2], Rz[ii]- self.rt_I[2] + fac*Uz[ii]*Eta[ii]]
                ax.plot(Ys,Zs,Xs,'r')
                
                # fac = 80000
                # Xs = [Rx[ii], Rx[ii] + fac*Ux[ii]*Eta[ii]]
                # Ys = [Ry[ii], Ry[ii] + fac*Uy[ii]*Eta[ii]]
                # Zs = [Rz[ii], Rz[ii] + fac*Uz[ii]*Eta[ii]]
                # ax.plot(Xs,Ys,Zs,'r')
            
            plt.figure(2)
            plt.plot(t_arr,Eta,'*-b')
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5')
            plt.grid(which='minor', linestyle=':', linewidth='0.5')
            plt.xlabel('Time (sec)')
            plt.ylabel('Throttle (-)')
            plt.title('Throttle')
            
            alt = [norm([Rx[k], Ry[k], Rz[k]]) - self.R_eq for k in range(0,n) ]
            plt.figure(3)
            plt.plot(t_arr, alt,'*-b')
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5')
            plt.grid(which='minor', linestyle=':', linewidth='0.5')
            plt.xlabel('Time (sec)')
            plt.ylabel('Alt (km)')
            plt.title('Altitude')
            
            print("FINAL MASS: %.3F KG"%(m[-1]))
            
            plt.show()

            # Altitude
            

        # Return everything
        return OBJVAL + CONSTR_EQ + CONSTR_INEQ


# TODO: Add omega x r terms for rotating planet
def run_problem3(npts=20,tof=600,mass0=20000,isp=300,Tmax=1.3*66000):
    """
    Solves the minimum control problem of a 2-D lander under a 
    uniform gravity field. Employs a trapezoidal collocation method
    """
    config = {}
    x_optimal_output = 'x_optimal_new.txt'
    from_file = True
    to_file = True

    # Planet definition
    R_eq = 1738e3
    
    # Config structure
    config['npts'] = npts
    config['mu'] = Moon.mu
    config['mass0'] = mass0
    config['isp'] = isp
    config['Tmax'] = Tmax
    config['tof'] = tof
    config['Eta_lb'] = 0.2
    config['Eta_ub'] = 0.8
    config['R_eq'] = R_eq
    
    # Initial state
    ra = R_eq+100000
    rp = R_eq+15000
    sma = 0.5*(ra+rp)
    ecc = (ra-rp)/(ra+rp)
    r,v = elts_to_rv(sma,ecc,d2r(0.0),d2r(0.0),d2r(-15.0),0.0,config['mu'])
    config['r0_I'] = r
    config['v0_I'] = v
    
    # Landing site
    config['LS_lon'] = 0.0
    config['LS_lat'] = 0.0
    config['LS_alt'] = 0.0
    R_I_UEN = DCM_I_UEN(config['LS_lon'], config['LS_lat'], degrees=True)
    config['rLS_I'] = Pos_LS(config['LS_lon'], config['LS_lat'], config['LS_alt'], R_eq, degrees=True)
    config['vLS_I'] = [0,0,0]
    
    # Target position/velocity
    config['rt_LS'] = [200,0,0]
    config['vt_LS'] = [-15,0,0]
    config['rt_I'] = config['rLS_I'] + R_I_UEN.inv().apply(config['rt_LS'])
    config['vt_I'] = R_I_UEN.inv().apply(config['vt_LS'])
    
    # Problem configuration
    udp = prob_3D_lander(config)
    prob = pg.problem(udp)
    prob.c_tol = 1e-3
    
    algo = pg.algorithm(pg.nlopt('slsqp'))
    algo.set_verbosity(100)
    algo.extract(pg.nlopt).xtol_rel = 0
    algo.extract(pg.nlopt).ftol_rel = 0
    algo.extract(pg.nlopt).maxeval = 5000
    
    # Initial guess from file.
    if from_file:
        print('LOADED FROM FILE')
        content = []
        with open(x_optimal_output) as f:
            content = f.readlines() 
        X0 = [float(item) for item in content]
    
    # Initial guess - linear profile
    else:
        print('LINEAR INITIAL GUESS')
        dt = tof/npts
        R = linspace(config['r0_I'], config['rt_I'], npts).transpose()
        V = linspace(config['v0_I'], config['vt_I'], npts).transpose()
        Ux = [1.0] + [(V[0,k+1] - V[0,k])/dt for k in range(0,npts-1)]
        Uy = [1.0] + [(V[1,k+1] - V[1,k])/dt for k in range(0,npts-1)]
        Uz = [1.0] + [(V[2,k+1] - V[2,k])/dt for k in range(0,npts-1)]
        U = array([Ux, Uy, Uz])
        for ii in range(0,npts):
            U[:,ii] = U[:,ii]/norm(U[:,ii])
        m = linspace(mass0, 0.1*mass0, npts)
        Eta = linspace(1.0,1.0, npts)
        X0 = list(R.flatten()) + list(V.flatten()) + list(U.flatten()) + list(m) + list(Eta) + [tof]
        
    # Create a population
    print("Creating population...")
    pop = pg.population(prob)
    pop.push_back(X0)

    # Evolve
    print("Evolving...")
    pop = algo.evolve(pop)

    # Check feasibility
    is_feas = prob.feasibility_x(pop.champion_x)
    if is_feas:
        print("===================")
        print("FEASIBLE TRAJECTORY")
        print("===================")
        
    if to_file:
        print('WRITING TO FILE')
        fout = open(x_optimal_output,'w+')
        champion_vec = pop.champion_x
        for item in champion_vec:
            fout.write("%30.15f\n"%(item))
        fout.close()

    udp.summary(pop.champion_x)


