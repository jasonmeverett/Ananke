#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 09:48:16 2019

MULTI-PHASE 3-Dimensional full gravity optimization problem.
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
import time
import json

class prob_3D_lander(object):
    
    # Function initialization
    def __init__(self,C):
        CP = C['planet']
        CV = C['vehicle']
        CT = C['target']
        CO = C['opt']
        self.objtype = CO['objtype']
        self.npts = CO['config']['npts']
        self.g0 = 9.81
        self.isp = CV['isp']
        self.Tmax = CV['Tmax']
        self.mass0 = CV['mass']
        self.mu = CP['mu']
        self.R_eq = CP['R_eq']
        self.plOm = CP['Omega']
        self.ep0 = CP['Epoch']
        self.LS_lat = CT['lat']
        self.LS_lon = CT['lon']
        self.LS_alt = CT['alt']
        self.rt_LS = CT['target_pos_UEN']
        self.vt_LS = CT['target_vel_UEN']
        self.R_PF_UEN = Rot_PF_UEN(self.LS_lon, self.LS_lat, CT['degrees'])
        self.R_UEN_PF = self.R_PF_UEN.inv()

        # Bounds
        self.bnds_Eta = [CO['bounds']['lo']['eta'], CO['bounds']['hi']['eta']]
        self.bnds_nu = [CO['bounds']['lo']['nu0'], CO['bounds']['hi']['nu0']]
        self.bnds_T = [CO['bounds']['lo']['T'], CO['bounds']['hi']['T']]
        
        # Initial state
        ra = CP['R_eq'] + CV['orbit']['alta']
        rp = CP['R_eq'] + CV['orbit']['altp']
        self.sma = 0.5*(ra+rp)
        self.ecc = (ra-rp)/(ra+rp)
        self.inc = CV['orbit']['inc']
        self.Om = CV['orbit']['raan']
        self.om = CV['orbit']['argper']
        self.nu0 = CV['orbit']['ta']
        self.orb_deg = CV['orbit']['degrees']
        self.DCM_I_Per = Rot_I_Perifocal(self.Om, self.inc, self.om, degrees=True)
        self.o, self.odot = calc_o_odot(self.sma, self.ecc, self.nu0, self.mu)
        self.r0_I,self.v0_I = elts_to_rv(self.sma,self.ecc,self.inc,self.Om,self.om,self.nu0,CP['mu'],CV['orbit']['degrees'])
        
        return
    
    def get_nic(self):
        return 2*self.npts + 1 + 1 + 1 + 1

    def get_nec(self):
        return 13 + 7*(self.npts-1) + self.npts
    
    # TODO: Set these bounds in a smarter fashion.
    def get_bounds(self):
        
        sf_r = 0.3
        sf_v = 2.0
        rx_lb = [self.r0_I[0] - sf_r*norm(self.r0_I)] * self.npts
        ry_lb = [self.r0_I[1] - sf_r*norm(self.r0_I)] * self.npts
        rz_lb = [self.r0_I[2] - sf_r*norm(self.r0_I)] * self.npts
        rx_ub = [self.r0_I[0] + sf_r*norm(self.r0_I)] * self.npts
        ry_ub = [self.r0_I[1] + sf_r*norm(self.r0_I)] * self.npts
        rz_ub = [self.r0_I[2] + sf_r*norm(self.r0_I)] * self.npts
        r_lb = rx_lb + ry_lb + rz_lb
        r_ub = rx_ub + ry_ub + rz_ub
        v_lb = [-sf_v*norm(self.v0_I)]*3*self.npts
        v_ub = [ sf_v*norm(self.v0_I)]*3*self.npts
        u_lb = [-2]*3*self.npts
        u_ub = [ 2]*3*self.npts
        m_lb = [0.0]*self.npts
        m_ub = [1.1*self.mass0]*self.npts
        Eta_lb = [-0.1]*self.npts
        Eta_ub = [1.1]*self.npts
        T_lb = [100]
        T_ub = [1000]
        nu_lb = [-10.0]
        nu_ub = [10.0]
        
        LB = r_lb + v_lb + u_lb + m_lb + Eta_lb + T_lb + nu_lb
        UB = r_ub + v_ub + u_ub + m_ub + Eta_ub + T_ub + nu_ub
        
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
        Eta_lb = self.bnds_Eta[0]
        Eta_ub = self.bnds_Eta[1]
        nu_lb = self.bnds_nu[0]
        nu_ub = self.bnds_nu[1]
        T_lb = self.bnds_T[0]
        T_ub = self.bnds_T[1]
        Om = self.plOm
        
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
        nu      = x[11*n + 1]

        # Collocation timestep
        dt = tof/nf
        
        # Zero gradient
        arr_shape = (1 + 13 + 7*(n-1) + n + 2*n + 1 + 1 + 1 + 1, len(x))
        grad = zeros(arr_shape)
        
        # Gradient estimation
        # grade = pg.estimate_gradient_h(lambda x: self.fitness(x), x, 1e-5)
        # arre = grade.reshape(arr_shape)
        
        # Matrices to use later
        dbeyeneg = -1.0*eye(n-1, M = n, k = 0) + eye(n-1, M = n, k = 1)
        dbeyepos =  1.0*eye(n-1, M = n, k = 0) + eye(n-1, M = n, k = 1)
        
        # Cost function partials
        if self.objtype == 'control':
            J = sum([ 0.5*dt*( (Tm*Eta[k]/m[k])**2.0 + (Tm*Eta[k+1]/m[k+1])**2.0 ) for k in range(0,n-1) ])
            dJ_dm = [-dt*Tm**2.0*Eta[k]**2.0/m[k]**3.0 for k in range(0,n)]
            dJ_dm[1:-1] = [2*a for a in dJ_dm[1:-1]]
            dJ_dEta = [dt*Tm**2.0/m[k]**2.0*Eta[k] for k in range(0,n)]
            dJ_dEta[1:-1] = [2*a for a in dJ_dEta[1:-1]]
            dJ_dT = J/tof
        elif self.objtype == 'fuel':
            J = sum( [ 0.5*dt*( Eta[k+1] + Eta[k] ) for k in range(0, n-1) ] )
            dJ_dm = zeros((n,))
            dJ_dT = J/tof
            dJ_dEta = 0.5*dt*ones((n))
            dJ_dEta[1:-1] = [2.0*a for a in dJ_dEta[1:-1]]
            
        grad[0,( 9*n):(10*n)] = dJ_dm
        grad[0,(10*n):(11*n)] = dJ_dEta 
        grad[0,(11*n)] = dJ_dT

        # Initial state constraints
        grad[1,0*n] = 1.0
        grad[2,1*n] = 1.0
        grad[3,2*n] = 1.0
        grad[4,3*n] = 1.0
        grad[5,4*n] = 1.0
        grad[6,5*n] = 1.0
        grad[7,9*n] = 1.0

        # True anomaly variability
        ecc = self.ecc
        sma = self.sma
        nu1 = (nu)*pi/180
        dnu1_dnu = pi/180
        E = 2*arctan(sqrt((1-ecc)/(1+ecc)) * tan(nu1/2) )
        B = sqrt((1-ecc)/(1+ecc))*tan(nu1/2)
        rc = sma*(1-ecc*cos(E))
        dB_dnu = 1/2*sqrt((1-ecc)/(1+ecc))*(1/cos(nu1/2)**2.0)*dnu1_dnu
        dE_dnu = 2/(1+B**2.0)*dB_dnu
        drc_dnu = sma*ecc*sin(E)*dE_dnu
        do_dnu = rc*dnu1_dnu*array([-sin(nu1),cos(nu1),0]) + drc_dnu*array([cos(nu1),sin(nu1),0])
        dr0_dnu = self.DCM_I_Per.inv().apply(do_dnu)
        dodot_dnu = sqrt(mu*sma)*array([ 
            (rc*-cos(E)*dE_dnu - -sin(E)*drc_dnu)/rc**2.0,  
            (rc*sqrt(1-ecc**2.0)*-sin(E)*dE_dnu - sqrt(1-ecc**2.0)*cos(E)*drc_dnu)/rc**2.0,
            0.0
            ])
        dv0_dnu = self.DCM_I_Per.inv().apply(dodot_dnu)
        grad[1:4,11*n+1] = -dr0_dnu
        grad[4:7,11*n+1] = -dv0_dnu
        
        # Final state constraints
        grad[ 8,1*n-1] = 1.0
        grad[ 9,2*n-1] = 1.0
        grad[10,3*n-1] = 1.0
        grad[11,4*n-1] = 1.0
        grad[12,5*n-1] = 1.0
        grad[13,6*n-1] = 1.0
        
        R_I_PF = Rot_I_PF(self.plOm, self.ep0, tof)
        R_UEN_I = R_I_PF.inv() * self.R_UEN_PF
        rLS_I = R_UEN_I.apply([self.R_eq + self.LS_alt,0,0])
        rt_I = rLS_I + R_UEN_I.apply(self.rt_LS)
        vt_I = R_UEN_I.apply(self.vt_LS) + cross([0,0,self.plOm],rt_I)
        
        Omvec = array([0,0,Om])
        dRf_dt = -cross(Omvec, rt_I)
        dVf_dt = -cross(Omvec, vt_I)
        grad[ 8:11,11*n] = dRf_dt
        grad[10:13,11*n] = dVf_dt

        # print(arre[ 8:11,11*n])
        # print(grad[ 8:11,11*n])
        # print('grad')
        # print(arre[11:14,11*n])
        # print(grad[10:14,11*n])
        
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
        
        # Inequality constraints for nu and T
        grad[(14+7*(n-1)+3*n),(11*n+1)] = -1.0
        grad[(14+7*(n-1)+3*n+1),(11*n+1)] = 1.0
        grad[(14+7*(n-1)+3*n+2),(11*n)] = -1.0
        grad[(14+7*(n-1)+3*n+3),(11*n)] = 1.0
        
        grad = grad.reshape((arr_shape[0]*arr_shape[1],))
        return grad
    
    def run_traj(self, x, plot_traj=0, write_sum=0, write_csv=0):
        
        n = self.npts
        nf = float(n)
        Tm = self.Tmax
        mu = self.mu
        isp = self.isp
        g0 = self.g0
        Eta_lb = self.bnds_Eta[0]
        Eta_ub = self.bnds_Eta[1]
        nu_lb = self.bnds_nu[0]
        nu_ub = self.bnds_nu[1]
        T_lb = self.bnds_T[0]
        T_ub = self.bnds_T[1]
        
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
        nu      = x[11*n + 1]

        # Collocation timestep
        dt = tof/nf

        # Initial state requirement
        r0_I,v0_I = elts_to_rv(self.sma,self.ecc,self.inc,self.Om,self.om,nu,self.mu,self.orb_deg)

        # Landing site position
        R_I_PF = Rot_I_PF(self.plOm, self.ep0, tof)
        R_UEN_I = R_I_PF.inv() * self.R_UEN_PF
        rLS_I = R_UEN_I.apply([self.R_eq + self.LS_alt,0,0])
        rt_I = rLS_I + R_UEN_I.apply(self.rt_LS)
        vt_I = R_UEN_I.apply(self.vt_LS) + cross([0,0,self.plOm],rt_I)
        
        # Starting and ending constraints
        r0 = array([Rx[0],Ry[0],Rz[0]])
        v0 = array([Vx[0],Vy[0],Vz[0]])
        rf = array([Rx[-1],Ry[-1],Rz[-1]])
        vf = array([Vx[-1],Vy[-1],Vz[-1]])
        CONSTR_EQ = \
            list(r0-r0_I) + \
            list(v0-v0_I) + \
            [m[0] - self.mass0] + \
            list(rf-rt_I) + \
            list(vf-vt_I)
            
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
        
        # Time and nu constraints
        nu_const_lb = nu_lb - nu
        nu_constr_ub = nu - nu_ub
        T_constr_lb = T_lb - tof
        T_constr_ub = tof - T_ub
        CONSTR_INEQ = CONSTR_INEQ + [nu_const_lb] + [nu_constr_ub] + [T_constr_lb] + [T_constr_ub] 
        
        # Other objective - minimize control effort
        if self.objtype == 'control':
            OBJVAL = [ sum([ 0.5*dt*( (Tm*Eta[k]/m[k])**2.0 + (Tm*Eta[k+1]/m[k+1])**2.0 ) for k in range(0,n-1) ]) ]
        elif self.objtype == 'fuel':
            OBJVAL = [ sum( [ 0.5*dt*( Eta[k+1] + Eta[k] ) for k in range(0, n-1) ] ) ]
            
        # Landing site position
        X = Rx - rt_I[0]
        Y = Ry - rt_I[1]
        Z = Rz - rt_I[2]

        # Plot, if enabled
        if plot_traj == 1:
            t_arr = linspace(0,tof,n)
            
            fig = plt.figure(1)
            ax = fig.gca(projection='3d')
            ax.set_xlim(-400000,400000)
            ax.set_ylim(-40000,40000)
            ax.set_zlim(-100000,100000)
            plt.title('Position (Landing Site)')
            
            alt = [norm([Rx[k], Ry[k], Rz[k]]) - self.R_eq for k in range(0,n) ]
            
            # Landing site position
            X = Rx - rt_I[0]
            Y = Ry - rt_I[1]
            Z = Rz - rt_I[2]
            
            # max_range = array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
            # Xb = 0.5*max_range*mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
            # Yb = 0.5*max_range*mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
            # Zb = 0.5*max_range*mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
            # # Comment or uncomment following both lines to test the fake bounding box:
            # for xb, yb, zb in zip(Xb, Yb, Zb):
            #    ax.plot([yb], [zb], [xb], 'w')
            
            ax.plot(Y, Z, X,'*-b')

            fout = open('OUT.csv', 'w+')
            fout.write('Time,Alt,Eta,Mass,rLSx,rLSy,rLSz,vLSx,vLSy,vLSz\n')
            for ii in range(0,n):
                outstr = ","
                outstr = outstr.join([ '%.20f'%(ll) for ll in [t_arr[ii],alt[ii],Eta[ii],m[ii],\
                                      Rx[ii]-rt_I[0],Ry[ii]-rt_I[1],Rz[ii]-rt_I[2],\
                                      Vx[ii]-vt_I[0],Vy[ii]-vt_I[1],Vz[ii]-vt_I[2] ] ]+ ["\n"] )
                fout.write(outstr)
            fout.close()
            
            for ii in range(0,n):
                fac = 200000
                Xs = [Rx[ii]- rt_I[0], Rx[ii]- rt_I[0] + fac*Ux[ii]*Eta[ii]**2.0]
                Ys = [Ry[ii]- rt_I[1], Ry[ii]- rt_I[1] + fac*Uy[ii]*Eta[ii]**2.0]
                Zs = [Rz[ii]- rt_I[2], Rz[ii]- rt_I[2] + fac*Uz[ii]*Eta[ii]**2.0]
                ax.plot(Ys,Zs,Xs,'r')

            plt.figure(2)
            plt.subplot(2,2,1)
            plt.plot(t_arr,Eta,'*-b')
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5')
            plt.grid(which='minor', linestyle=':', linewidth='0.5')
            plt.xlabel('Time (sec)')
            plt.ylabel('Throttle (-)')
            plt.title('Throttle')
            
            plt.figure(2)
            plt.subplot(2,2,2)
            plt.plot(t_arr, alt,'*-b')
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5')
            plt.grid(which='minor', linestyle=':', linewidth='0.5')
            plt.xlabel('Time (sec)')
            plt.ylabel('Alt (km)')
            plt.title('Altitude')
            
            plt.figure(2)
            plt.subplot(2,2,3)
            plt.plot(t_arr, m,'*-b')
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5')
            plt.grid(which='minor', linestyle=':', linewidth='0.5')
            plt.xlabel('Time (sec)')
            plt.ylabel('Mass (kg)')
            plt.title('Mass')
            
            print("FINAL MASS: %.3f kg"%(m[-1]))
            print("TOF:        %.3f sec"%(tof))
            print("TA:         %.3f deg"%(nu))
            plt.show()

        # Return everything
        return OBJVAL + CONSTR_EQ + CONSTR_INEQ


# TODO: Add omega x r terms for rotating planet
def run_problem3(config_file):
    """
    Solves the minimum control problem of a 2-D lander under a 
    uniform gravity field. Employs a trapezoidal collocation method
    """
    C = json.load(open(config_file))
    
    # Optimizer config inputs
    opt_config = C['opt']['config']
    opt_in = C['opt']['input']
    opt_out = C['opt']['output']
    CP = C['planet']
    CV = C['vehicle']
    CT = C['target']
    
    # Problem and optimization configuration
    udp = prob_3D_lander(C)
    prob = pg.problem(udp)
    prob.c_tol = opt_config['c_tol']
    npts = opt_config['npts']
    uda = pg.algorithm(pg.nlopt(opt_config['nlopt_alg']))
    uda.set_verbosity(opt_config['verbosity'])
    uda.extract(pg.nlopt).xtol_rel = opt_config['xtol_rel']
    uda.extract(pg.nlopt).ftol_rel = opt_config['ftol_rel']
    uda.extract(pg.nlopt).maxeval = opt_config['maxeval']
    
    # Initial guess from file.
    if opt_in['guess_from_file']:
        print('LOADING GUESS FROM FILE')
        
        Xdata_in = json.load(open(opt_in['in_file']))
        X0 = Xdata_in['X0']

        file_npts = Xdata_in['npts']
        if not (file_npts == npts):
            X0new = []
            for ii in range(0,11):
                X0new = X0new + recast_pts(X0[ii*file_npts:(ii+1)*file_npts],npts) 
            X0 = X0new + [X0[-2]] + [X0[-1]]
    
    # Initial guess - linear profile
    elif opt_in['use_linear_guess']:
        print('CONSTRUCTING LINEAR INITIAL GUESS')
        opt_lg = opt_in['linear_guess']
        tof = opt_lg['tof']
        nu0 = opt_lg['nu0']
        
        # Initial state
        ra = CP['R_eq'] + CV['orbit']['alta']
        rp = CP['R_eq'] + CV['orbit']['altp']
        sma = 0.5*(ra+rp)
        ecc = (ra-rp)/(ra+rp)
        inc = CV['orbit']['inc']
        Om = CV['orbit']['raan']
        om = CV['orbit']['argper']
        nu = nu0
        r0_I,v0_I = elts_to_rv(sma,ecc,inc,Om,om,nu,CP['mu'],CV['orbit']['degrees'])
        
        # Target state
        R_PF_UEN = Rot_PF_UEN(CT['lon'], CT['lat'], degrees=True)
        R_UEN_PF = R_PF_UEN.inv()
        R_I_PF = Rot_I_PF(CP['Omega'], CP['Epoch'], tof)
        R_UEN_I = R_I_PF.inv() * R_UEN_PF
        rLS_I = R_UEN_I.apply([CP['R_eq'] + CT['alt'],0,0])
        rt_I = rLS_I + R_UEN_I.apply(CT['target_pos_UEN'])
        vt_I = R_UEN_I.apply(CT['target_vel_UEN']) + cross([0,0,CP['Omega']],rt_I)
        
        # Construct X vector
        dt = tof/npts
        R = linspace(r0_I, rt_I, npts).transpose()
        V = linspace(v0_I, vt_I, npts).transpose()
        Ux = [1.0] + [(V[0,k+1] - V[0,k])/dt/norm(V[:,k]) for k in range(0,npts-1)]
        Uy = [1.0] + [(V[1,k+1] - V[1,k])/dt/norm(V[:,k]) for k in range(0,npts-1)]
        Uz = [1.0] + [(V[2,k+1] - V[2,k])/dt/norm(V[:,k]) for k in range(0,npts-1)]
        U = array([Ux, Uy, Uz])
        m = linspace(CV['mass']*opt_lg['mfrac_0'], CV['mass']*opt_lg['mfrac_f'], npts)
        Eta =  [1.0] + [(m[k]-m[k+1])/dt*9.81*CV['isp']/CV['Tmax']  for k in range(0,npts-1)] # linspace(opt_lg['eta_0'],opt_lg['eta_f'], npts)
        X0 = list(R.flatten()) + list(V.flatten()) + list(U.flatten()) + list(m) + list(Eta) + [tof] + [nu0]
        
    # Create a population
    print("Creating population...")
    pop = pg.population(prob)
    pop.push_back(X0)

    # Monotonic basin hopping option
    if opt_config['use_mbh_wrapper']:
        mbhuda = pg.mbh(uda, stop=opt_config['mbh']['stop'], perturb=opt_config['mbh']['perturb'])
        algo = pg.algorithm(mbhuda)
    else:
        algo = uda
    
    # Evolve
    if C['opt']['evolve']:
        print("Evolving...")
        pop = algo.evolve(pop)

    # Check feasibility
    is_feas = prob.feasibility_x(pop.champion_x)
    print("Feasible trajectory? ", is_feas)
        
    # Produce summary
    udp.run_traj(pop.champion_x, plot_traj=opt_out['plot_traj'], write_sum=opt_out['write_sum'], write_csv=opt_out['write_csv'] )
        
    if opt_out['write_X']:
        print('WRITING TO FILE')
        outdict = { 'npts':npts, 'X0':list(pop.champion_x) }
        if (not opt_out['only_write_feasible']) or (opt_out['only_write_feasible'] and is_feas):
            json.dump(outdict, open(opt_out['file_X_out'],'w+'), indent=4)
        

    


