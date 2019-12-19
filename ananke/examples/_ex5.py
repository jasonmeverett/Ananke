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
from itertools import chain

class prob_3D_lander_multiphase(object):
    
    # Function initialization
    def __init__(self,C):
        self.C = C
        CP = C['planet']
        CV = C['vehicle']
        CT = C['target']
        CO = C['opt']
        self.objtype = CO['objtype']
        self.nphases = len(CV['phases'])
        self.phases = CV['phases']
        self.npts = CO['config']['npts']
        self.g0 = 9.81
        self.isps = [CV['engines'][ii]['isp'] for ii in range(0, len(CV['engines']))]
        self.Tmaxs = [CV['engines'][ii]['Tmax'] for ii in range(0, len(CV['engines']))]
        self.mass0 = sum([CV['stages'][ii]['ms'] + CV['stages'][ii]['mp'] for ii in CV['phases'][0][0]])
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
        self.Eta_lbs = CO['bounds']['lo']['eta']
        self.Eta_ubs = CO['bounds']['hi']['eta']
        self.nu_lb = CO['bounds']['lo']['nu0']
        self.nu_ub = CO['bounds']['hi']['nu0']
        self.T_lbs = CO['bounds']['lo']['T']
        self.T_ubs = CO['bounds']['hi']['T']
        
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
        self.DCM_I_Per = Rot_I_Perifocal(self.Om, self.inc, self.om, CV['orbit']['degrees'])
        self.r0_I,self.v0_I = elts_to_rv(self.sma,self.ecc,self.inc,self.Om,self.om,self.nu0,CP['mu'],CV['orbit']['degrees'])
        
        return
    
    def get_nec(self):
        P = self.nphases
        npcs = sum([n-1 for n in self.npts])
        npts_tot = sum(self.npts)
        num_nec = int(3 + 3 + P + 3 + 3 + 2*3*(P-1) + 2*3*npcs + npcs + npts_tot)
        return num_nec
    
    def get_nic(self):
        P = self.nphases
        npts_tot = sum(self.npts)
        # return int(npts_tot + P + 2*npts_tot + P + P + 1 + 1)
        return int(P + 2*npts_tot + P + P + 1 + 1)  # Removed alt constraint
    
    # TODO: Set these bounds in a smarter fashion.
    def get_bounds(self):
        
        sf_r = 1.0
        sf_v = 3.0
        rx_lb = [self.r0_I[0] - sf_r*norm(self.r0_I)] * sum(self.npts)
        ry_lb = [self.r0_I[1] - sf_r*norm(self.r0_I)] * sum(self.npts)
        rz_lb = [self.r0_I[2] - sf_r*norm(self.r0_I)] * sum(self.npts)
        rx_ub = [self.r0_I[0] + sf_r*norm(self.r0_I)] * sum(self.npts)
        ry_ub = [self.r0_I[1] + sf_r*norm(self.r0_I)] * sum(self.npts)
        rz_ub = [self.r0_I[2] + sf_r*norm(self.r0_I)] * sum(self.npts)
        r_lb = rx_lb + ry_lb + rz_lb
        r_ub = rx_ub + ry_ub + rz_ub
        v_lb = [-sf_v*norm(self.v0_I)]*3*sum(self.npts)
        v_ub = [ sf_v*norm(self.v0_I)]*3*sum(self.npts)
        u_lb = [-2]*3*sum(self.npts)
        u_ub = [ 2]*3*sum(self.npts)
        m_lb = [0.1]*sum(self.npts)
        m_ub = [1.1*self.mass0]*sum(self.npts)
        Eta_lb = [-0.1]*sum(self.npts)
        Eta_ub = [1.1]*sum(self.npts)
        T_lb = [1]*self.nphases
        T_ub = [1000]*self.nphases
        nu_lb = [-30.0]
        nu_ub = [30.0]
        LB = r_lb + v_lb + u_lb + m_lb + Eta_lb + nu_lb + T_lb
        UB = r_ub + v_ub + u_ub + m_ub + Eta_ub + nu_ub + T_ub
        return (LB, UB)
    
    def fitness(self, x):
        return self.run_traj(x, plot_traj=0)
    
    def summary(self, x):
        return self.run_traj(x, plot_traj=1)
    
    def gradient(self, x):
        
        # Parameters
        ns = self.C['opt']['config']['npts']
        nsm1t = sum([nnn-1 for nnn in ns])
        nt = sum(ns)
        P = self.nphases
        mu = self.mu
        Om = self.plOm
        g0 = self.g0
        
        Rx = []
        Ry = []
        Rz = []
        Vx = []
        Vy = []
        Vz = []
        Ux = []
        Uy = []
        Uz = []
        m = []
        Eta = []
        nu = 0.0
        T = []

        # Split up the state vector into a more meaningful shape
        i0 = 0
        for ii in range(0,P):
            Rx = Rx + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            Ry = Ry + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            Rz = Rz + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            Vx = Vx + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            Vy = Vy + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            Vz = Vz + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            Ux = Ux + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            Uy = Uy + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            Uz = Uz + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            m  = m  + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            Eta= Eta+ [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        nu = x[i0]
        i0 = i0 + 1
        T = x[(i0):(i0 + P)]
        tof = sum(T)
        
        # Gradient array shape
        arr_x = len(x)
        arr_y = 1 + self.get_nec() + self.get_nic()
        arr_shape = (arr_y,arr_x)
        
        # Estimate the gradient
        # t1e = time.clock()
        # grad_est = pg.estimate_gradient(lambda x: self.fitness(x), x, 1e-6)
        # t2e = time.clock()
        # gearr = grad_est.reshape(arr_shape)
        
        # t1g = time.clock()
        
        # Real gradient initialization
        grad = zeros(arr_shape)
        
        # Cost function
        i0 = 9*nt
        i0t = 11*nt+1
        if self.objtype == 'control':
            for ii in range(0,P):
                dt = T[ii]/ns[ii]
                Tm = self.Tmaxs[self.phases[ii][1]]
                J = sum([0.5*dt*((Tm*Eta[ii][k]/m[ii][k])**2.0 + (Tm*Eta[ii][k+1]/m[ii][k+1])**2.0)  for k in range(0, ns[ii]-1)])
                dJ_dm = [-dt*Tm**2.0*Eta[ii][k]**2.0/m[ii][k]**3.0 for k in range(0,ns[ii])]
                dJ_dm[1:-1] = [2*a for a in dJ_dm[1:-1]]
                dJ_dEta = [dt*Tm**2.0*Eta[ii][k]/m[ii][k]**2.0 for k in range(0, ns[ii])]
                dJ_dEta[1:-1] = [2*a for a in dJ_dEta[1:-1]]
                dJ_dT = J/T[ii]
                grad[0,(i0):(i0+ns[ii])] = dJ_dm
                grad[0,(i0+nt):(i0+nt+ns[ii])] = dJ_dEta
                grad[0,(i0t)] = dJ_dT
                i0t = i0t + 1
                i0 = i0 + ns[ii]
        elif self.objtype == 'fuel':
            for ii in range(0,P):
                dt = T[ii]/ns[ii]
                Tm = self.Tmaxs[self.phases[ii][1]]
                J = sum([0.5*dt*(Eta[ii][k] + Eta[ii][k+1])  for k in range(0, ns[ii]-1)])
                dJ_dm = zeros((ns[ii],))
                dJ_dT = J/T[ii]
                dJ_dEta = 0.5*dt*ones((ns[ii],))
                dJ_dEta[1:-1] = [2*a for a in dJ_dEta[1:-1]]
                grad[0,(i0):(i0+ns[ii])] = dJ_dm
                grad[0,(i0+nt):(i0+nt+ns[ii])] = dJ_dEta
                grad[0,(i0t)] = dJ_dT
                i0t = i0t + 1
                i0 = i0 + ns[ii]
        if self.objtype == 'controlF':
            for ii in range(0,P):
                dt = T[ii]/ns[ii]
                Tm = self.Tmaxs[self.phases[ii][1]]
                J = sum([0.5*dt*((Tm*Eta[ii][k]/m[ii][k])**2.0 + (Tm*Eta[ii][k+1]/m[ii][k+1])**2.0)  for k in range(0, ns[ii]-1)])
                dJ_dm = [-dt*Tm**2.0*Eta[ii][k]**2.0/m[ii][k]**3.0 for k in range(0,ns[ii])]
                dJ_dm[1:-1] = [2*a for a in dJ_dm[1:-1]]
                dJ_dEta = [dt*Tm**2.0*Eta[ii][k]/m[ii][k]**2.0 for k in range(0, ns[ii])]
                dJ_dEta[1:-1] = [2*a for a in dJ_dEta[1:-1]]
                dJ_dT = J/T[ii]
                if ii == P-1:
                    grad[0,(i0):(i0+ns[ii])] = dJ_dm
                    grad[0,(i0+nt):(i0+nt+ns[ii])] = dJ_dEta
                    grad[0,(i0t)] = dJ_dT
                i0t = i0t + 1
                i0 = i0 + ns[ii]
        elif self.objtype == 'fuelF':
            for ii in range(0,P):
                dt = T[ii]/ns[ii]
                Tm = self.Tmaxs[self.phases[ii][1]]
                J = sum([0.5*dt*(Eta[ii][k] + Eta[ii][k+1])  for k in range(0, ns[ii]-1)])
                dJ_dm = zeros((ns[ii],))
                dJ_dT = J/T[ii]
                dJ_dEta = 0.5*dt*ones((ns[ii],))
                dJ_dEta[1:-1] = [2*a for a in dJ_dEta[1:-1]]
                if ii == P-1:
                    grad[0,(i0):(i0+ns[ii])] = dJ_dm
                    grad[0,(i0+nt):(i0+nt+ns[ii])] = dJ_dEta
                    grad[0,(i0t)] = dJ_dT
                i0t = i0t + 1
                i0 = i0 + ns[ii]
                
        # Initial position and velocity constraint
        grad[1,0*nt] = 1.0
        grad[2,1*nt] = 1.0
        grad[3,2*nt] = 1.0
        grad[4,3*nt] = 1.0
        grad[5,4*nt] = 1.0
        grad[6,5*nt] = 1.0
        
        # True anomaly variability (for starting position/velocity constraint)
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
        grad[1:4,11*nt] = -dr0_dnu
        grad[4:7,11*nt] = -dv0_dnu
        
        # Mass constraints. For starts of stages, ensure mass aligns directly
        # with the total mass. Otherwise, match mass of previous stage's mass.
        # First phase requires the former.
        grad[7,9*nt] = 1.0
        ns_p0 = len(self.phases[0][0])
        i0 = ns[0]
        i0m = 1
        for ii in range(1,P):
            ns_pii = len(self.phases[ii][0])
            if ns_pii > ns_p0:
                print("ERROR: GAINING STAGES DURING DESCENT")
            elif ns_pii < ns_p0:
                grad[7+i0m,9*nt+i0] = 1.0
            elif ns_pii == ns_p0:
                grad[7+i0m,9*nt+i0] = 1.0
                grad[7+i0m,9*nt+i0-1] = -1.0
            i0 = i0 + ns[ii]
            i0m = i0m + 1
            ns_p0 = len(self.phases[ii][0])
        
        # Final position/velocity constraints
        grad[7+P+0,1*nt-1] = 1.0
        grad[7+P+1,2*nt-1] = 1.0
        grad[7+P+2,3*nt-1] = 1.0
        grad[7+P+3,4*nt-1] = 1.0
        grad[7+P+4,5*nt-1] = 1.0
        grad[7+P+5,6*nt-1] = 1.0

        # Rotating planet (for final position/velocity constraints)
        R_I_PF = Rot_I_PF(self.plOm, self.ep0, tof)
        R_UEN_I = R_I_PF.inv() * self.R_UEN_PF
        rLS_I = R_UEN_I.apply([self.R_eq + self.LS_alt,0,0])
        rt_I = rLS_I + R_UEN_I.apply(self.rt_LS)
        vt_I = R_UEN_I.apply(self.vt_LS) + cross([0,0,self.plOm],rt_I)
        Omvec = array([0,0,Om])
        dRf_dt = -cross(Omvec, rt_I)
        dVf_dt = -cross(Omvec, vt_I)
        i0t = 11*nt+1
        for ii in range(0,P):
            grad[(7+P+0):(7+P+3),i0t] = dRf_dt
            grad[(7+P+3):(7+P+6),i0t] = dVf_dt
            i0t = i0t + 1

        # Phase boundary position constraint
        i0 = 0
        i0y = 7+P+6
        for ii in range(0,P-1):
            grad[(i0y),(i0+ns[ii]-1)] = 1.0
            grad[(i0y),(i0+ns[ii])] = -1.0
            grad[(i0y+1),(i0+ns[ii]+nt-1)] = 1.0
            grad[(i0y+1),(i0+ns[ii]+nt)] = -1.0
            grad[(i0y+2),(i0+ns[ii]+2*nt-1)] = 1.0
            grad[(i0y+2),(i0+ns[ii]+2*nt)] = -1.0
            i0y = i0y + 3
            i0 = i0 + ns[ii]
        
        # Phase boundary velocity constraint
        i0 = 0
        i0y = 7+P+6+3*(P-1)
        for ii in range(0,P-1):
            grad[(i0y),(i0+ns[ii]+3*nt-1)] = 1.0
            grad[(i0y),(i0+ns[ii]+3*nt)] = -1.0
            grad[(i0y+1),(i0+ns[ii]+4*nt-1)] = 1.0
            grad[(i0y+1),(i0+ns[ii]+4*nt)] = -1.0
            grad[(i0y+2),(i0+ns[ii]+5*nt-1)] = 1.0
            grad[(i0y+2),(i0+ns[ii]+5*nt)] = -1.0
            i0y = i0y + 3
            i0 = i0 + ns[ii]

        # Path constraints - position
        i0x = 0
        i0y = 7+P+6+3*(P-1)+3*(P-1)
        i0t = 0
        for ii in range(0,P):
            dt = T[ii]/ns[ii]
            
            # wrt. R
            grad[(i0y)          :(i0y+ns[ii]-1)             ,(i0x)          :(i0x+ns[ii])       ] = -1.0*eye(ns[ii]-1, M = ns[ii], k = 0) + eye(ns[ii]-1, M = ns[ii], k = 1)
            grad[(i0y+(nt-P))   :(i0y+(nt-P)+ns[ii]-1)      ,(i0x+nt)       :(i0x+nt+ns[ii])    ] = -1.0*eye(ns[ii]-1, M = ns[ii], k = 0) + eye(ns[ii]-1, M = ns[ii], k = 1)
            grad[(i0y+2*(nt-P)) :(i0y+2*(nt-P)+ns[ii]-1)    ,(i0x+2*nt)     :(i0x+2*nt+ns[ii])  ] = -1.0*eye(ns[ii]-1, M = ns[ii], k = 0) + eye(ns[ii]-1, M = ns[ii], k = 1)
            
            # wrt. V
            grad[(i0y)          :(i0y+ns[ii]-1)             ,(3*nt+i0x)          :(3*nt+i0x+ns[ii])       ] = -0.5*dt*(eye(ns[ii]-1, M = ns[ii], k = 0) + eye(ns[ii]-1, M = ns[ii], k = 1))
            grad[(i0y+(nt-P))   :(i0y+(nt-P)+ns[ii]-1)      ,(3*nt+i0x+nt)       :(3*nt+i0x+nt+ns[ii])    ] = -0.5*dt*(eye(ns[ii]-1, M = ns[ii], k = 0) + eye(ns[ii]-1, M = ns[ii], k = 1))
            grad[(i0y+2*(nt-P)) :(i0y+2*(nt-P)+ns[ii]-1)    ,(3*nt+i0x+2*nt)     :(3*nt+i0x+2*nt+ns[ii])  ] = -0.5*dt*(eye(ns[ii]-1, M = ns[ii], k = 0) + eye(ns[ii]-1, M = ns[ii], k = 1))
            
            # wrt. T
            grad[(i0y)          :(i0y+ns[ii]-1)             ,(11*nt+1+i0t)] = [ -1/(2*ns[ii])*(Vx[ii][kk] + Vx[ii][kk+1]) for kk in range(0,ns[ii]-1)]
            grad[(i0y+(nt-P))   :(i0y+(nt-P)+ns[ii]-1)      ,(11*nt+1+i0t)] = [ -1/(2*ns[ii])*(Vy[ii][kk] + Vy[ii][kk+1]) for kk in range(0,ns[ii]-1)]
            grad[(i0y+2*(nt-P)) :(i0y+2*(nt-P)+ns[ii]-1)    ,(11*nt+1+i0t)] = [ -1/(2*ns[ii])*(Vz[ii][kk] + Vz[ii][kk+1]) for kk in range(0,ns[ii]-1)]
            i0y = i0y + ns[ii]-1
            i0x = i0x + ns[ii]
            i0t = i0t + 1
        
        
        # Path constraints - Velocity
        i0y = 7+P+6+3*(P-1)+3*(P-1)+3*(nt-P)
        i0x = 0
        i0t = 0
        for ii in range(0,P):
            Tm = self.Tmaxs[self.phases[ii][1]]
            dt = T[ii]/ns[ii]
            nf = float(ns[ii])
            dVxp_dRx = zeros((ns[ii]-1,ns[ii]))
            dVxp_dRy = zeros((ns[ii]-1,ns[ii]))
            dVxp_dRz = zeros((ns[ii]-1,ns[ii]))
            dVyp_dRx = zeros((ns[ii]-1,ns[ii]))
            dVyp_dRy = zeros((ns[ii]-1,ns[ii]))
            dVyp_dRz = zeros((ns[ii]-1,ns[ii]))
            dVzp_dRx = zeros((ns[ii]-1,ns[ii]))
            dVzp_dRy = zeros((ns[ii]-1,ns[ii]))
            dVzp_dRz = zeros((ns[ii]-1,ns[ii]))
            dVxp_dVx = -1.0*eye(ns[ii]-1, M = ns[ii], k = 0) + eye(ns[ii]-1, M = ns[ii], k = 1)
            dVyp_dVy = -1.0*eye(ns[ii]-1, M = ns[ii], k = 0) + eye(ns[ii]-1, M = ns[ii], k = 1)
            dVzp_dVz = -1.0*eye(ns[ii]-1, M = ns[ii], k = 0) + eye(ns[ii]-1, M = ns[ii], k = 1)
            dVxp_dUx = zeros((ns[ii]-1,ns[ii]))
            dVyp_dUy = zeros((ns[ii]-1,ns[ii]))
            dVzp_dUz = zeros((ns[ii]-1,ns[ii]))
            dVxp_dm = zeros((ns[ii]-1,ns[ii]))
            dVyp_dm = zeros((ns[ii]-1,ns[ii]))
            dVzp_dm = zeros((ns[ii]-1,ns[ii]))
            dVxp_dEta = zeros((ns[ii]-1,ns[ii]))
            dVyp_dEta = zeros((ns[ii]-1,ns[ii]))
            dVzp_dEta = zeros((ns[ii]-1,ns[ii]))
            dVxp_dT = zeros((ns[ii]-1,))
            dVyp_dT = zeros((ns[ii]-1,))
            dVzp_dT = zeros((ns[ii]-1,))
            for k in range(0, ns[ii]-1):
                Rk = sqrt(Rx[ii][k]**2.0 + Ry[ii][k]**2.0 + Rz[ii][k]**2.0)
                Rkp1 = sqrt(Rx[ii][k+1]**2.0 + Ry[ii][k+1]**2.0 + Rz[ii][k+1]**2.0)
                muoRk5 = mu/Rk**5.0
                muoRk3 = mu/Rk**3.0
                dVxp_dRx[k,k] = 0.5*dt*(3*muoRk5*Rx[ii][k]*Rx[ii][k] + muoRk3)
                dVxp_dRy[k,k] = 0.5*dt*(3*muoRk5*Rx[ii][k]*Ry[ii][k])
                dVxp_dRz[k,k] = 0.5*dt*(3*muoRk5*Rx[ii][k]*Rz[ii][k])
                dVyp_dRx[k,k] = 0.5*dt*(3*muoRk5*Ry[ii][k]*Rx[ii][k])
                dVyp_dRy[k,k] = 0.5*dt*(3*muoRk5*Ry[ii][k]*Ry[ii][k] + muoRk3)
                dVyp_dRz[k,k] = 0.5*dt*(3*muoRk5*Ry[ii][k]*Rz[ii][k])
                dVzp_dRx[k,k] = 0.5*dt*(3*muoRk5*Rz[ii][k]*Rx[ii][k])
                dVzp_dRy[k,k] = 0.5*dt*(3*muoRk5*Rz[ii][k]*Ry[ii][k])
                dVzp_dRz[k,k] = 0.5*dt*(3*muoRk5*Rz[ii][k]*Rz[ii][k] + muoRk3)
                muoRkp15 = mu/Rkp1**5.0
                muoRkp13 = mu/Rkp1**3.0
                dVxp_dRx[k,k+1] = 0.5*dt*(3*muoRkp15*Rx[ii][k+1]*Rx[ii][k+1] + muoRkp13)
                dVxp_dRy[k,k+1] = 0.5*dt*(3*muoRkp15*Rx[ii][k+1]*Ry[ii][k+1])
                dVxp_dRz[k,k+1] = 0.5*dt*(3*muoRkp15*Rx[ii][k+1]*Rz[ii][k+1])
                dVyp_dRx[k,k+1] = 0.5*dt*(3*muoRkp15*Ry[ii][k+1]*Rx[ii][k+1])
                dVyp_dRy[k,k+1] = 0.5*dt*(3*muoRkp15*Ry[ii][k+1]*Ry[ii][k+1] + muoRkp13)
                dVyp_dRz[k,k+1] = 0.5*dt*(3*muoRkp15*Ry[ii][k+1]*Rz[ii][k+1])
                dVzp_dRx[k,k+1] = 0.5*dt*(3*muoRkp15*Rz[ii][k+1]*Rx[ii][k+1])
                dVzp_dRy[k,k+1] = 0.5*dt*(3*muoRkp15*Rz[ii][k+1]*Ry[ii][k+1])
                dVzp_dRz[k,k+1] = 0.5*dt*(3*muoRkp15*Rz[ii][k+1]*Rz[ii][k+1] + muoRkp13)
                dVxp_dUx[k,k] = -0.5*dt*Tm*Eta[ii][k]/m[ii][k]
                dVyp_dUy[k,k] = -0.5*dt*Tm*Eta[ii][k]/m[ii][k]
                dVzp_dUz[k,k] = -0.5*dt*Tm*Eta[ii][k]/m[ii][k]
                dVxp_dUx[k,k+1] = -0.5*dt*Tm*Eta[ii][k+1]/m[ii][k+1]
                dVyp_dUy[k,k+1] = -0.5*dt*Tm*Eta[ii][k+1]/m[ii][k+1]
                dVzp_dUz[k,k+1] = -0.5*dt*Tm*Eta[ii][k+1]/m[ii][k+1]
                dVxp_dm[k,k] = 0.5*dt*Tm*Eta[ii][k]*Ux[ii][k]/m[ii][k]**2.0
                dVyp_dm[k,k] = 0.5*dt*Tm*Eta[ii][k]*Uy[ii][k]/m[ii][k]**2.0
                dVzp_dm[k,k] = 0.5*dt*Tm*Eta[ii][k]*Uz[ii][k]/m[ii][k]**2.0
                dVxp_dm[k,k+1] = 0.5*dt*Tm*Eta[ii][k+1]*Ux[ii][k+1]/m[ii][k+1]**2.0
                dVyp_dm[k,k+1] = 0.5*dt*Tm*Eta[ii][k+1]*Uy[ii][k+1]/m[ii][k+1]**2.0
                dVzp_dm[k,k+1] = 0.5*dt*Tm*Eta[ii][k+1]*Uz[ii][k+1]/m[ii][k+1]**2.0
                dVxp_dEta[k,k] = -0.5*dt*Tm*Ux[ii][k]/m[ii][k]
                dVyp_dEta[k,k] = -0.5*dt*Tm*Uy[ii][k]/m[ii][k]
                dVzp_dEta[k,k] = -0.5*dt*Tm*Uz[ii][k]/m[ii][k]
                dVxp_dEta[k,k+1] = -0.5*dt*Tm*Ux[ii][k+1]/m[ii][k+1]
                dVyp_dEta[k,k+1] = -0.5*dt*Tm*Uy[ii][k+1]/m[ii][k+1]
                dVzp_dEta[k,k+1] = -0.5*dt*Tm*Uz[ii][k+1]/m[ii][k+1]
                dVxp_dT[k] = - 0.5/nf*( (Tm*Eta[ii][k]*Ux[ii][k]/m[ii][k]) + (Tm*Eta[ii][k+1]*Ux[ii][k+1]/m[ii][k+1]) + (-mu/Rk**3.0)*Rx[ii][k] + (-mu/Rkp1**3.0)*Rx[ii][k+1] )
                dVyp_dT[k] = - 0.5/nf*( (Tm*Eta[ii][k]*Uy[ii][k]/m[ii][k]) + (Tm*Eta[ii][k+1]*Uy[ii][k+1]/m[ii][k+1]) + (-mu/Rk**3.0)*Ry[ii][k] + (-mu/Rkp1**3.0)*Ry[ii][k+1] )
                dVzp_dT[k] = - 0.5/nf*( (Tm*Eta[ii][k]*Uz[ii][k]/m[ii][k]) + (Tm*Eta[ii][k+1]*Uz[ii][k+1]/m[ii][k+1]) + (-mu/Rk**3.0)*Rz[ii][k] + (-mu/Rkp1**3.0)*Rz[ii][k+1] )
            grad[(i0y+0*(nt-P)):(i0y+0*(nt-P)+ns[ii]-1),(i0x+0*nt):(i0x+0*nt+ns[ii])] = dVxp_dRx
            grad[(i0y+0*(nt-P)):(i0y+0*(nt-P)+ns[ii]-1),(i0x+1*nt):(i0x+1*nt+ns[ii])] = dVxp_dRy
            grad[(i0y+0*(nt-P)):(i0y+0*(nt-P)+ns[ii]-1),(i0x+2*nt):(i0x+2*nt+ns[ii])] = dVxp_dRz
            grad[(i0y+1*(nt-P)):(i0y+1*(nt-P)+ns[ii]-1),(i0x+0*nt):(i0x+0*nt+ns[ii])] = dVyp_dRx
            grad[(i0y+1*(nt-P)):(i0y+1*(nt-P)+ns[ii]-1),(i0x+1*nt):(i0x+1*nt+ns[ii])] = dVyp_dRy
            grad[(i0y+1*(nt-P)):(i0y+1*(nt-P)+ns[ii]-1),(i0x+2*nt):(i0x+2*nt+ns[ii])] = dVyp_dRz
            grad[(i0y+2*(nt-P)):(i0y+2*(nt-P)+ns[ii]-1),(i0x+0*nt):(i0x+0*nt+ns[ii])] = dVzp_dRx
            grad[(i0y+2*(nt-P)):(i0y+2*(nt-P)+ns[ii]-1),(i0x+1*nt):(i0x+1*nt+ns[ii])] = dVzp_dRy
            grad[(i0y+2*(nt-P)):(i0y+2*(nt-P)+ns[ii]-1),(i0x+2*nt):(i0x+2*nt+ns[ii])] = dVzp_dRz
            grad[(i0y+0*(nt-P)):(i0y+0*(nt-P)+ns[ii]-1),(i0x+3*nt):(i0x+3*nt+ns[ii])] = dVxp_dVx
            grad[(i0y+1*(nt-P)):(i0y+1*(nt-P)+ns[ii]-1),(i0x+4*nt):(i0x+4*nt+ns[ii])] = dVyp_dVy
            grad[(i0y+2*(nt-P)):(i0y+2*(nt-P)+ns[ii]-1),(i0x+5*nt):(i0x+5*nt+ns[ii])] = dVzp_dVz
            grad[(i0y+0*(nt-P)):(i0y+0*(nt-P)+ns[ii]-1),(i0x+6*nt):(i0x+6*nt+ns[ii])] = dVxp_dUx
            grad[(i0y+1*(nt-P)):(i0y+1*(nt-P)+ns[ii]-1),(i0x+7*nt):(i0x+7*nt+ns[ii])] = dVyp_dUy
            grad[(i0y+2*(nt-P)):(i0y+2*(nt-P)+ns[ii]-1),(i0x+8*nt):(i0x+8*nt+ns[ii])] = dVzp_dUz
            grad[(i0y+0*(nt-P)):(i0y+0*(nt-P)+ns[ii]-1),(i0x+9*nt):(i0x+9*nt+ns[ii])] = dVxp_dm
            grad[(i0y+1*(nt-P)):(i0y+1*(nt-P)+ns[ii]-1),(i0x+9*nt):(i0x+9*nt+ns[ii])] = dVyp_dm
            grad[(i0y+2*(nt-P)):(i0y+2*(nt-P)+ns[ii]-1),(i0x+9*nt):(i0x+9*nt+ns[ii])] = dVzp_dm
            grad[(i0y+0*(nt-P)):(i0y+0*(nt-P)+ns[ii]-1),(i0x+10*nt):(i0x+10*nt+ns[ii])] = dVxp_dEta
            grad[(i0y+1*(nt-P)):(i0y+1*(nt-P)+ns[ii]-1),(i0x+10*nt):(i0x+10*nt+ns[ii])] = dVyp_dEta
            grad[(i0y+2*(nt-P)):(i0y+2*(nt-P)+ns[ii]-1),(i0x+10*nt):(i0x+10*nt+ns[ii])] = dVzp_dEta
            grad[(i0y+0*(nt-P)):(i0y+0*(nt-P)+ns[ii]-1),(11*nt+1+i0t)] = dVxp_dT
            grad[(i0y+1*(nt-P)):(i0y+1*(nt-P)+ns[ii]-1),(11*nt+1+i0t)] = dVyp_dT
            grad[(i0y+2*(nt-P)):(i0y+2*(nt-P)+ns[ii]-1),(11*nt+1+i0t)] = dVzp_dT
            i0y = i0y + ns[ii]-1
            i0x = i0x + ns[ii]
            i0t = i0t + 1
            
        # a1 = gearr[(i0y+0*(nt-P)):(i0y+0*(nt-P)+ns[ii]-1),(11*nt+1+i0t)]
        # a2 = grad[(i0y+0*(nt-P)):(i0y+0*(nt-P)+ns[ii]-1),(11*nt+1+i0t)]
        # print(norm(a1-a2))
        # quit()
            
        # Mass path constraints
        i0y = 7+P+6+3*(P-1)+3*(P-1)+3*(nt-P)+3*(nt-P)
        i0x = 0
        i0t = 0
        for ii in range(0, P):
            dt = T[ii]/ns[ii]
            nf = float(ns[ii])
            Tm = self.Tmaxs[self.phases[ii][1]]
            isp = self.isps[self.phases[ii][1]]
            dmp_dm = 1.0*eye(ns[ii]-1, M = ns[ii], k = 0) - 1.0*eye(ns[ii]-1, M = ns[ii], k = 1)
            dmp_dEta = -0.5*dt*Tm/(g0*isp)*(1.0*eye(ns[ii]-1, M = ns[ii], k = 0) + 1.0*eye(ns[ii]-1, M = ns[ii], k = 1))
            dmp_dT = [-0.5/nf*Tm/(g0*isp)*(Eta[ii][k]+Eta[ii][k+1]) for k in range(0,ns[ii]-1)]
            grad[(i0y):(i0y+ns[ii]-1),(i0x+9*nt):(i0x+9*nt+ns[ii])] = dmp_dm
            grad[(i0y):(i0y+ns[ii]-1),(i0x+10*nt):(i0x+10*nt+ns[ii])] = dmp_dEta
            grad[(i0y):(i0y+ns[ii]-1),(11*nt+1+i0t)] = dmp_dT
            i0y = i0y + ns[ii]-1
            i0x = i0x + ns[ii]
            i0t = i0t + 1
        
        # Thrust unit vector magnitudes
        i0y = 7+P+6+3*(P-1)+3*(P-1)+3*(nt-P)+3*(nt-P)+(nt-P)
        i0x = 0
        for ii in range(0, P):
            dUm_dUx = 2*diag(Ux[ii])
            dUm_dUy = 2*diag(Uy[ii])
            dUm_dUz = 2*diag(Uz[ii])
            grad[(i0y):(i0y+ns[ii]),(i0x+6*nt):(i0x+6*nt+ns[ii])] = dUm_dUx
            grad[(i0y):(i0y+ns[ii]),(i0x+7*nt):(i0x+7*nt+ns[ii])] = dUm_dUy
            grad[(i0y):(i0y+ns[ii]),(i0x+8*nt):(i0x+8*nt+ns[ii])] = dUm_dUz
            i0y = i0y + ns[ii]
            i0x = i0x + ns[ii]

        # Final Mass
        i0y = 7+P+6+3*(P-1)+3*(P-1)+3*(nt-P)+3*(nt-P)+(nt-P)+nt
        i0x = 0
        for ii in range(0,P):
            grad[i0y,(i0x+9*nt+ns[ii]-1)] = -1.0
            i0y = i0y + 1
            i0x = i0x + ns[ii]

        # Inequality thrust magnitude constraints
        i0y = 7+P+6+3*(P-1)+3*(P-1)+3*(nt-P)+3*(nt-P)+(nt-P)+nt+P
        i0x = 0
        for ii in range(0, P):
            dElb_dEta = -eye(ns[ii])
            dEub_dEta = eye(ns[ii])
            grad[(i0y):(i0y+ns[ii]),(i0x+10*nt):(i0x+10*nt+ns[ii])] = dElb_dEta
            grad[(i0y+nt):(i0y+nt+ns[ii]),(i0x+10*nt):(i0x+10*nt+ns[ii])] = dEub_dEta
            i0y = i0y + ns[ii]
            i0x = i0x + ns[ii]
            
        # True anomaly constraints
        i0y = 7+P+6+3*(P-1)+3*(P-1)+3*(nt-P)+3*(nt-P)+(nt-P)+nt+P+2*nt
        i0x = 0
        grad[i0y,11*nt] = -1.0
        grad[i0y+1,11*nt] = 1.0
            
        # Time constraints
        i0y = 7+P+6+3*(P-1)+3*(P-1)+3*(nt-P)+3*(nt-P)+(nt-P)+nt+P+2*nt+2
        i0x = 0
        for ii in range(0,P):
            grad[i0y,i0x+11*nt+1] = -1.0
            grad[i0y+P,i0x+11*nt+1] = 1.0
            i0y = i0y + 1
            i0x = i0x + 1

        grad_rtn = grad.reshape((arr_shape[0]*arr_shape[1],))
        # t2g = time.clock()
        
        # elg = t2g-t1g
        # ele = t2e-t1e
        
        # print("Estimated:  %.8f sec"%(ele) )
        # print("Calculated: %.8f sec"%(elg) )
        # quit()
        
        # a1 = gearr
        # a2 = grad
        # print(norm(a1-a2))
        # print(a1)
        # print(a2)
        # quit()

        return grad_rtn

    
##############################################################################
##############################################################################
##############################################################################
##############################################################################    
##############################################################################
##############################################################################
##############################################################################
##############################################################################    
##############################################################################
##############################################################################
##############################################################################
##############################################################################    
##############################################################################
##############################################################################
##############################################################################

    
    def run_traj(self, x, plot_traj=0, write_sum=0, write_csv=0):
        P = self.nphases
        npcs = sum([n-1 for n in self.npts])
        npt = sum(self.npts)
        ns = self.npts
        mu = self.mu
        g0 = self.g0
        Eta_lbs = self.Eta_lbs
        Eta_ubs = self.Eta_ubs
        T_lbs = self.T_lbs
        T_ubs = self.T_ubs
        nu_lb = self.nu_lb
        nu_ub = self.nu_ub
        
        CP = self.C['planet']
        CV = self.C['vehicle']
        CT = self.C['target']
        CO = self.C['opt']
                
        # Return lists
        OBJVAL = []
        CONSTR_EQ = []
        CONSTR_INEQ = []
        
        # Extract state information
        Rx      = x[(0*npt)  :(1*npt)  ]
        Ry      = x[(1*npt)  :(2*npt)  ]
        Rz      = x[(2*npt)  :(3*npt)  ]
        Vx      = x[(3*npt)  :(4*npt)  ]
        Vy      = x[(4*npt)  :(5*npt)  ]
        Vz      = x[(5*npt)  :(6*npt)  ]
        Ux      = x[(6*npt)  :(7*npt)  ]
        Uy      = x[(7*npt)  :(8*npt)  ]
        Uz      = x[(8*npt)  :(9*npt)  ]
        m       = x[(9*npt)  :(10*npt) ]
        Eta     = x[(10*npt) :(11*npt) ]
        nu      = x[11*npt]
        tofs    = x[(11*npt+1):(11*npt + 1 + P)]
        tof_tot = sum(tofs)
        
        # Cost function
        J = 0
        if self.objtype == 'control':
            nidx0 = 0
            for ii in range(0,P):
                Tm = self.Tmaxs[self.phases[ii][1]]
                dt = tofs[ii]/ns[ii]
                J = J + sum([ 0.5*dt*( (Tm*Eta[k]/m[k])**2.0 + (Tm*Eta[k+1]/m[k+1])**2.0 ) for k in range(nidx0,nidx0 + ns[ii]-1) ])    
                nidx0 = nidx0 + ns[ii]    
        elif self.objtype == 'fuel':
            nidx0 = 0
            for ii in range(0,P):
                dt = tofs[ii]/ns[ii]
                J = J + sum( [ 0.5*dt*( Eta[k+1] + Eta[k] ) for k in range(nidx0,nidx0 + ns[ii]-1) ] )
                nidx0 = nidx0 + ns[ii]
        elif self.objtype == 'controlF':
            nidx0 = 0
            for ii in range(0,P):
                Tm = self.Tmaxs[self.phases[ii][1]]
                dt = tofs[ii]/ns[ii]
                if ii == P-1:
                    J = J + sum([ 0.5*dt*( (Tm*Eta[k]/m[k])**2.0 + (Tm*Eta[k+1]/m[k+1])**2.0 ) for k in range(nidx0,nidx0 + ns[ii]-1) ])    
                nidx0 = nidx0 + ns[ii]    
        elif self.objtype == 'fuelF':
            nidx0 = 0
            for ii in range(0,P):
                dt = tofs[ii]/ns[ii]
                if ii == P-1:
                    J = J + sum( [ 0.5*dt*( Eta[k+1] + Eta[k] ) for k in range(nidx0,nidx0 + ns[ii]-1) ] )
                nidx0 = nidx0 + ns[ii]
                
        # Set objective value        
        OBJVAL = [J]
        
        # Initial State equality constraint
        r0_I,v0_I = elts_to_rv(self.sma,self.ecc,self.inc,self.Om,self.om,nu,self.mu,self.orb_deg)

        # Landing site position
        R_I_PF = Rot_I_PF(self.plOm, self.ep0, tof_tot)
        R_UEN_I = R_I_PF.inv() * self.R_UEN_PF
        rLS_I = R_UEN_I.apply([self.R_eq + self.LS_alt,0,0])
        rt_I = rLS_I + R_UEN_I.apply(self.rt_LS)
        vt_I = R_UEN_I.apply(self.vt_LS) + cross([0,0,self.plOm],rt_I)
        
        # Starting state constraints
        r0 = array([Rx[0],Ry[0],Rz[0]])
        v0 = array([Vx[0],Vy[0],Vz[0]])
        CONSTR_EQ = CONSTR_EQ + list(r0-r0_I) + list(v0-v0_I)
        
        # Mass constraints. For starts of stages, ensure mass aligns directly
        # with the total mass. Otherwise, match mass of previous stage's mass.
        # First phase requires the former.
        mcon_p0 = m[0] - sum([CV['stages'][k]['ms'] + CV['stages'][k]['mp'] for k in CV['phases'][0][0]])
        CONSTR_EQ = CONSTR_EQ + [mcon_p0]
        ns_p0 = len(self.phases[0][0])
        nidx = ns[0]
        for ii in range(1,P):
            ns_pii = len(self.phases[ii][0])
            if ns_pii > ns_p0:
                print("ERROR: GAINING STAGES DURING DESCENT")
            elif ns_pii < ns_p0:
                mcon_pii = m[nidx + 0] - sum([CV['stages'][k]['ms'] + CV['stages'][k]['mp'] for k in CV['phases'][ii][0]])
            elif ns_pii == ns_p0:
                mcon_pii = m[nidx + 0] - m[nidx - 1]
            nidx = nidx + ns[ii]
            CONSTR_EQ = CONSTR_EQ + [mcon_pii]
            ns_p0 = len(self.phases[ii][0])
            
        # Ending state constraints
        rf = array([Rx[-1],Ry[-1],Rz[-1]])
        vf = array([Vx[-1],Vy[-1],Vz[-1]])
        CONSTR_EQ = CONSTR_EQ + list(rf-rt_I) + list(vf-vt_I)
        
        # Phase boundary state constraints
        nidx = ns[0]
        for ii in range(0,P-1):
            r0 = array([Rx[nidx-1],Ry[nidx-1],Rz[nidx-1]])
            rf = array([Rx[nidx],Ry[nidx],Rz[nidx]])
            CONSTR_EQ = CONSTR_EQ + list(r0-rf)
            nidx = nidx + ns[ii+1]
        nidx = ns[0]
        for ii in range(0,P-1):    
            v0 = array([Vx[nidx-1],Vy[nidx-1],Vz[nidx-1]])
            vf = array([Vx[nidx],Vy[nidx],Vz[nidx]])
            CONSTR_EQ = CONSTR_EQ + list(v0-vf)
            nidx = nidx + ns[ii+1]
            
        # Path constraints - position
        idx0 = 0
        Rxp = []
        Ryp = []
        Rzp = []
        for ii in range(0,P):
            dt = tofs[ii]/ns[ii]
            Rxp = Rxp + [(Rx[idx0+k+1]-Rx[idx0+k]) - 0.5*dt*(Vx[idx0+k]+Vx[idx0+k+1]) for k in range(0,ns[ii]-1)]
            Ryp = Ryp + [(Ry[idx0+k+1]-Ry[idx0+k]) - 0.5*dt*(Vy[idx0+k]+Vy[idx0+k+1]) for k in range(0,ns[ii]-1)]
            Rzp = Rzp + [(Rz[idx0+k+1]-Rz[idx0+k]) - 0.5*dt*(Vz[idx0+k]+Vz[idx0+k+1]) for k in range(0,ns[ii]-1)]
            idx0 = idx0+ ns[ii]
        CONSTR_EQ = CONSTR_EQ + Rxp + Ryp + Rzp
        
        # Path constraints - velocity
        idx0 = 0
        Vxp = []
        Vyp = []
        Vzp = []
        for ii in range(0,P):
            Tm = self.Tmaxs[self.phases[ii][1]]
            dt = tofs[ii]/ns[ii]
            for k in range(0,ns[ii]-1):
                idx = k + idx0
                Ridx = (Rx[idx]**2.0 + Ry[idx]**2.0 + Rz[idx]**2.0)**0.5
                Ridxp1 = (Rx[idx+1]**2.0 + Ry[idx+1]**2.0 + Rz[idx+1]**2.0)**0.5
                Vxp = Vxp + [ (Vx[idx+1]-Vx[idx]) - 0.5*dt*( (Tm*Eta[idx]*Ux[idx]/m[idx]) + (Tm*Eta[idx+1]*Ux[idx+1]/m[idx+1]) + (-mu/Ridx**3.0)*Rx[idx] + (-mu/Ridxp1**3.0)*Rx[idx+1] ) ]
                Vyp = Vyp + [ (Vy[idx+1]-Vy[idx]) - 0.5*dt*( (Tm*Eta[idx]*Uy[idx]/m[idx]) + (Tm*Eta[idx+1]*Uy[idx+1]/m[idx+1]) + (-mu/Ridx**3.0)*Ry[idx] + (-mu/Ridxp1**3.0)*Ry[idx+1] ) ]
                Vzp = Vzp + [ (Vz[idx+1]-Vz[idx]) - 0.5*dt*( (Tm*Eta[idx]*Uz[idx]/m[idx]) + (Tm*Eta[idx+1]*Uz[idx+1]/m[idx+1]) + (-mu/Ridx**3.0)*Rz[idx] + (-mu/Ridxp1**3.0)*Rz[idx+1] ) ]
            idx0 = idx0+ ns[ii]
        CONSTR_EQ = CONSTR_EQ + Vxp + Vyp + Vzp
        
        # Path constraints - mass
        idx0 = 0
        mp = []
        for ii in range(0,P):
            Tm = self.Tmaxs[self.phases[ii][1]]
            dt = tofs[ii]/ns[ii]
            isp = self.isps[self.phases[ii][1]]
            mp = mp + [ (m[idx0+k] - m[idx0+k+1]) - 0.5*dt*Tm/(g0*isp)*( Eta[idx0+k] + Eta[idx0+k+1] ) for k in range(0,ns[ii]-1) ]
            idx0 = idx0 + ns[ii]
        CONSTR_EQ = CONSTR_EQ + mp
        
        # Control vector magnitudes
        Um = [ Ux[k]**2.0 + Uy[k]**2.0 + Uz[k]**2.0 - 1.0 for k in range(0,npt) ]
        CONSTR_EQ = CONSTR_EQ + Um
        
        # Radius magnitude values - above surface!
        # Rmag = [ -(Rx[k]**2.0 + Ry[k]**2.0 + Rz[k]**2.0 - self.R_eq**2.0)/self.R_eq**2.0 for k in range(0,npt) ]
        # CONSTR_INEQ = CONSTR_INEQ + Rmag
        
        # Final mass inequality constraint. For this one, add up all mass of
        # current phase stages, and subtract out the prop mass of the stage 
        # currently in use. That is the minimum allowed mass at the end of
        # this specific phase.
        idx0 = 0
        mcon_final = []
        for ii in range(0,P):
            mf_min = sum([CV['stages'][k]['ms'] + CV['stages'][k]['mp'] for k in CV['phases'][ii][0]]) - CV['stages'][ CV['phases'][ii][0][0] ]['mp']
            mcon_final = mcon_final + [ mf_min - m[idx0 + ns[ii] - 1] ]
            idx0 = idx0 + ns[ii]
        CONSTR_INEQ = CONSTR_INEQ + mcon_final
        
        # Throttle inequality constraints and time constraints
        idx0 = 0
        eta_lb_con = []
        eta_ub_con = []
        T_lb_con = []
        T_ub_con = []
        for ii in range(0,P):
            eta_lb_con = eta_lb_con + [Eta_lbs[ii] - Eta[idx0+k] for k in range(0,ns[ii])]
            eta_ub_con = eta_ub_con + [Eta[idx0+k] - Eta_ubs[ii] for k in range(0,ns[ii])]
            T_lb_con = T_lb_con + [T_lbs[ii] - tofs[ii]]
            T_ub_con = T_ub_con + [tofs[ii] - T_ubs[ii]]
            idx0 = idx0 + ns[ii]
        CONSTR_INEQ = CONSTR_INEQ + eta_lb_con + eta_ub_con + [nu_lb - nu] + [nu - nu_ub] + T_lb_con + T_ub_con
        
        # Summarize output
        if write_sum == True:
            print("Total TOF:       %10.3f sec"%(sum(tofs)))
            print("    TOF 1:       %10.3f sec"%(tofs[0]))
            print("    TOF 2:       %10.3f sec"%(tofs[1]))
            print("    TOF 3:       %10.3f sec"%(tofs[2]))
            print("       nu:       %10.3f deg"%(nu))
            print("Final Mass:      %10.3f kg"%(m[-1]))
            print("    Mf ST1:      %10.3f kg"%(m[ns[0]-1] - 8000 - 1000 - 5500))
        
        alt = array([norm([Rx[k], Ry[k], Rz[k]]) - self.R_eq for k in range(0,npt) ])
        if write_csv == True:
            fout = open(self.C['opt']['output']['file_csv_out'], 'w+')
            fout.write('Time,Alt,Eta,Mass,rLSx,rLSy,rLSz,vLSx,vLSy,vLSz\n')
            t_arr = list(linspace(0,tofs[0],ns[0])) + \
                list(linspace(tofs[0],tofs[0]+tofs[1],ns[1])) + \
                list(linspace(tofs[0]+tofs[1],tofs[0]+tofs[1]+tofs[2],ns[2]))
            for ii in range(0,npt):
                outstr = ","
                outstr = outstr.join([ '%.20f'%(ll) for ll in [t_arr[ii],alt[ii],Eta[ii],m[ii],\
                                      Rx[ii]-rt_I[0],Ry[ii]-rt_I[1],Rz[ii]-rt_I[2],\
                                      Vx[ii]-vt_I[0],Vy[ii]-vt_I[1],Vz[ii]-vt_I[2] ] ]+ ["\n"] )
                fout.write(outstr)
            fout.close()
        
        # Plot trajectory output
        if plot_traj == 1:
            
            # Throttle
            plt.figure(1)
            t0 = 0.0
            idx0 = 0
            cols = ['*-r','*-b','*-g','*-k','*-m']
            for ii in range(0,P):
                tarr = linspace(t0, t0 + tofs[ii], ns[ii])
                Etas = Eta[idx0:(idx0+ns[ii])]
                plt.plot(tarr,Etas,cols[ii])
                idx0 = idx0 + ns[ii]
                t0 = t0 + tofs[ii]
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5')
            plt.grid(which='minor', linestyle=':', linewidth='0.5')
            plt.xlabel('Time (sec)')
            plt.ylabel('Throttle (-)')
            
            # Mass
            plt.figure(2)
            t0 = 0.0
            idx0 = 0
            for ii in range(0,P):
                tarr = linspace(t0, t0 + tofs[ii], ns[ii])
                ms = m[idx0:(idx0+ns[ii])]
                plt.plot(tarr,ms,cols[ii])
                idx0 = idx0 + ns[ii]
                t0 = t0 + tofs[ii]
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5')
            plt.grid(which='minor', linestyle=':', linewidth='0.5')
            plt.xlabel('Time (sec)')
            plt.ylabel('Mass (kg)')
            
            # Altitude
            
            plt.figure(3)
            t0 = 0.0
            idx0 = 0
            cols = ['*-r','*-b','*-g']
            for ii in range(0,P):
                tarr = linspace(t0, t0 + tofs[ii], ns[ii])
                alts = alt[idx0:(idx0+ns[ii])]
                plt.plot(tarr,0.001*alts,cols[ii])
                idx0 = idx0 + ns[ii]
                t0 = t0 + tofs[ii]
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5')
            plt.grid(which='minor', linestyle=':', linewidth='0.5')
            plt.xlabel('Time (sec)')
            plt.ylabel('Altitude (km)')
            
            fig = plt.figure(4)
            ax = fig.gca(projection='3d')
            ax.set_xlim(-400000,400000)
            ax.set_ylim(-40000,40000)
            ax.set_zlim(-100000,100000)
            plt.title('Position (Landing Site)')
            X = Rx - rt_I[0]
            Y = Ry - rt_I[1]
            Z = Rz - rt_I[2]
            ax.plot(Y[0:ns[0]], Z[0:ns[0]], X[0:ns[0]],'*-r')
            ax.plot(Y[(ns[0]):(ns[0]+ns[1])], Z[(ns[0]):(ns[0]+ns[1])], X[(ns[0]):(ns[0]+ns[1])],'*-b')
            ax.plot(Y[(ns[0]+ns[1]):(ns[0]+ns[1]+ns[2])], Z[(ns[0]+ns[1]):(ns[0]+ns[1]+ns[2])], X[(ns[0]+ns[1]):(ns[0]+ns[1]+ns[2])],'*-g')
            for ii in range(0,npt):
                fac = 200000
                Xs = [Rx[ii]- rt_I[0], Rx[ii]- rt_I[0] + fac*Ux[ii]*Eta[ii]**2.0]
                Ys = [Ry[ii]- rt_I[1], Ry[ii]- rt_I[1] + fac*Uy[ii]*Eta[ii]**2.0]
                Zs = [Rz[ii]- rt_I[2], Rz[ii]- rt_I[2] + fac*Uz[ii]*Eta[ii]**2.0]
                ax.plot(Ys,Zs,Xs,'k')
                
            plt.show()
        
        return OBJVAL + CONSTR_EQ + CONSTR_INEQ

# TODO: Add omega x r terms for rotating planet
def run_problem5(config_file):
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
    CO = C['opt']
    
    # Problem and optimization configuration
    udp = prob_3D_lander_multiphase(C)
    prob = pg.problem(udp)
    prob.c_tol = opt_config['c_tol']
    uda = pg.algorithm(pg.nlopt(opt_config['nlopt_alg']))
    uda.set_verbosity(opt_config['verbosity'])
    uda.extract(pg.nlopt).xtol_rel = opt_config['xtol_rel']
    uda.extract(pg.nlopt).ftol_rel = opt_config['ftol_rel']
    uda.extract(pg.nlopt).maxeval = opt_config['maxeval']
    
    # Initial guess from file.
    if opt_in['guess_from_file']:
        print('LOADING GUESS FROM FILE')
        
        # Load in the file state vector
        Xdata_in = json.load(open(opt_in['in_file']))
        file_npts = Xdata_in['npts']
        ns = file_npts
        P = len(file_npts)
        x = Xdata_in['X0']
        
        # Other important information
        npts = opt_config['npts']

        Rx = []
        Ry = []
        Rz = []
        Vx = []
        Vy = []
        Vz = []
        Ux = []
        Uy = []
        Uz = []
        m = []
        Eta = []
        nu = 0.0
        T = []

        # Split up the state vector into a more meaningful shape
        i0 = 0
        for ii in range(0,P):
            Rx = Rx + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            Ry = Ry + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            Rz = Rz + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            Vx = Vx + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            Vy = Vy + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            Vz = Vz + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            Ux = Ux + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            Uy = Uy + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            Uz = Uz + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            m  = m  + [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        for ii in range(0,P):
            Eta= Eta+ [ list ( x[ (i0):(i0+ns[ii]) ] ) ]
            i0 = i0 + ns[ii]
        nu = [x[i0]]
        i0 = i0 + 1
        T = x[(i0):(i0 + P)]

        for pp in range(0,len(file_npts)):
            
            # If number of points in phase does not line up:
            if not (file_npts[pp] == npts[pp]):
                
                Rx[pp] = recast_pts(Rx[pp],npts[pp])
                Ry[pp] = recast_pts(Ry[pp],npts[pp])
                Rz[pp] = recast_pts(Rz[pp],npts[pp])
                Vx[pp] = recast_pts(Vx[pp],npts[pp])
                Vy[pp] = recast_pts(Vy[pp],npts[pp])
                Vz[pp] = recast_pts(Vz[pp],npts[pp])
                Ux[pp] = recast_pts(Ux[pp],npts[pp])
                Uy[pp] = recast_pts(Uy[pp],npts[pp])
                Uz[pp] = recast_pts(Uz[pp],npts[pp])
                m[pp] = recast_pts(m[pp],npts[pp])
                Eta[pp] = recast_pts(Eta[pp],npts[pp])
        
        Rx = list(chain.from_iterable(Rx))
        Ry = list(chain.from_iterable(Ry))
        Rz = list(chain.from_iterable(Rz))
        Vx = list(chain.from_iterable(Vx))
        Vy = list(chain.from_iterable(Vy))
        Vz = list(chain.from_iterable(Vz))
        Ux = list(chain.from_iterable(Ux))
        Uy = list(chain.from_iterable(Uy))
        Uz = list(chain.from_iterable(Uz))
        m  = list(chain.from_iterable(m))
        Eta= list(chain.from_iterable(Eta))
        
        X0 = Rx + Ry + Rz + Vx + Vy + Vz + Ux + Uy + Uz + m + Eta + nu + T
        
    # Initial guess - linear profile
    elif opt_in['use_linear_guess']:
        print('CONSTRUCTING LINEAR INITIAL GUESS')
        
        opt_lg = opt_in['linear_guess']
        npts = sum(opt_config['npts'])
        ns = opt_config['npts']
        tof = sum(opt_lg['tof'])
        nu0 = opt_lg['nu0']
        mass0 = sum([CV['stages'][ii]['ms'] + CV['stages'][ii]['mp'] for ii in CV['phases'][0][0]])
        isp0 = CV['engines'][CV['phases'][0][1]]['isp']
        tmax0 = CV['engines'][CV['phases'][0][1]]['Tmax']
        
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
        Ux = [1.0] + [(V[0,k+1] - V[0,k])/dt for k in range(0,npts-1)]
        Uy = [1.0] + [(V[1,k+1] - V[1,k])/dt for k in range(0,npts-1)]
        Uz = [1.0] + [(V[2,k+1] - V[2,k])/dt for k in range(0,npts-1)]
        for kk in range(0,len(Ux)):
            Umag = (Ux[kk]**2.0 + Uy[kk]**2.0 + Uz[kk]**2.0 )**0.5
            Ux[kk] = Ux[kk]/Umag
            Uy[kk] = Uy[kk]/Umag
            Uz[kk] = Uz[kk]/Umag 
        U = array([Ux, Uy, Uz])
        Eta = [0.7]*ns[0] + [0.0]*ns[1] + [0.7]*ns[2]
        m = [mass0]
        for k in range(1,npts):
            m = m + [m[k-1] - tmax0*Eta[k]/(9.81*isp0)]
        X0 = list(R.flatten()) + list(V.flatten()) + list(U.flatten()) + list(m) + list(Eta) + [nu0] + opt_lg['tof']

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
        outdict = { 'npts':list(opt_config['npts']), 'X0':list(pop.champion_x) }
        if (not opt_out['only_write_feasible']) or (opt_out['only_write_feasible'] and is_feas):
            json.dump(outdict, open(opt_out['file_X_out'],'w+'), indent=4)
        

    


