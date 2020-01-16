#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 09:48:16 2019

MULTI-PHASE 3-Dimensional full gravity optimization problem.
This should match optimality!!!!

@author: jasonmeverett
"""

from numpy import *
import matplotlib.pyplot as plt
import pygmo as pg
from scipy.linalg import norm
from ananke.orbit import *
from ananke.frames import *
from ananke.util import *
from ananke.planets import *
from mpl_toolkits.mplot3d import Axes3D
import time
import json
from itertools import chain
from typing import List
from copy import deepcopy
import AnankeC as ac

class ColX():
    def __init__(self,R=zeros(3),V=zeros(3),m=0.0):
        self.R = array(R)
        self.V = array(V)
        self.m = m
    def to_array(self):
        return array( list(self.R) + list(self.V) + [self.m] )
    def __str__(self):
        os = "\t{\n"
        os = os + "\t\t\"R\":[%.3f, %.3f, %.3f],\n"%(self.R[0],self.R[1],self.R[2])
        os = os + "\t\t\"V\":[%.3f, %.3f, %.3f],\n"%(self.V[0],self.V[1],self.V[2])
        os = os + "\t\t\"m\": %.3f\n"%(self.m)
        os = os + "\t}"
        return os
    
class ColU():
    def __init__(self,u=zeros(3),eta=0.0):
        self.u = array(u)
        self.eta = eta
    def to_array(self):
        return array( list(self.u) + [self.eta] ) 
    def __str__(self):
        os = "\t{\n"
        os = os + "\t\t\"u\": [%.3f, %.3f, %.3f],\n"%(self.u[0],self.u[1],self.u[2])
        os = os + "\t\t\"eta\": %.3f\n"%(self.eta)
        os = os + "\t}"
        return os
    
class ColPt():
    def __init__(self,X=ColX(),U=ColU()):
        self.X = X
        self.U = U
    def __str__(self):
        os = "{\n"
        os = os + "\t\"X\":\n"
        os = os + str(self.X) + ",\n"
        os = os + "\t\"U\":\n"
        os = os + str(self.U) + "\n"
        os = os + "}"
        return os

class TrajLeg():
    def __init__(self,ColPts=[],T=0):
        self.ColPts = ColPts
        self.T = T
    def __str__(self):
        os = "{\n"
        os = os + "\"T\": %.3f,\n"%(self.T)
        os = os + "\"ColPts\":\n"
        os = os + "[\n"
        for cp in self.ColPts:
            os = os + str(cp) + ",\n"
        os = os[0:-2] + "\n]\n" + "}"
        return os

class OptTraj():
    def __init__(self,TrajLegs = [], nu0 = 0.0):
        self.TrajLegs = TrajLegs
        self.nu0 = nu0
    def __str__(self):
        os = "{\n"
        os = os + "\"nu0\": %.3f,\n"%(self.nu0)
        os = os + "\"TrajLegs\":\n"
        os = os + "[\n"
        for ii, tl in enumerate(self.TrajLegs):
            os = os + str(tl)
            os = os + ",\n"
        os = os[0:-2] + "\n" + "]\n}\n"
        return os

def ConstructDV(OT : OptTraj):
    X = []
    for TrajLeg in OT.TrajLegs:
        for ColPt in TrajLeg.ColPts:
            X = X + list(ColPt.X.R)
            X = X + list(ColPt.X.V)
            X = X + [ColPt.X.m]
            X = X + list(ColPt.U.u)
            X = X + [ColPt.U.eta]
    X = X + [OT.nu0]
    for TrajLeg in OT.TrajLegs:
        X = X + [TrajLeg.T]
    return(X)

def ConstructOptTraj(X,ns):
    idx = 0
    P = len(ns)
    X_short = X[0:(-1-P)]
    tls = []
    for ii in range(0,P):
        cps = []
        for kk in range(0,ns[ii]):
            cx = deepcopy(ColX(X[idx:idx+3],X[idx+3:idx+6],X[idx+6]))
            cu = deepcopy(ColU(X[idx+7:idx+10],X[idx+10]))
            cp = deepcopy(ColPt(cx,cu))
            cps.append(cp)
            idx = idx + 11
        tl = deepcopy(TrajLeg(cps,X[-1-P+ii+1]))
        tls.append(tl)
    OT = deepcopy(OptTraj(tls,X[-P-1]))
    return OT

def OptTrajFromJson(OTd):
    OT = deepcopy(OptTraj())
    OT.nu0 = OTd['nu0']
    for ii in range(0,len(OTd['TrajLegs'])):
        tld = OTd['TrajLegs'][ii]
        tl = deepcopy(TrajLeg())
        tl.T = tld['T']
        for k in range(0, len(tld['ColPts'])):
            cpd = tld['ColPts'][k]
            cp = deepcopy(ColPt())
            cp.X.R = array(cpd['X']['R'])
            cp.X.V = array(cpd['X']['V'])
            cp.X.m = cpd['X']['m']
            cp.U.u = array(cpd['U']['u'])
            cp.U.eta = cpd['U']['eta']
            tl.ColPts.append(cp)
        OT.TrajLegs.append(tl)
    return OT

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

# Gravity vector
def grav(R, mu=4902799000000.0):
    R = array(R)
    return -mu/norm(R)**3.0*R

# Equations of motion
def f(X, U, mu=4902799000000.0, Tm=66000.0, isp=450.0):
    g0 = 9.81
    Xdot = zeros(7)
    Xdot[0:3] = X[3:6]
    Xdot[3:6] = grav(X[0:3],mu) + Tm*U[3]/X[6]*array(U[0:3])
    Xdot[6] = -Tm*U[3]/(g0*isp)
    return Xdot

# Partial of f wrt. X
def dfdX(X, U, mu=4902799000000.0, Tm=66000.0, isp=450.0):
    
    Rk = X[0:3]
    Vk = X[3:6]
    m = X[6]
    u = U[0:3]
    eta = U[3]
    
    I = eye(3)
    A = zeros((7,7))
    A[0:3,3:6] = I
    A[3:6,6] = -Tm*eta/(m**2.0)*u
    A[3:6,0:3] = -mu/norm(Rk)**3.0*I + 3*mu/norm(Rk)**5.0 * outer(Rk,Rk)

    return A

# Partial of f wrt. U
def dfdU(X, U, mu=4902799000000.0, Tm=66000.0, isp=450.0):
    
    m = X[6]
    u = U[0:3]
    eta = U[3]
    g0 = 9.81
    
    I = eye(3)
    A = zeros((7,4))
    A[3:6,0:3] = Tm*eta/m*I
    A[3:6,3] = Tm/m*u
    A[6,3] = -Tm/(g0*isp)
    
    return A
    
# ----------------------------
# Initial boundary constraints
# ----------------------------
   
# Returns a 7x1 vector
def g0_1(OT : OptTraj, C : dict):
    
    # Vehicle definition
    CV = C['vehicle']
    CP = C['planet']
    
    ra = CP['R_eq'] + CV['orbit']['alta']
    rp = CP['R_eq'] + CV['orbit']['altp']
    sma = 0.5*(ra+rp)
    ecc = (ra-rp)/(ra+rp)
    inc = CV['orbit']['inc']
    Om = CV['orbit']['raan']
    om = CV['orbit']['argper']
    
    # Initial true anomaly
    nu = OT.nu0
    
    # Initial State equality constraint
    r0_I,v0_I = elts_to_rv(sma, ecc, inc, Om, om, nu, CP['mu'], CV['orbit']['degrees'])
    
    # Calculate constraint performances.
    RVcon = list(OT.TrajLegs[0].ColPts[0].X.R - r0_I) + list(OT.TrajLegs[0].ColPts[0].X.V - v0_I)
    mcon = OT.TrajLegs[0].ColPts[0].X.m - sum([CV['stages'][k]['ms'] + CV['stages'][k]['mp'] for k in CV['phases'][0][0]])
    
    # Return R,V,m constraint information
    Xcon  = RVcon + [mcon]
    return Xcon

def g0_2(OT : OptTraj, C : dict):
    return None

def g0_3(OT : OptTraj, C : dict):
    return None

# ----------------------------
# Final boundary constraints
# ----------------------------

def gf_1(OT : OptTraj, C : dict):
    return None

def gf_2(OT : OptTraj, C : dict):
    return None

def gf_3(OT : OptTraj, C : dict):
    return None
    
# -----------------------------
# Boundary constraint functions
# -----------------------------
   
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

class prob_MPlander(object):
    
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
        npt = sum(self.npts)
        num_nec = int(3 + 3 + P + 3 + 3 + 2*3*(P-1) + 2*3*(npt-P) + (npt-P) + npt)
        return num_nec
    
    def get_nic(self):
        P = self.nphases
        npt = sum(self.npts)
        # return int(npts_tot + P + 2*npts_tot + P + P + 1 + 1)
        return int(P + 2*npt + 1 + 1 + P + P)  # Removed alt constraint
    
    def get_bounds(self):
        nt = sum(self.npts)
        P = len(self.npts)
        
        sr = 2.0
        sv = 2.0
        su = 2.0
        se = 2.0
        Rmin = -sr*array([self.R_eq, self.R_eq, self.R_eq])
        Vmin = -sv*array([sqrt(self.mu/self.R_eq), sqrt(self.mu/self.R_eq), sqrt(self.mu/self.R_eq)])
        mmin = 0.0
        Umin = -su*array([1.0, 1.0, 1.0])
        Etamin = -se*1.0
        LB = (list(Rmin) + list(Vmin) + [mmin] + list(Umin) + [Etamin])*nt + [-60] + [0]*P 

        Rmax = sr*array([self.R_eq, self.R_eq, self.R_eq])
        Vmax = sv*array([sqrt(self.mu/self.R_eq), sqrt(self.mu/self.R_eq), sqrt(self.mu/self.R_eq)])
        mmax = self.mass0
        Umax = su*array([1.0, 1.0, 1.0])
        Etamax = se*1.0
        UB = (list(Rmax) + list(Vmax) + [mmax] + list(Umax) + [Etamax])*nt + [30] + [10000]*P 

        return (LB, UB)
    
    def fitness(self, x):
        return self.run_traj(x, plot_traj=0)
    
    def summary(self, x):
        return self.run_traj(x, plot_traj=1)
    
    def gradient(self, x):
        P = self.nphases
        npcs = sum([n-1 for n in self.npts])
        npt = sum(self.npts)
        ns = self.npts
        nt = sum(ns)
        mu = self.mu
        g0 = self.g0
        Eta_lbs = self.Eta_lbs
        Eta_ubs = self.Eta_ubs
        T_lbs = self.T_lbs
        T_ubs = self.T_ubs
        nu_lb = self.nu_lb
        nu_ub = self.nu_ub
        Om = self.plOm
        
        # Construct collocation trajectory from DV
        OT = ConstructOptTraj(x,ns)
        nu = OT.nu0
        tof_tot = sum([a.T for a in OT.TrajLegs])
        
        # Array shape
        len_y = 1 + self.get_nec() + self.get_nic()
        len_x = len(x)
        arr_shape = ((len_y,len_x))
        
        # Estimated gradient
        # t1e = time.clock()
        # grade = pg.estimate_gradient_h(lambda x: self.fitness(x), x, 1e-5)
        # arre = grade.reshape(arr_shape)
        # t2e = time.clock()
        # t1g = time.clock()
        
        # Real gradient
        grad = zeros(arr_shape)
        
        # Cost function partials
        i0x = 0
        i0y = 0
        i0t = 11*nt+1
        if self.objtype == 'control':
            # i0x = i0x + 11*ns[0]
            # i0x = i0x + 11*ns[1]
            # i0t = i0t + 1
            # i0t = i0t + 1
            for ii in range(0,P):
                tl = OT.TrajLegs[ii]
                dt = tl.T/ns[ii]
                Tm = self.Tmaxs[self.phases[ii][1]]
                J = sum([0.5*dt*((Tm*tl.ColPts[k].U.eta/tl.ColPts[k].X.m)**2.0 + (Tm*tl.ColPts[k+1].U.eta/tl.ColPts[k+1].X.m)**2.0)  for k in range(0, ns[ii]-1)])
                dJ_dm = [ -dt*Tm**2.0*tl.ColPts[k].U.eta**2.0/tl.ColPts[k].X.m**3.0 for k in range(0,ns[ii]) ]
                dJ_dm[1:-1] = [2*a for a in dJ_dm[1:-1]]
                dJ_dEta = [ dt*Tm**2.0*tl.ColPts[k].U.eta/tl.ColPts[k].X.m**2.0 for k in range(0, ns[ii]) ]
                dJ_dEta[1:-1] = [2*a for a in dJ_dEta[1:-1]]
                dJ_dT = J/tl.T
                for k in range(0,ns[ii]):
                    grad[0,(i0x+6+11*k)] = dJ_dm[k]
                    grad[0,(i0x+10+11*k)] = dJ_dEta[k]
                grad[0,i0t] = J/tl.T
                i0x = i0x + 11*ns[ii]
                i0t = i0t + 1
                
        if self.objtype == 'fuel':
            i0x = i0x + 11*ns[0]
            i0x = i0x + 11*ns[1]
            i0t = i0t + 1
            i0t = i0t + 1
            for ii in range(P-1,P):
                tl = OT.TrajLegs[ii]
                dt = tl.T/ns[ii]
                J = sum([0.5*dt*(tl.ColPts[k].U.eta + tl.ColPts[k+1].U.eta) for k in range(0, ns[ii]-1)])
                dJ_dT = J/tl.T
                dJ_dEta = 0.5*dt*ones(ns[ii])
                dJ_dEta[1:-1] = [2*a for a in dJ_dEta[1:-1]]
                for k in range(0,ns[ii]):
                    grad[0,(i0x+10+11*k)] = dJ_dEta[k]
                grad[0,i0t] = J/tl.T
                i0x = i0x + 11*ns[ii]
                i0t = i0t + 1
                
        # print(norm(grad[0,:]-arre[0,:]))
        # print(grad[0,:])
        # print(arre[0,:])
        # quit()
                
        # Initial state constraint (with mass!)
        i0y = 1
        grad[i0y:(i0y+7),0:7] = eye(7)
        
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
        grad[1:4,11*nt] = -dr0_dnu
        grad[4:7,11*nt] = -dv0_dnu
        
        # Mass constraints. For starts of stages, ensure mass aligns directly
        # with the total mass. Otherwise, match mass of previous stage's mass.
        # First phase requires the former.
        ns_p0 = len(self.phases[0][0])
        i0 = ns[0]
        i0m = 1
        i0x = 11*ns[0]
        for ii in range(1,P):
            ns_pii = len(self.phases[ii][0])
            if ns_pii > ns_p0:
                print("ERROR: GAINING STAGES DURING DESCENT")
            elif ns_pii < ns_p0:
                grad[7+i0m,6+i0x] = 1.0
            elif ns_pii == ns_p0:
                grad[7+i0m,6+i0x] = 1.0
                grad[7+i0m,6+i0x-11] = -1.0
            i0 = i0 + ns[ii]
            i0m = i0m + 1
            i0x = i0x + 11*ns[ii]
            ns_p0 = len(self.phases[ii][0])

        # Final position/velocity constraints (without mass!)
        i0y = 7 + P
        i0x = 11*(nt-1)
        grad[i0y:(i0y+6),i0x:(i0x+6)] = eye(6)
        
        # Rotating planet for final position/velocity constraints wrt. time
        R_I_PF = Rot_I_PF(self.plOm, self.ep0, tof_tot)
        R_UEN_I = R_I_PF.inv() * self.R_UEN_PF
        rLS_I = R_UEN_I.apply([self.R_eq + self.LS_alt,0,0])
        rt_I = rLS_I + R_UEN_I.apply(self.rt_LS)
        vt_I = R_UEN_I.apply(self.vt_LS) + cross([0,0,self.plOm],rt_I)
        Omvec = array([0,0,Om])
        dRf_dt = -cross(Omvec, rt_I)
        dVf_dt = -cross(Omvec, vt_I)
        i0y = 7 + P
        i0t = 11*nt+1
        for ii in range(0,P):
            grad[(i0y):(i0y+3),i0t] = dRf_dt
            grad[(i0y+3):(i0y+6),i0t] = dVf_dt
            i0t = i0t + 1
        
        # Phase boundary position/velocity constraints
        i0y = 7 + P + 6
        i0x = 0
        for ii in range(0,P-1):
            grad[(i0y):(i0y+3),(i0x+11*(ns[ii]-1)):(i0x+11*(ns[ii]-1)+3)] = eye(3)
            grad[(i0y):(i0y+3),(i0x+11*(ns[ii])):(i0x+11*(ns[ii])+3)] = -eye(3)
            grad[(i0y+3*(P-1)):(i0y+3*(P-1)+3),(i0x+11*(ns[ii]-1)+3):(i0x+11*(ns[ii]-1)+6)] = eye(3)
            grad[(i0y+3*(P-1)):(i0y+3*(P-1)+3),(i0x+11*(ns[ii])+3):(i0x+11*(ns[ii])+6)] = -eye(3)
            i0y = i0y + 3
            i0x = i0x + 11*ns[ii]
            
        # Path constraints
        i0y = 7 + P + 6 + 6*(P-1)
        i0x = 0
        i0t = 0
        for ii in range(0,P):
            dt = OT.TrajLegs[ii].T/ns[ii]
            Tm = self.Tmaxs[self.phases[ii][1]]
            isp = self.isps[self.phases[ii][1]]
            
            for k in range(0,ns[ii]-1):
                Xk = OT.TrajLegs[ii].ColPts[k].X.to_array()
                Uk = OT.TrajLegs[ii].ColPts[k].U.to_array()
                fk = f(Xk, Uk, mu, Tm, isp)
                Xkp1 = OT.TrajLegs[ii].ColPts[k+1].X.to_array()
                Ukp1 = OT.TrajLegs[ii].ColPts[k+1].U.to_array()
                fkp1 = f(Xkp1, Ukp1, mu, Tm, isp)
                Uc = 0.5*(Uk+Ukp1)
                Xc = 0.5*(Xk+Xkp1) + dt/8*(fk-fkp1)
                fc = f(Xc, Uc, mu, Tm, isp)
                Ak = dfdX(Xk,Uk, mu, Tm, isp)
                Akp1 = dfdX(Xkp1,Ukp1, mu, Tm, isp)
                Ac = dfdX(Xc,Uc, mu, Tm, isp)
                Bk = dfdU(Xk,Uk, mu, Tm, isp)
                Bkp1 = dfdU(Xkp1,Ukp1, mu, Tm, isp)
                Bc = dfdU(Xc,Uc, mu, Tm, isp)
                I = eye(7,7)
                
                dDel_dXk = I + dt/6*(Ak + 4*matmul(Ac, (0.5*I + dt/8*Ak) ))
                dDel_dXkp1 = -I + dt/6*(Akp1 + 4*matmul( Ac, (0.5*I - dt/8*Akp1) ) )
                dDel_dUk = dt/6*(Bk + 4*(matmul(Ac, dt/8*Bk) + 1/2*Bc) )
                dDel_dUkp1 = dt/6*(Bkp1 + 4*(matmul(Ac, -dt/8*Bkp1) + 1/2*Bc) )
                dDel_dT = 1/(6*ns[ii])*(fk + 4*fc + fkp1 + dt/2*matmul(Ac, fk-fkp1))
                
                grad[(i0y+0):(i0y+7),(i0x+0):(i0x+7)] = dDel_dXk
                grad[(i0y+0):(i0y+7),(i0x+7):(i0x+11)] = dDel_dUk
                grad[(i0y+0):(i0y+7),(i0x+11):(i0x+18)] = dDel_dXkp1
                grad[(i0y+0):(i0y+7),(i0x+18):(i0x+22)] = dDel_dUkp1
                grad[(i0y+0):(i0y+7),11*nt+1+i0t] = dDel_dT
                
                i0y = i0y + 7
                i0x = i0x + 11

            i0x = i0x + 11    
            i0t = i0t + 1
                
        # Control vector magnitude constraint
        i0y = 7 + P + 6 + 6*(P-1) + 7*(nt-P)   
        i0x = 0
        for ii in range(0,P):
            for k in range(0,ns[ii]):
                grad[(i0y),(i0x+7):(i0x+10)] = 2*OT.TrajLegs[ii].ColPts[k].U.u
                i0y = i0y + 1
                i0x = i0x + 11
        
        # Final mass inequality constraint
        i0y = 7 + P + 6 + 6*(P-1) + 7*(nt-P) + nt
        i0x = 0
        for ii in range(0,P):
            grad[i0y,(i0x + 11*ns[ii] - 5)] = -1.0
            i0y = i0y + 1
            i0x = i0x + 11*ns[ii]
        
        # Throttle limit LB inequality constraints
        i0y = 7 + P + 6 + 6*(P-1) + 7*(nt-P) + nt + P
        i0x = 0
        for ii in range(0,P):
            for k in range(0,ns[ii]):
                grad[i0y,(i0x + 10)] = -1.0
                i0x = i0x + 11
                i0y = i0y + 1
        
        # Throttle limit UB inequality constraints
        i0y = 7 + P + 6 + 6*(P-1) + 7*(nt-P) + nt + P + nt
        i0x = 0
        for ii in range(0,P):
            for k in range(0,ns[ii]):
                grad[i0y,(i0x + 10)] = 1.0
                i0x = i0x + 11
                i0y = i0y + 1        
        
        # True anomaly constraints
        i0y = 7 + P + 6 + 6*(P-1) + 7*(nt-P) + nt + P + nt + nt
        grad[i0y,(11*nt)] = -1.0
        grad[i0y+1,(11*nt)] = 1.0
        
        # Time constraints LB
        i0y = 7 + P + 6 + 6*(P-1) + 7*(nt-P) + nt + P + nt + nt + 2
        for ii in range(0,P):
            grad[i0y+ii,(11*nt+1+ii)] = -1.0
        
        # Time constraints
        i0y = 7 + P + 6 + 6*(P-1) + 7*(nt-P) + nt + P + nt + nt + 2 + P
        for ii in range(0,P):
            grad[i0y+ii,(11*nt+1+ii)] = 1.0
        
        # i0y = 7 + P + 6 + 6*(P-1) + 7*(nt-P) + nt + P + nt + nt + 2
        # a1 = grad
        # a2 = arre
        # print(norm(a1-a2))
        # t2g = time.clock()
        # print("Estimation: %.3f sec"%(t2e-t1e))
        # print("Calculated: %.3f sec"%(t2g-t1g))
        # print("Factor: %.2f"%((t2e-t1e)/(t2g-t1g)))
        # print(a1)
        # print(a2)
        # quit()
        
        grad_rtn = grad.reshape((arr_shape[0]*arr_shape[1],))
        return grad_rtn

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
        
        # Construct the trajectory out of the DV.
        OT = ConstructOptTraj(x,ns)    
        nu = OT.nu0
        tof_tot = sum([a.T for a in OT.TrajLegs])
        
        # Cost function
        J = 0
        if self.objtype == 'control':
            for ii in range(0,P):
                tl = OT.TrajLegs[ii]
                Tm = self.Tmaxs[self.phases[ii][1]]
                dt = tl.T/ns[ii]
                for k in range(0, len(tl.ColPts)-1):
                    cp = tl.ColPts[k]
                    cp1 = tl.ColPts[k+1]
                    J = J + 0.5*dt*((Tm*cp.U.eta/cp.X.m)**2.0 + (Tm*cp1.U.eta/cp1.X.m)**2.0) 
                    
        if self.objtype == 'fuel':
            for ii in range(P-1,P):
                tl = OT.TrajLegs[ii]
                dt = tl.T/ns[ii]
                for k in range(0, len(tl.ColPts)-1):
                    cp = tl.ColPts[k]
                    cp1 = tl.ColPts[k+1]
                    J = J + 0.5*dt*(cp.U.eta + cp1.U.eta) 
        
        # Set objective value
        OBJVAL = [J]
        
        # Initial State equality constraint
        r0_I,v0_I = elts_to_rv(self.sma,self.ecc,self.inc,self.Om,self.om,nu,self.mu,self.orb_deg)
        
        # Landing site position
        R_I_PF = Rot_I_PF(self.plOm, self.ep0, tof_tot)
        R_UEN_I = R_I_PF.inv() * self.R_UEN_PF
        rLS_I = R_UEN_I.apply([self.R_eq + self.LS_alt,0,0])
        vLS_I = cross([0,0,self.plOm],rLS_I)
        rt_I = rLS_I + R_UEN_I.apply(self.rt_LS)
        vt_I = R_UEN_I.apply(self.vt_LS) + cross([0,0,self.plOm],rt_I)
        
        # Starting state constraints
        CONSTR_EQ = CONSTR_EQ + list(OT.TrajLegs[0].ColPts[0].X.R - r0_I)
        CONSTR_EQ = CONSTR_EQ + list(OT.TrajLegs[0].ColPts[0].X.V - v0_I)
        
        
        
        # Mass constraints. For starts of stages, ensure mass aligns directly
        # with the total mass. Otherwise, match mass of previous stage's mass.
        # First phase requires the former.
        mcon_p0 = OT.TrajLegs[0].ColPts[0].X.m - sum([CV['stages'][k]['ms'] + CV['stages'][k]['mp'] for k in CV['phases'][0][0]])
        CONSTR_EQ = CONSTR_EQ + [mcon_p0]
        ns_p0 = len(self.phases[0][0])
        for ii in range(1,P):
            ns_pii = len(self.phases[ii][0])
            if ns_pii > ns_p0:
                print("ERROR: GAINING STAGES DURING DESCENT")
            elif ns_pii < ns_p0:
                mcon_pii = OT.TrajLegs[ii].ColPts[0].X.m - sum([CV['stages'][k]['ms'] + CV['stages'][k]['mp'] for k in CV['phases'][ii][0]])
            elif ns_pii == ns_p0:
                mcon_pii = OT.TrajLegs[ii].ColPts[0].X.m - OT.TrajLegs[ii-1].ColPts[-1].X.m
            CONSTR_EQ = CONSTR_EQ + [mcon_pii]
            ns_p0 = len(self.phases[ii][0])
            
        # Ending state constraints
        CONSTR_EQ = CONSTR_EQ + list(OT.TrajLegs[-1].ColPts[-1].X.R - rt_I)
        CONSTR_EQ = CONSTR_EQ + list(OT.TrajLegs[-1].ColPts[-1].X.V - vt_I)
        
        # Phase boundary state constraints
        for ii in range(0,P-1):
            r0 = OT.TrajLegs[ii].ColPts[-1].X.R
            rf = OT.TrajLegs[ii+1].ColPts[0].X.R
            CONSTR_EQ = CONSTR_EQ + list(r0-rf)
        for ii in range(0,P-1):
            v0 = OT.TrajLegs[ii].ColPts[-1].X.V
            vf = OT.TrajLegs[ii+1].ColPts[0].X.V
            CONSTR_EQ = CONSTR_EQ + list(v0-vf)

        # Path Constraints
        for ii in range(0,P):
            tl = deepcopy(OT.TrajLegs[ii])
            Tm = self.Tmaxs[self.phases[ii][1]]
            isp= self.isps[self.phases[ii][1]]
            dt = tl.T/ns[ii]
            for k in range(0,len(tl.ColPts)-1):
                Xk = tl.ColPts[k].X.to_array()
                Uk = tl.ColPts[k].U.to_array()
                fk = f(Xk, Uk, mu, Tm, isp)
                Xkp1 = tl.ColPts[k+1].X.to_array()
                Ukp1 = tl.ColPts[k+1].U.to_array()
                fkp1 = f(Xkp1, Ukp1, mu, Tm, isp)
                Uc = 0.5*(Uk+Ukp1)
                Xc = 0.5*(Xk+Xkp1) + dt/8*(fk-fkp1)
                fc = f(Xc, Uc, mu, Tm, isp)
                constr_p = Xk-Xkp1 + dt/6*(fk + 4*fc + fkp1)
                CONSTR_EQ = CONSTR_EQ + list(constr_p)
            
        # Control vector magnitudes
        for tl in OT.TrajLegs:
            for cp in tl.ColPts:
                constr_umag = norm(cp.U.u)**2.0 - 1.0
                CONSTR_EQ = CONSTR_EQ + [constr_umag]
            
        # Final mass inequality constraint. For this one, add up all mass of
        # current phase stages, and subtract out the prop mass of the stage 
        # currently in use. That is the minimum allowed mass at the end of
        # this specific phase.
        mcon_final = []
        for ii in range(0,P):
            mf_min = sum([CV['stages'][k]['ms'] + CV['stages'][k]['mp'] for k in CV['phases'][ii][0]]) - CV['stages'][ CV['phases'][ii][0][0] ]['mp']
            mcon_final = mcon_final + [ mf_min - OT.TrajLegs[ii].ColPts[-1].X.m ]
        CONSTR_INEQ = CONSTR_INEQ + mcon_final
            
        # Throttle inequality constraints and time constraints
        eta_lb_con = []
        eta_ub_con = []
        T_lb_con = []
        T_ub_con = []
        for ii in range(0,P):
            eta_lb_con = eta_lb_con + [Eta_lbs[ii] - OT.TrajLegs[ii].ColPts[k].U.eta for k in range(0,ns[ii])]
            eta_ub_con = eta_ub_con + [OT.TrajLegs[ii].ColPts[k].U.eta - Eta_ubs[ii] for k in range(0,ns[ii])]
            T_lb_con = T_lb_con + [T_lbs[ii] - OT.TrajLegs[ii].T]
            T_ub_con = T_ub_con + [OT.TrajLegs[ii].T - T_ubs[ii]]
        CONSTR_INEQ = CONSTR_INEQ + eta_lb_con + eta_ub_con + [nu_lb - nu] + [nu - nu_ub] + T_lb_con + T_ub_con
 
        # Summarize output
        if write_sum == True:
            print("Total TOF:       %10.3f sec"%(tof_tot))
            for ii in range(0,P):
                print("    TOF %d:       %10.3f sec"%(ii+1,OT.TrajLegs[ii].T))
            print("Final Mass:      %10.3f kg"%(OT.TrajLegs[-1].ColPts[-1].X.m))
            print("Final st1 mass:  %10.3f kg"%(OT.TrajLegs[0].ColPts[-1].X.m - 14000 - 15000 - 8000))
            print("J %.5f"%(J))
            
        # Write to CSV file
        if write_csv == True:
            fout = open(self.C['opt']['output']['file_csv_out'], 'w+')
            fout.write('Time,Alt,Eta,Mass,rLSx,rLSy,rLSz,vLSx,vLSy,vLSz\n')
            
            t0 = 0
            for ii in range(0,P):
                dt = OT.TrajLegs[ii].T/ns[ii]
                tt = linspace(t0, t0 + OT.TrajLegs[ii].T, ns[ii])
                for k in range(0,ns[ii]):
                    r = OT.TrajLegs[ii].ColPts[k].X.R
                    v = OT.TrajLegs[ii].ColPts[k].X.V
                    m = OT.TrajLegs[ii].ColPts[k].X.m
                    eta = OT.TrajLegs[ii].ColPts[k].U.eta
                    alt = norm(r) - self.R_eq
                    t = tt[k]
                    outstr = ","
                    outstr = outstr.join(['%.20f'%(ll) for ll in [
                        t, alt, eta, m, r[0]-rLS_I[0], r[1]-rLS_I[1],r[2]-rLS_I[2],
                        vLS_I[0], v[1]-vLS_I[1],v[2]-vLS_I[2]
                        ]] + ['\n'])
                    fout.write(outstr)
                t0 = t0 + OT.TrajLegs[ii].T
            fout.close()
     
            
        if plot_traj == 1:

            # Colors
            cols = '*-r', '*-b', '*-g'
            
            # Altitude
            fig1 = plt.figure(1)
            fig2 = plt.figure(2)
            fig3 = plt.figure(3)
            ax1 = fig1.gca()
            ax1.minorticks_on()
            ax1.grid(which='major', linestyle='-', linewidth='0.5')
            ax1.grid(which='minor', linestyle=':', linewidth='0.5')
            ax2 = fig2.gca()
            ax2.minorticks_on()
            ax2.grid(which='major', linestyle='-', linewidth='0.5')
            ax2.grid(which='minor', linestyle=':', linewidth='0.5')
            ax3 = fig3.gca()
            ax3.minorticks_on()
            ax3.grid(which='major', linestyle='-', linewidth='0.5')
            ax3.grid(which='minor', linestyle=':', linewidth='0.5')
            
            t0 = 0.0
            for ii,tl in enumerate(OT.TrajLegs):
                alt = []
                eta = []
                mass = []
                tarr = list(linspace(t0, t0 + tl.T, len(tl.ColPts)))
                for jj,cp in enumerate(tl.ColPts):
                    alt.append(norm(cp.X.R)-self.R_eq)
                    eta.append(cp.U.eta)
                    mass.append(cp.X.m)
                t0 = t0 + tl.T
                ax1.plot(tarr,alt, cols[ii])
                ax2.plot(tarr,eta, cols[ii])
                ax3.plot(tarr,mass, cols[ii])
            plt.show()
        
        return OBJVAL + CONSTR_EQ + CONSTR_INEQ
        # return [0] + [0]*self.get_nec() + [0]*self.get_nic()

# TODO: Add omega x r terms for rotating planet
def run_problem6(config_file):
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
    udp = prob_MPlander(C)
    prob = pg.problem(udp)
    prob.c_tol = opt_config['c_tol']
    uda = pg.algorithm(pg.nlopt(opt_config['nlopt_alg']))
    uda.set_verbosity(opt_config['verbosity'])
    uda.extract(pg.nlopt).xtol_rel = opt_config['xtol_rel']
    uda.extract(pg.nlopt).ftol_rel = opt_config['ftol_rel']
    uda.extract(pg.nlopt).maxeval = opt_config['maxeval']
        
    # Hardcoding to linear guess to start
    npts = sum(opt_config['npts'])
    ns = opt_config['npts']
    P = len(ns)
    nu0 = CV['orbit']['ta']
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
    mu = CP['mu']
    r0_I,v0_I = elts_to_rv(sma,ecc,inc,Om,om,nu,CP['mu'],CV['orbit']['degrees'])

    # Create linear profile. Prime one step forward, because step is taken back
    # at start of each new trajectory leg.
    if opt_in['use_coast_guess']:
        TrajLegs = []
        dt = CO['input']['coast_times'][0]
        r = r0_I
        v = v0_I
        rp = r
        vp = v
        mp = mass0
        for ii in range(0,P):
            dt = CO['input']['coast_times'][ii]/ns[ii]
            isp = CV['engines'][CV['phases'][ii][1]]['isp']
            tmax = CV['engines'][CV['phases'][ii][1]]['Tmax']
            r = rp
            v = vp
            m = mp
            t = 0
            ColPts = []
            for kk in range(0,ns[ii]):
                X = ColX(r,v,m)
                u = -v/norm(v)
                U = ColU(u,0.7)
                cp = ColPt(X,U)
                ColPts.append(cp)
                rp = r
                vp = v
                mp = m
                r = r + v*dt
                v = v + (-mu/norm(r)**3.0*r + tmax*0.7/m*u)*dt
                m = m - tmax*0.8/(9.81*isp)
                t = t + dt
            TrajLegs.append(TrajLeg(ColPts,t))
        OT = OptTraj(TrajLegs, nu0=nu0)
        X0 = ConstructDV(OT)
    elif opt_in['guess_from_file']:
        OTin = OptTrajFromJson(json.load(open(opt_in['in_file'])))
        X0 = ConstructDV(OTin)
        
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
        outdict = json.loads(str(ConstructOptTraj(pop.champion_x,ns)))
        if (not opt_out['only_write_feasible']) or (opt_out['only_write_feasible'] and is_feas):
            json.dump(outdict, open(opt_out['file_X_out'],'w+'), indent=4)
        

    


