#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 09:48:16 2019

MULTI-PHASE 3-Dimensional full gravity optimization problem.
This should match optimality!!!!

@author: jasonmeverett
"""

import numpy as np
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
from enum import Enum

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

class RegionFlags(Enum):
    FRONT = 1
    BACK = 2
    PATH = 3

class ObjectiveFlags(Enum):
    LAGRANGE = 1
    MAYER = 2 # Not yet implemented
    BOLZA = 3 # Not yet implemented
    TIME = 4

class TrajLeg(object):
    def __init__(self, num_nodes=0, Tinit=0.0):
        self.num_nodes = num_nodes
        self.T = Tinit
        self.f = ()
        self.J = ()
        self.params = []
        self.dynamicsSet = False
        self.objs = []
        self.coneqs_f = []
        self.conins_f = []
        self.coneqs_b = []
        self.conins_b = []
        self.coneqs_p = []
        self.conins_p = []
        self.conpth = []
        self.lenX = 0
        self.lenU = 0
        self.lenN = 0 
        self.objSet = False
        self.objType = 0
        self.Tmin = -1.0
        self.Tmax = -1.0
        self.TOFset = False
        self.bnds_min = []
        self.bnds_max = []
    def set_dynamics(self, f, dfdX, dfdU, params=[]):
        self.f = (f, dfdX, dfdU, params)
        self.dynamicsSet = True
    def set_len_X_U(self, lenX, lenU):
        self.lenX = lenX
        self.lenU = lenU
        self.lenN = lenX + lenU
    def add_eq(self, con, dcon, lcon, reg, params=[], td=False):
        if reg == RegionFlags.FRONT:
            self.coneqs_f.append((con, dcon, lcon, params, td))
        elif reg == RegionFlags.BACK:
            self.coneqs_b.append((con, dcon, lcon, params, td))
        elif reg == RegionFlags.PATH:
            self.coneqs_p.append((con, dcon, lcon, params, td))
    def add_ineq(self, con, dcon, lcon, reg, params=[], td=False):
        if reg == RegionFlags.FRONT:
            self.conins_f.append((con, dcon, lcon, params, td))
        elif reg == RegionFlags.BACK:
            self.conins_b.append((con, dcon, lcon, params, td))
        elif reg == RegionFlags.PATH:
            self.conins_p.append((con, dcon, lcon, params, td))
    def getTotLength(self):
        return int(1 + self.num_nodes*(self.lenX + self.lenU))
    def set_TOF(self, Tmin, Tmax):
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.TOFset = True
    def set_bounds(self, bnds_min, bnds_max):
        self.bnds_min = bnds_min
        self.bnds_max = bnds_max
    def set_obj(self, fobj, dfobj, typ, params=[]):
        self.J = (fobj, dfobj, params)
        self.objType = typ
        self.objSet = True
        return
        
    
class Ananke_Config(object):
    
    # Function initialization
    def __init__(self):
        self.TrajLegs = []
        self.LegLinks = []
        self.maxTOF = -1.0
        self.minTOF = -1.0
        self.idxLegObj = 0
        return
    
    def add_leg_link(self, l1, l2, lfun, dlfun, length, params, td=False):
        self.LegLinks.append((l1, l2, lfun, dlfun, length, params, td))
        return
    
    def set_TOF(self, minTOF, maxTOF):
        self.maxTOF = maxTOF
        self.minTOF = minTOF
        return
    
    def get_array_data(self, x):
        out = []
        t0 = 0.0
        for ii,TL in enumerate(self.TrajLegs):
            t0 = sum([ x[self.get_dvi_T(jj)] for jj in range(0, ii) ])
            TTs = np.linspace(t0, t0 + x[self.get_dvi_T(ii)], TL.num_nodes)
            outLeg = []
            for jj in range(0, TL.num_nodes):
                id0 = self.get_dvi_N(ii, jj)
                X = x[id0:(id0 + TL.lenX)].tolist()
                U = x[(id0 + TL.lenX):(id0 + TL.lenX + TL.lenU)] .tolist() 
                outLeg.append([TTs[jj]] + X + U)
            out.append(np.array(outLeg))    
        return out
                
    def get_nec(self):
        num_defects = sum([ TL.lenX * (TL.num_nodes-1) for TL in self.TrajLegs])
        num_con_user = 0
        for TL in self.TrajLegs:
            for coneq in TL.coneqs_p:
                num_con_user += coneq[2] * TL.num_nodes
            for coneq in TL.coneqs_f:
                num_con_user += coneq[2]
            for coneq in TL.coneqs_b:
                num_con_user += coneq[2]
        for LL in self.LegLinks:
            num_con_user += LL[4]
        return int(num_defects + num_con_user)
    
    def get_nic(self):
        num_con_user = 0
        for TL in self.TrajLegs:
            for conin in TL.conins_p:
                num_con_user += conin[2] * TL.num_nodes
            for conin in TL.conins_f:
                num_con_user += conin[2]
            for conin in TL.conins_b:
                num_con_user += conin[2]
            if(TL.Tmin > 0.0):
                num_con_user += 2
        if (self.maxTOF > 0.0):
            num_con_user += 2
        return int(num_con_user)
    
    def get_bounds(self):
        LB = []
        UB = []
        for TL in self.TrajLegs:
            LB.append(-10000.0)
            LB = LB + TL.num_nodes * TL.bnds_min
        for TL in self.TrajLegs:
            UB.append( 10000.0)
            UB = UB + TL.num_nodes * TL.bnds_max
        return (LB, UB)
    
    # Objective value. Find leg that has obj enabled.
    def calc_J(self, x):
        TLobj = self.TrajLegs[self.idxLegObj]
        TL = TLobj
        T0 = sum([ x[self.get_dvi_T(ii)] for ii in range(0, self.idxLegObj) ])
        idT = self.get_dvi_T(self.idxLegObj)
        dt = x[idT] / float(TLobj.num_nodes)
        J = 0.0
        if TLobj.objType == ObjectiveFlags.LAGRANGE:
            J = 0.0
            for ii in range(0, TLobj.num_nodes-1):
                Tk = T0 + dt*ii
                Tkp1 = T0 + dt*(ii+1)
                id0_Xk = self.get_dvi_N(self.idxLegObj, ii)
                idf_Xk = id0_Xk + TL.lenX
                id0_Uk = idf_Xk
                idf_Uk = id0_Uk + TL.lenU
                id0_Xkp1 = idf_Uk
                idf_Xkp1 = id0_Xkp1 + TL.lenX
                id0_Ukp1 = idf_Xkp1
                idf_Ukp1 = id0_Ukp1 + TL.lenU
                Xk   = x[id0_Xk:idf_Xk]
                Uk   = x[id0_Uk:idf_Uk]
                Xkp1 = x[id0_Xkp1:idf_Xkp1]
                Ukp1 = x[id0_Ukp1:idf_Ukp1]   
                params = TLobj.J[2]
                J1 = TLobj.J[0](Xk, Uk, Tk, params)
                J2 = TLobj.J[0](Xkp1, Ukp1, Tkp1, params)
                J += 0.5 * (J1 + J2) * dt
        return J
    
    def fitness(self, x):
        OBJVAL = []
        CONEQ = []
        CONIN = []
        CONLK = []
        
        # Find all times
        Ts = np.take(x, [self.get_dvi_T(ii) for ii in range(0, len(self.TrajLegs))])
        
        # Calculate objective.
        J = self.calc_J(x)
        OBJVAL = [J]
        
        # Equality constraints
        constr_eqs = []
        constr_ins = []
        tofTOT = 0.0
        for ii, TL in enumerate(self.TrajLegs):
            
            # DT for this specific leg
            T0 = sum([ x[self.get_dvi_T(jj)] for jj in range(0, ii) ])
            idT = self.get_dvi_T(ii)
            dt = x[idT] / float(TL.num_nodes)
            tofTOT += x[idT]
            
            # Equality constraints - front, back, path
            for jj in range(0, len(TL.coneqs_f)):
                id0 = self.get_dvi_N(ii, 0)
                X = x[id0:(id0 + TL.lenX)]
                U = x[(id0 + TL.lenX):(id0 + TL.lenX + TL.lenU)]
                T = T0
                constr_eqs = constr_eqs + TL.coneqs_f[jj][0](X, U, T, TL.coneqs_f[jj][3]).tolist()
            for jj in range(0, len(TL.coneqs_b)):
                id0 = self.get_dvi_N(ii, -1)
                X = x[id0:(id0 + TL.lenX)]
                U = x[(id0 + TL.lenX):(id0 + TL.lenX + TL.lenU)]
                T = T0 + x[self.get_dvi_T(ii)]
                constr_eqs = constr_eqs + TL.coneqs_b[jj][0](X, U, T, TL.coneqs_b[jj][3]).tolist()
            for jj in range(0, len(TL.coneqs_p)):
                for kk in range(0, TL.num_nodes):
                    id0 = self.get_dvi_N(ii, kk)
                    X = x[id0:(id0 + TL.lenX)]
                    U = x[(id0 + TL.lenX):(id0 + TL.lenX + TL.lenU)]
                    T = T0 + dt*kk
                    constr_eqs = constr_eqs + TL.coneqs_p[jj][0](X, U, T, TL.coneqs_p[jj][3]).tolist()

            # Equality constraints - collocation
            for jj in range(0, TL.num_nodes-1):
                Tk = T0 + dt*jj
                Tkp1 = T0 + dt*(jj+1)
                id0_Xk = self.get_dvi_N(ii, jj)
                idf_Xk = id0_Xk + TL.lenX
                id0_Uk = idf_Xk
                idf_Uk = id0_Uk + TL.lenU
                id0_Xkp1 = idf_Uk
                idf_Xkp1 = id0_Xkp1 + TL.lenX
                id0_Ukp1 = idf_Xkp1
                idf_Ukp1 = id0_Ukp1 + TL.lenU
                Xk   = x[id0_Xk:idf_Xk]
                Uk   = x[id0_Uk:idf_Uk]
                Xkp1 = x[id0_Xkp1:idf_Xkp1]
                Ukp1 = x[id0_Ukp1:idf_Ukp1]  
                fk = TL.f[0](Xk, Uk, Tk, TL.f[3])
                fkp1 = TL.f[0](Xkp1, Ukp1, Tkp1, TL.f[3])
                Uc = 0.5*(Uk+Ukp1)
                Xc = 0.5*(Xk+Xkp1) + dt/8*(fk-fkp1)
                Tc = 0.5*(Tk+Tkp1)
                fc = TL.f[0](Xc, Uc, Tc, TL.f[3])
                constr_p = Xk-Xkp1 + dt/6*(fk + 4*fc + fkp1)
                constr_eqs = constr_eqs + constr_p.tolist()
            
            # Inequality constraints for Leg - front, back, path
            for jj in range(0, len(TL.conins_f)):
                id0 = self.get_dvi_N(ii, 0)
                X = x[id0:(id0 + TL.lenX)]
                U = x[(id0 + TL.lenX):(id0 + TL.lenX + TL.lenU)]
                T = T0
                constr_ins = constr_ins + TL.conins_f[jj][0](X, U, T, TL.conins_f[jj][3]).tolist()
            for jj in range(0, len(TL.conins_b)):
                id0 = self.get_dvi_N(ii, -1)
                X = x[id0:(id0 + TL.lenX)]
                U = x[(id0 + TL.lenX):(id0 + TL.lenX + TL.lenU)]
                T = T0 + x[self.get_dvi_T(ii)]
                constr_ins = constr_ins + TL.conins_b[jj][0](X, U, T, TL.conins_b[jj][3]).tolist()
            for jj in range(0, len(TL.conins_p)):
                for kk in range(0, TL.num_nodes):
                    id0 = self.get_dvi_N(ii, kk)
                    X = x[id0:(id0 + TL.lenX)]
                    U = x[(id0 + TL.lenX):(id0 + TL.lenX + TL.lenU)]
                    T = T0 + dt*kk
                    constr_ins = constr_ins + TL.conins_p[jj][0](X, U, T, TL.conins_p[jj][3]).tolist()
            
            # TOF Constraints for Leg
            constr_ins = constr_ins + [TL.Tmin - x[idT]]
            constr_ins = constr_ins + [x[idT] - TL.Tmax]
            
        # Equality constraints - linking
        for ii, LL in enumerate(self.LegLinks):
            li0 = LL[0]
            li1 = LL[1]
            params = LL[5]
            id0_Xk = self.get_dvi_N(li0, -1)
            idf_Xk = id0_Xk + self.TrajLegs[li0].lenX
            id0_Uk = idf_Xk
            idf_Uk = id0_Uk + self.TrajLegs[li0].lenU
            id0_Xkp1 = self.get_dvi_N(li1, 0)
            idf_Xkp1 = id0_Xkp1 + self.TrajLegs[li1].lenX
            id0_Ukp1 = idf_Xkp1
            idf_Ukp1 = id0_Ukp1 + self.TrajLegs[li1].lenU
            Xk   = x[id0_Xk:idf_Xk]
            Uk   = x[id0_Uk:idf_Uk]
            Xkp1 = x[id0_Xkp1:idf_Xkp1]
            Ukp1 = x[id0_Ukp1:idf_Ukp1]  
            T = self.get_dvi_T(li1)
            con = LL[2](Xk, Uk, Xkp1, Ukp1, T, params)
            CONLK = CONLK + con.tolist()
            
        # Set total equality constraints
        CONEQ = constr_eqs
        
        # Add total TOF constraints
        constr_ins = constr_ins + [self.minTOF - tofTOT]
        constr_ins = constr_ins + [tofTOT - self.maxTOF]
        CONIN = constr_ins

        return OBJVAL + CONEQ + CONLK + CONIN
    
    def estimate_grad(self, x, idxs, delta):
        fout = np.zeros((1 + self.get_nec() + self.get_nic(), len(idxs)))
        for iiout,ii in enumerate(idxs):
            x0 = x.copy()
            x1 = x.copy()
            x0[ii] = x0[ii] - delta
            x1[ii] = x1[ii] + delta
            f0 = np.array(self.fitness(x0))
            f1 = np.array(self.fitness(x1))
            fav = (f1 - f0) / (2.0 * delta)
            fout[:,iiout] = fav
        return fout    
           
    def gradient(self, x):
        
        # Set up gradient size.
        grad_clc = np.zeros((1+self.get_nec() + self.get_nic(), len(x)))
        
        # Calculate partial for cost runction.
        T0 = sum([ x[self.get_dvi_T(ii)] for ii in range(0, self.idxLegObj) ])
        TLobj = self.TrajLegs[self.idxLegObj]
        idT = self.get_dvi_T(self.idxLegObj)
        dt = x[idT] / float(TLobj.num_nodes)
        J = self.calc_J(x)
        dJ = np.zeros((1, len(x)))
        if TLobj.objType == ObjectiveFlags.LAGRANGE:
            dJ[0,idT] = J / x[idT]
            for jj in range(0, TLobj.num_nodes):
                Tk = T0 + dt*jj
                id0X = self.get_dvi_N(self.idxLegObj, jj)
                idfX = id0X + TLobj.lenX
                Xk = x[id0X:idfX]
                id0U = idfX
                idfU = id0U + TLobj.lenU
                Uk = x[id0U:idfU]
                mlt = 1.0 + float(not(jj == 0) and not (jj == TLobj.num_nodes-1))
                dJ[0,id0X:idfU] = (mlt) * 0.5 * dt * TLobj.J[1](Xk, Uk, Tk, TLobj.J[2])
        grad_clc[0,:] = dJ
        
        # Equality constraints
        idrow = 1
        for ii, TL in enumerate(self.TrajLegs):
            
            T0 = sum([ x[self.get_dvi_T(jj)] for jj in range(0, ii) ])
            idT = self.get_dvi_T(ii)
            dt = x[idT] / float(TL.num_nodes)
            
            for jj, coneq in enumerate(TL.coneqs_f):
                id0 = self.get_dvi_N(ii, 0)
                idf = id0 + TL.lenX + TL.lenU
                X = x[id0:(id0 + TL.lenX)]
                U = x[(id0 + TL.lenX):idf]
                T = T0
                td = coneq[4]
                if td:
                    dcon,delt = coneq[1](X, U, T, coneq[3])
                    grad_clc[idrow:(idrow+coneq[2]),id0:idf] = dcon
                    grad_clc[idrow:(idrow+coneq[2]),idT] = delt
                else:
                    dcon = coneq[1](X, U, T, coneq[3])
                    grad_clc[idrow:(idrow+coneq[2]),id0:idf] = dcon
                idrow += coneq[2]
                
            for jj, coneq in enumerate(TL.coneqs_b):
                id0 = self.get_dvi_N(ii, -1)
                idf = id0 + TL.lenX + TL.lenU
                X = x[id0:(id0 + TL.lenX)]
                U = x[(id0 + TL.lenX):idf]
                T = T0 + x[self.get_dvi_T(ii)]
                td = coneq[4]
                if td:
                    dcon,delt = coneq[1](X, U, T, coneq[3])
                    grad_clc[idrow:(idrow+coneq[2]),id0:idf] = dcon
                    grad_clc[idrow:(idrow+coneq[2]),idT] = delt
                else:
                    dcon = coneq[1](X, U, T, coneq[3])
                    grad_clc[idrow:(idrow+coneq[2]),id0:idf] = dcon
                idrow += coneq[2]
                
            for jj, coneq in enumerate(TL.coneqs_p):
                for kk in range(0, TL.num_nodes):
                    id0 = self.get_dvi_N(ii, kk)
                    idf = id0 + TL.lenX + TL.lenU
                    X = x[id0:(id0 + TL.lenX)]
                    U = x[(id0 + TL.lenX):idf]
                    T = T0 + dt*kk
                    dcon = coneq[1](X, U, T, coneq[3])
                    grad_clc[idrow:(idrow+coneq[2]),id0:idf] = dcon
                    idrow += coneq[2]
        
            # Equality constraints - collocation
            idT = self.get_dvi_T(ii)
            for jj in range(0, TL.num_nodes-1):
                Tk = T0 + dt*jj
                Tkp1 = T0 + dt*(jj+1)
                id0_Xk = self.get_dvi_N(ii, jj)
                idf_Xk = id0_Xk + TL.lenX
                id0_Uk = idf_Xk
                idf_Uk = id0_Uk + TL.lenU
                id0_Xkp1 = idf_Uk
                idf_Xkp1 = id0_Xkp1 + TL.lenX
                id0_Ukp1 = idf_Xkp1
                idf_Ukp1 = id0_Ukp1 + TL.lenU
                Xk   = x[id0_Xk:idf_Xk]
                Uk   = x[id0_Uk:idf_Uk]
                Xkp1 = x[id0_Xkp1:idf_Xkp1]
                Ukp1 = x[id0_Ukp1:idf_Ukp1]  
                fk = TL.f[0](Xk, Uk, Tk, TL.f[3])
                fkp1 = TL.f[0](Xkp1, Ukp1, Tkp1, TL.f[3])
                Uc = 0.5*(Uk+Ukp1)
                Xc = 0.5*(Xk+Xkp1) + dt/8*(fk-fkp1)
                Tc = 0.5*(Tk+Tkp1)
                fc = TL.f[0](Xc, Uc, Tc, TL.f[3])
                dfdX = TL.f[1]
                dfdU = TL.f[2]
                params = TL.f[3]
                Ak = dfdX(Xk,Uk, Tk,params)
                Akp1 = dfdX(Xkp1,Ukp1, Tkp1,params)
                Ac = dfdX(Xc,Uc, Tc,params)
                Bk = dfdU(Xk,Uk, Tk, params)
                Bkp1 = dfdU(Xkp1,Ukp1, Tkp1, params)
                Bc = dfdU(Xc,Uc, Tc, params)
                I = np.eye(TL.lenX)
                dDel_dXk = I + dt/6*(Ak + 4*matmul(Ac, (0.5*I + dt/8*Ak) ))
                dDel_dXkp1 = -I + dt/6*(Akp1 + 4*matmul( Ac, (0.5*I - dt/8*Akp1) ) )
                dDel_dUk = dt/6*(Bk + 4*(matmul(Ac, dt/8*Bk) + 1/2*Bc) )
                dDel_dUkp1 = dt/6*(Bkp1 + 4*(matmul(Ac, -dt/8*Bkp1) + 1/2*Bc) )
                dDel_dT = 1/(6*TL.num_nodes)*(fk + 4*fc + fkp1 + dt/2*matmul(Ac, fk-fkp1))
                grad_clc[idrow:(idrow+TL.lenX),id0_Xk:idf_Xk] = dDel_dXk
                grad_clc[idrow:(idrow+TL.lenX),id0_Uk:idf_Uk] = dDel_dUk
                grad_clc[idrow:(idrow+TL.lenX),id0_Xkp1:idf_Xkp1] = dDel_dXkp1
                grad_clc[idrow:(idrow+TL.lenX),id0_Ukp1:idf_Ukp1] = dDel_dUkp1
                grad_clc[idrow:(idrow+TL.lenX),idT] = dDel_dT
                idrow += TL.lenX
        
        for ii, LL in enumerate(self.LegLinks):
            li0 = LL[0]
            li1 = LL[1]
            params = LL[5]
            id0_Xk = self.get_dvi_N(li0, -1)
            idf_Xk = id0_Xk + self.TrajLegs[li0].lenX
            id0_Uk = idf_Xk
            idf_Uk = id0_Uk + self.TrajLegs[li0].lenU
            id0_Xkp1 = self.get_dvi_N(li1, 0)
            idf_Xkp1 = id0_Xkp1 + self.TrajLegs[li1].lenX
            id0_Ukp1 = idf_Xkp1
            idf_Ukp1 = id0_Ukp1 + self.TrajLegs[li1].lenU
            Xk   = x[id0_Xk:idf_Xk]
            Uk   = x[id0_Uk:idf_Uk]
            Xkp1 = x[id0_Xkp1:idf_Xkp1]
            Ukp1 = x[id0_Ukp1:idf_Ukp1]  
            T = self.get_dvi_T(li1)
            dl1, dl2 = LL[3](Xk, Uk, Xkp1, Ukp1, T, params)
            grad_clc[idrow:(idrow+LL[4]),id0_Xk:idf_Uk] = dl1
            grad_clc[idrow:(idrow+LL[4]),id0_Xkp1:idf_Ukp1] = dl2
            idrow += LL[4]
        
        # Inequality constraints
        for ii, TL in enumerate(self.TrajLegs):
            
            T0 = sum([ x[self.get_dvi_T(jj)] for jj in range(0, ii) ])
            idT = self.get_dvi_T(ii)
            dt = x[idT] / float(TL.num_nodes)
            
            for jj, conin in enumerate(TL.conins_f):
                id0 = self.get_dvi_N(ii, 0)
                idf = id0 + TL.lenX + TL.lenU
                X = x[id0:(id0 + TL.lenX)]
                U = x[(id0 + TL.lenX):idf]
                T = T0
                dcon = TL.conins_f[jj][1](X, U, T, TL.conins_f[jj][3])
                grad_clc[idrow:(idrow+TL.conins_f[jj][2]),id0:idf] = dcon
                idrow += TL.conins_f[jj][2]
            for jj, conin in enumerate(TL.conins_b):
                id0 = self.get_dvi_N(ii, -1)
                idf = id0 + TL.lenX + TL.lenU
                X = x[id0:(id0 + TL.lenX)]
                U = x[(id0 + TL.lenX):idf]
                T = T0 + x[self.get_dvi_T(ii)]
                dcon = TL.conins_b[jj][1](X, U, T, TL.conins_b[jj][3])
                grad_clc[idrow:(idrow+TL.conins_b[jj][2]),id0:idf] = dcon
                idrow += TL.conins_b[jj][2]
            for jj in range(0, len(TL.conins_p)):
                for kk in range(0, TL.num_nodes):
                    id0 = self.get_dvi_N(ii, kk)
                    idf = id0 + TL.lenX + TL.lenU
                    X = x[id0:(id0 + TL.lenX)]
                    U = x[(id0 + TL.lenX):idf]
                    T = T0 + dt*kk
                    dcon = TL.conins_p[jj][1](X, U, T, TL.conins_p[jj][3])
                    grad_clc[idrow:(idrow+TL.conins_p[jj][2]),id0:idf] = dcon
                    idrow += TL.conins_p[jj][2]
        
            # Leg TOF Constraints
            grad_clc[idrow,idT] = -1.0
            idrow += 1
            grad_clc[idrow,idT] = 1.0
            idrow += 1
        
        # TOF Total Constraints
        for ii, TL in enumerate(self.TrajLegs):
            idT = self.get_dvi_T(ii)
            grad_clc[idrow,idT] = -1.0
            grad_clc[idrow+1,idT] = 1.0
        idrow += 2
                    
        # Gradient estimation and comparison
        # grad_est = pg.estimate_gradient_h(lambda x: self.fitness(x), x, 1e-6)
        # grad_est = grad_est.reshape((grad_clc.shape[0],grad_clc.shape[1]))
        # grad_rtn_est = grad_est.reshape((grad_clc.shape[0]*grad_clc.shape[1],))
        # for ii in range(0, grad_est.shape[0]):
        #     print(ii, " , ", norm(grad_est[ii,:] - grad_clc[ii,:]))
        # print('----------')
        # print(norm(grad_est - grad_clc))
        # print('----------')
        
        grad_rtn_clc = grad_clc.reshape((grad_clc.shape[0]*grad_clc.shape[1],))
        return grad_rtn_clc

    def get_num_legs(self):
        return len(self.Legs)

    def add_leg(self, TL):
        self.TrajLegs.append(TL)
        if TL.objSet == True:
            self.idxLegObj = len(self.TrajLegs)-1
        return
        
    def getTotLength(self):
        return int(sum([TL.getTotLength() for TL in self.TrajLegs]))

    def get_dvi_L(self, idx_leg):
        i0 = 0
        if idx_leg == -1:
            idx_leg = len(self.TrajLegs)-1
        for ii in range(0, idx_leg):
            i0 = i0 + self.TrajLegs[ii].getTotLength()
        return i0
    
    def get_dvi_N(self, idx_leg, idx_node):
        i0 = self.get_dvi_L(idx_leg)
        i0 = i0 + 1 # Compensate for time
        if idx_node == -1:
            idx_node = self.TrajLegs[idx_leg].num_nodes - 1
        for ii in range(0, idx_node):
            i0 = i0 + self.TrajLegs[idx_leg].lenN
        return i0
    
    def get_dvi_U(self, idx_leg, idx_node):
        i0 = self.get_dvi_N(idx_leg, idx_node)
        i0 = i0  + self.TrajLegs[idx_leg].lenX
        return i0
        
    def get_dvi_X(self, idx_leg, idx_node):
        i0 = self.get_dvi_N(idx_leg, idx_node)
        return i0
    
    def get_dvi_T(self, idx_leg):
        i0 = self.get_dvi_L(idx_leg)
        return i0
