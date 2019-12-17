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
        return int(npts_tot + P + 2*npts_tot + P + P + 1 + 1)
    
    # TODO: Set these bounds in a smarter fashion.
    def get_bounds(self):
        
        sf_r = 20.0
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
        m_lb = [0.0]*sum(self.npts)
        m_ub = [1.1*self.mass0]*sum(self.npts)
        Eta_lb = [-0.1]*sum(self.npts)
        Eta_ub = [1.1]*sum(self.npts)
        T_lb = [0]*self.nphases
        T_ub = [1000]*self.nphases
        nu_lb = [-20.0]
        nu_ub = [10.0]
        LB = r_lb + v_lb + u_lb + m_lb + Eta_lb + nu_lb + T_lb
        UB = r_ub + v_ub + u_ub + m_ub + Eta_ub + nu_ub + T_ub
        return (LB, UB)
    
    def fitness(self, x):
        return self.run_traj(x, plot_traj=0)
    
    def summary(self, x):
        return self.run_traj(x, plot_traj=1)
    
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x, 1e-7)
    
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
        elif self.objecttype == 'fuel':
            nidx0 = 0
            for ii in range(0,P):
                dt = tofs[ii]/ns[ii]
                J = J + sum( [ 0.5*dt*( Eta[k+1] + Eta[k] ) for k in range(nidx0,nidx0 + ns[ii]-1) ] )
                nidx0 = nidx0 + ns[ii]
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
        Rmag = [ -(Rx[k]**2.0 + Ry[k]**2.0 + Rz[k]**2.0 - self.R_eq**2.0)/self.R_eq**2.0 for k in range(0,npt) ]
        CONSTR_INEQ = CONSTR_INEQ + Rmag
        
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
        CONSTR_INEQ = CONSTR_INEQ + eta_lb_con + eta_ub_con + T_lb_con + T_ub_con
        
        # Nu constraints
        CONSTR_INEQ = CONSTR_INEQ + [nu_lb - nu] + [nu - nu_ub]
        
        # Summarize output
        if write_sum == True:
            print("Total TOF:       %10.3f sec"%(sum(tofs)))
            print("    TOF 1:       %10.3f sec"%(tofs[0]))
            print("    TOF 2:       %10.3f sec"%(tofs[1]))
            print("    TOF 3:       %10.3f sec"%(tofs[2]))
            print("Final Mass:      %10.3f kg"%(m[-1]))
            print("  M0 Final:      %10.3f kg"%(m[ns[0]-1] - 8000 - 1000 - 1000))
        
        alt = array([norm([Rx[k], Ry[k], Rz[k]]) - self.R_eq for k in range(0,npt) ])
        if write_csv == True:
            fout = open('OUT_mp.csv', 'w+')
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
        
        Xdata_in = json.load(open(opt_in['in_file']))
        X0 = Xdata_in['X0']

        file_npts = Xdata_in['npts']
        npts = opt_config['npts']
        idx0 = 0
        for ii in range(0,len(file_npts)):
            if not (file_npts[ii] == npts[ii]):
                print("MISMATCH ", npts[ii], "  -  ",file_npts[ii])
            idx0 = idx0 + npts[ii]

        # if not (file_npts == npts):
        #     X0new = []
        #     for ii in range(0,11):
        #         X0new = X0new + recast_pts(X0[ii*file_npts:(ii+1)*file_npts],npts) 
        #     X0 = X0new + [X0[-2]] + [X0[-1]]
    
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
        

    


