#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 17:53:21 2017

@author: Dartigues
"""

from hytoolpy.tools.laplace import stefhest
from hytoolpy.tools.derivative import ldiffs, ldiff
from hytoolpy.tools.hyclean import hyclean
from scipy.special import kv
import numpy as np
import matplotlib.pyplot as plt
from hytoolpy.models import ths

# ============================================================
# 1. Analytical solution in Laplace domain (Warren-Root model)
# ============================================================

def lap(x,p):
    '''WAR_LAP - Warren and Root (1963) solution in  Laplace domain

 Syntax:
   war.lap(x,p) provides the dimensionless drawdown at the well

   x(1) = sigma
   x(2) = lamda
   p    = Laplace parameter
'''
 
    s = x[0]
    lam = x[1]
    
    s = 1/p*kv(0,np.sqrt(p+(lam*s*p)/(s*p+lam)))
    
    return s


# ============================================================
# 2. Numerical Laplace inversion
# ============================================================
    
def dls(x,td):
    '''WAR_DLS - Dimensionless drawdown of Warren and Root (1963) solution

 Syntax: war.dls(x,t)
   x(1) = sigma 
   x(2) = lamda
   t = time

 Description:
   Calculates the dimensionless drawdown

 References:
   Warren, J. E., and P. J. Root (1963), The behaviour of naturally 
   fractured reservoirs, Society of Petroleum Engineers Journal, 3, 
   245-255.
''' 
        
    return np.array([stefhest(lap, x, ti) for ti in td])



# ============================================================
# 3. Dimensional scaling (real-world units)
# ============================================================
    
def dim(p,t):
    '''WAR_DIM - Warren and Root (1965) solution

 Syntax: s = war.dim( p, t)

   p(1) = a  = slope of Jacob Straight Line
   p(2) = t0 = intercept with the horizontal axis for 
               the early time asymptote
   p(3) = t1 = intercept with the horizontal axis for 
               the late time asymptote
   p(4) = tm = time of the minimum of the derivative

 Description:
   Calculate the drawdown at time t for confined aquifer with double 
   porosity 

 Example:
   s=war_dim(p,t)

 See also: war_dmo, war_rpt, war_gss'''
 
    a = p[0]
    t0 = p[1]
    t1 = p[2]
    tm = p[3]
    
    td = 0.445268*t/t0
     
    sigma = (t1-t0)/t0
    
    lam = 2.2458394*t0*np.log(t1/t0)/tm #changed the name, otherwise gives an error
    
    sd = dls([sigma,lam],td)
    
    
    s = []
    
    for i in range(0,len(sd)):
        s.append(0.868589*a*sd[i])

    return s

# ============================================================
# 4. Initial parameter estimation
# ============================================================
    
def gss(t,s):
    '''WAR_GSS - First guess for the parameters of the Warren and Root solution

 Syntax: p = war.gss(t,s)

   p(1) = a  = slope of Jacob Straight Line
   p(2) = t0 = intercept with the horizontal axis for 
               the early time asymptote
   p(3) = t1 = intercept with the horizontal axis for 
               the late time asymptote
   p(4) = tm = time of the minimum of the derivative

   t    = time
   s    = drawdown

 Description: 
   First guess for the parameters of the Warren and Root solution

   See also: war_dmo, war_dim, war_rpt'''
    t = np.asarray(t)
    s = np.asarray(s)
    
    # 1. Dérivée log-log
    td,ds = ldiffs(t,s,npoints=40)

    # 2.Moyenne de la dérivée à la fin
    dd = np.mean(ds[-4:-1])
    a = np.log(10) * dd

    #3.Estimation de t0 et t1
    t0 = t[0]*np.exp(-s[0]/dd)
    t1 = t[-1]*np.exp(-s[-1]/dd)

    # 4. Temps du minimum de la dérivée
    i = np.argmin(ds)
    tm = td[i]
    
    
    return [a, t0, t1, tm]


# ============================================================
# 5. Type-curve generation
# ============================================================
def drw():
    t = np.logspace(-2, 7, 500)

    # === FIGURE 1 : Courbes types log-log pour différentes lambda ===
    sigma = 100
    lambdas = [1, 0.1, 0.01]
    fig1, ax1 = plt.subplots()
    sl_list = []

    for lam in lambdas:
        s = dls([sigma, lam], t)
        sl_list.append(s)
        ax1.loglog(t, s, linewidth=2, label=f'λ={lam}')

    for s in sl_list:
        dt, ds = ldiffs(t, s)
        ax1.loglog(dt, ds, ':k', linewidth=2)

    ax1.set_xlim(1e-1, 1e6)
    ax1.set_ylim(1e-2, 1e1)
    ax1.set_xlabel('tD/rD²')
    ax1.set_ylabel('sD')
    ax1.set_title(f'σ = {sigma}')
    ax1.grid(True)
    ax1.legend(fontsize=12, loc='lower right')

    # === FIGURE 2 : Courbes semi-log pour différentes lambda ===
    fig2, ax2 = plt.subplots()
    for s, lam in zip(sl_list, lambdas):
        ax2.semilogx(t, s, linewidth=2, label=f'λ={lam}')

    ax2.set_xlim(1e-1, 1e5)
    ax2.set_ylim(0, 4)
    ax2.set_xlabel('tD/rD²')
    ax2.set_ylabel('sD')
    ax2.set_title(f'σ = {sigma}')
    ax2.grid(True)
    ax2.legend(fontsize=12, loc='lower right')

    # === FIGURE 3 : Courbes types log-log pour différentes sigma ===
    lambda_fixed = 0.1
    sigmas = [10, 100, 1000]
    fig3, ax3 = plt.subplots()

    for sig in sigmas:
        s = dls([sig, lambda_fixed], t)
        ax3.loglog(t, s, linewidth=2, label=f'σ={sig}')
        dt, ds = ldiffs(t, s)
        ax3.loglog(dt, ds, ':k', linewidth=2)

    ax3.set_xlim(1e-1, 1e6)
    ax3.set_ylim(1e-3, 1e1)
    ax3.set_xlabel('tD/rD²')
    ax3.set_ylabel('sD')
    ax3.set_title(f'λ = {lambda_fixed}')
    ax3.grid(True)
    ax3.legend(fontsize=12, loc='lower right')

    plt.show()


# ============================================================
# 6. Report generation (plots + statistics)
# ============================================================
def rpt(p,stats,t,s,d, npoint, name, title = 'Interference test', Author = 'My name',  Rapport = 'My Rapport', filetype = 'img'):
    '''WAR_RPT - Produces the final figure and results for the Warren and Root model
    
    Syntax: war.rpt( p, t, s, d, ttle ) 
    p(1) = a = slope of Jacob Straight Line 
    p(2) = t0 = intercept with the horizontal axis for the early time asymptote 
    p(3) = t1 = intercept with the horizontal axis for the late time asymptote 
    p(4) = tm = time of the minimum of the derivative 
    
    t = measured time 
     s = measured drawdown 
     d(1) = Q = Pumping rate 
     d(2) = r = Distance to the pumping well 
     ttle = Title of the figure 
     
     Description: 
       Produces the final figure and results for the Warren and Root model
         
       See also: war_dmo, war_dim, war_gss'''
    t,s = hyclean(t,s)
    #Rename the parameters for a more intuitive check of the formulas
    Q,r = d
    a,t0,t1,tm = p
    
    #Compute the transmissivity, storativity and radius of influence
    Tf=0.1832339*Q/a
    Sf=2.245839*Tf*t0/r**2
    Sm=2.245839*Tf*t1/r**2-Sf;
    
    sigma = (t1-t0)/t0
    lambada = 2.2458394*t0*np.log(t1/t0)/tm
    

    # Model curves
    tplot = np.logspace(np.log10(t[0]), np.log10(t[-1]), base=10.0, num=100)
    sc = dim(p, tplot)
    tc, sc = hyclean(tplot, sc)

    td, sd = ldiffs(t, s, npoint)
    td, sd = hyclean(td, sd)

    tdc, sdc = ldiffs(tc, sc,npoint)
    tdc, sdc = hyclean(tdc, sdc)

    # Stats
    r2 = stats["r2"]
    rmse = stats["rmse"]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.loglog(t, s, 'bo', label='Data')
    ax.loglog(tc, sc, 'r-', label='Warren Root model')
    ax.loglog(td, sd, 'gx', label='Derivative')
    ax.loglog(tdc, sdc, 'm--', label='Model derivative')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Drawdown (m)')
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()

    # Text report
    text_report = (
        f" Test parameters :\n"
        f"   q = {q:.2f} m³/s, r = {r:.2f} m\n\n"
        f" Hydraulic parameters :\n"
        f"   T = {T:.2e}  m²/s\n"
        f"   S_frac = {Sf:.2e} \n"
        f"   S_matrix = {Sm:.2e} \n"
        f"   Interporosity flow lambda = {lambada:.2f} \n\n"
        f" Fit quality :\n"
        f"   R² = {r2:.3f}, RMSE = {rmse:.3f}"
    )

    plt.figtext(0.1, -0.25, text_report, ha="left", fontsize=12, family="arial")

    plt.tight_layout()

    # Sauvegarde
    if filetype == 'pdf':
        fig.savefig('Warren_Root_report.pdf', bbox_inches='tight')
    else:
        fig.savefig('Warren_Root_report.png', bbox_inches='tight')

    plt.show()
         
