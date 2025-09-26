#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:07:07 2017

@author: Dartigues
"""


import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from hytoolpy.tools.derivative import ldiffs, ldiff
from hytoolpy.tools.hyclean import hyclean
from hytoolpy.models import ths

# =========================
# 1. Dimensional solution
# =========================

def dim(p,t):
    '''THN_DIM - Theis model with a no-flow boundary

 Syntax: s = thn.dim(p,t)
   p(1) = a  = slope of Jacob Straight Line
   p(2) = t0 = intercept of the first segment of straight line
   p(3) = ti = time of intersection between the 2 straight lines

 Description:
   Calculate the drawdown at time t for confined aquifer with a no-flow
   boundary

 Example:
   s = thn.dim( p,t )
'''

    a = p[0]
    
    s = ths.dim([a,p[1]],t) + ths.dim([a,p[2]],t)
    
    return s


# =========================
# 2. Dimensionless solution
# =========================
    
def dls(p,t):
    '''THN_DLS - Dimensionless drawdown of the Theis model with a no-flow boundary

 Syntax: s = thn.dls(p,t)

   p(1) = r1/r2  
   with r1 = radius to the pumping well 
   and  r2 = radius to the image well 

   provides the dimensionless drawdown at reduced time t'''

    t = 0.25/t
    
    s = 0.5*sp.special.expn(1,t)+0.5*sp.special.expn(1,t*p[0]**2)
    
    return s    


# =========================
# 4. Initial guess
# =========================

def gss(t,s):
    '''THN_GSS - First guess for the parameters of the Theis model with a no-flow boundary

 Syntax: p = thn.gss(t,s)

   p(1) = a  = slope of Jacob straight line 
   p(2) = t0 = intercept of the Jacob straight line 
   p(3) = ti = time of intersection between the 2 straight lines
   t    = time
   s    = drawdown

 Description:
   First guess for the parameters of theis solution with a no-flow
   boundary

 See also: thn_dmo, thn_rpt, thn_dim
'''
    t = np.array(t)
    s = np.array(s)
    
    #Automatic identification of the "control" points
    td,d = ldiffs(t,s, npoints=10) #First log derivative
    tdd,dd = ldiffs(td,d, npoints=10) #Second log derivative

    
    # Vérification que dd est non vide
    if len(dd) == 0:
        raise ValueError("Dérivée seconde vide : vérifie tes données.")

    # Valeur maximale de la dérivée seconde
    i = np.argmax(dd)
    if i == 0:
        raise ValueError("Maximum de dérivée seconde au bord gauche, estimation incertaine.")
    
    ti = tdd[i - 1]

    # Pente de la droite de Jacob (hypothèse pour thn)
    a = d[-1] * np.log(10) / 2  # 2.30 = ln(10)

    # Origine t0 de la droite de Jacob
    if a == 0:
        raise ZeroDivisionError("La pente a est nulle, t0 ne peut pas être calculé.")
    
    t_end = t[-1]
    s_end = s[-1]

    argument = (a * np.log10(t_end**2 / ti) - s_end) / a
    t0 = 10 ** argument

    return [a, t0, ti]    


# =========================
# 6. Reporting
# =========================
    
def rpt(p,stats,t,s,d, npoint, name, title = 'Theis no flow boundary test', Author = 'My name',  Rapport = 'My Rapport', filetype = 'img'):
    '''THN_RPT - Produces the final figure and results for the Theis model with a no flow boundary
    
    Syntax: thn.rpt( p, t, s, d, ttle )
    
    p = parameters of the model 
    t = measured time 
    s = measured drawdown 
    d(1) = Q = Pumping rate 
    d(2) = r = Distance between the observation and the pumping well 
    title = Title of the figure (Optional) 
    
    Description: 
         Produces the final figure and results for Theis model (1935) with 
          a no flow boundary.'''
    # Clean data
    t,s = hyclean(t,s)
    a,t0,ti = p
    q,r = d
    
    # Hydraulic parameters
    T = 0.1832339*q/a
    S = 2.458394*T*t0/r**2
    Ri = np.sqrt(2.2458394*T*ti/S)
    
    
    # Model curve
    tplot = np.logspace(np.log10(t[0]), np.log10(t[-1]), base=10.0, num=100)
    sc = dim(p, tplot)
    tc, sc = hyclean(tplot, sc)

    td, sd = ldiffs(t, s, npoint)
    td, sd = hyclean(td, sd)

    tdc, sdc = ldiffs(tc, sc, npoint)
    tdc, sdc = hyclean(tdc, sdc)

    # Statistics
    r2 = stats["r2"]
    rmse = stats["rmse"]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.loglog(t, s, 'bo', label='Data')
    ax.loglog(tc, sc, 'r-', label='Theis model with no flow boundary')
    ax.loglog(td, sd, 'gx', label='Derivative')
    ax.loglog(tdc, sdc, 'm--', label='Model derivative')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Drawdown (m)')
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()

    # === Rapport texte en dessous ===
    text_report = (
        f" Test parameters :\n"
        f"   q = {q:.2f} m³/s, r = {r:.2f} m\n\n"
        f" Hydraulic parameters :\n"
        f"   T = {T:.2e}  m²/s\n"
        f"   S = {S:.2e} \n"
        f"   Distance to boundary = {Ri:.2f} m\n\n"
        f" Fit quality :\n"
        f"   R² = {r2:.3f}, RMSE = {rmse:.3f}"
    )

    plt.figtext(0.1, -0.25, text_report, ha="left", fontsize=12, family="arial")

    plt.tight_layout()

    # Save
    if filetype == 'pdf':
        fig.savefig('Theis_no_flow_boundary_report.pdf', bbox_inches='tight')
    else:
        fig.savefig('Theis_no_flow_boundary_report.png', bbox_inches='tight')

    plt.show()   
    
    
