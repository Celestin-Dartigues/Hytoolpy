"""
Papadopulos and Cooper (1967) solution for a pumping test in a large
diameter well.
The aquifer is confined and homogeneous.
The well is fully penetrating and the pumping rate constant.   


This module provides:
    - Laplace domain solution for the derivative (der)
    - Laplace domain solution (lap)
    - Dimensionless time solution (dls)
    - Dimensional solution (dim)
    - Initial parameter guess (gss)
    - Reporting and plotting (rpt)

References:
Papadopulos, I.S., and H.H.J. Cooper. 1967. Drawdown in a
well of large diameter. Water Resources Research 3, no. 1: 241-244.
"""

import math 
import numpy as np
from scipy.special import kv
import matplotlib.pyplot as plt
from hytoolpy.tools.laplace import stefhest
from hytoolpy.tools.derivative import ldiffs, ldiff
from hytoolpy.tools.hyclean import hyclean
from hytoolpy.models import ths

###function der ###

def der(x,p):
    '''PCW_DER - Papadopulos-Cooper solution the well : Log derivative of the solution in Laplace domain
                
 Syntax: s = pcw.der( x, p)
   x[1] = Cd
   p = Laplace parameter 

 See also: pcw_lap
'''

    Sp = np.sqrt(p)
    
    k0 = kv(0,Sp)
    
    k1 = kv(1,Sp)
    
    s = 0.5*((2*x[0]-1)*k0**2+k1**2)/((Sp*k1+x[0]*p*k0)**2)
    
    return s

###function lap ###

def lap(x,p):
    '''PCW_LAP - Papadopulos Cooper Laplace domain solution in the well

 Syntax: s = pcw.lap( x, p)

   x[1] = Cd
   p = Laplace parameter 

 Description:
   Solution at the pumping well: 

                           K0( sqrt(p) )
  s = --------------------------------------------------------
       p [ sqrt(p)*K1( sqrt(p) ) + Cd * p * K0( sqrt(p) ) ]

 See also: pcw_dls
'''

    Sp = np.sqrt(p)
    
    k0 = kv(0,Sp)
    
    s = k0/(p*(Sp*kv(1,Sp)+p*x[0]*k0))

    return s

###function dls ###

def dls(x,t):
    '''PCW_DLS - Papadopulos Cooper dimensionless solution in the well

 Syntax: s,ds = pcw.dls( x, t)

   x(1) = Cd
   t = time 

 Description:
   Calculates the Dimensionless solution at the pumping well

    Reference: Papadopulos, I.S., and H.H.J. Cooper. 1967. Drawdown in a
    well of large diameter. Water Resources Research 3, no. 1: 241-244. 

 See also: pcw_lap
'''
    s = []
    for i in range(0, len(t)):
        s.append(stefhest(lap,x,t[i]))
    
    ds = []
    for i in range(0, len(t)):
        ds.append(stefhest(der,x,t[i]))
    
    
    return s,ds

###function dim ###

def dim(p,t):
    '''PCW_DIM - Papadopulos Cooper (1967) solution 

 Syntax: s,d = pcw.dim( p, t)

   p(1) = a  = slope of late time straight line
   p(2) = t0 = intercept of late time straight line
   p(3) = Cd = dimensionless well-bore storage coefficient

   t = time
   s = drawdonw
   d = derivative
  
% Description:
    Conputes the drawdown as a function of time with the Papadopulos and 
    Cooper (1967) solution for a pumping test in a large diameter well.
    The aquifer is confined and homogeneous. The well is fully penetrating 
    and the pumping rate constant.   
 
 
    The solution is parametrized as a function of a, to and Cd.

    The dimensionless well bore storage coefficient is:

     Cd = rc^2/(2 rw^2 S)

    a and to are the slope and time intercept of the late time straight
    line asymptotes.
     a = 0.183 Q /T
     t0 = 2.25 T t / r2 S

    NB: Note that in the original publication of Cooper et al.
    The dimensionless parameter was alpha, it is related to 
   Cd by: alpha = 1 / (2 Cd)
 
    Reference: Papadopulos, I.S., and H.H.J. Cooper. 1967. Drawdown in a
   well of large diameter. Water Resources Research 3, no. 1: 241-244. 

 Example:
   s=pcw_dim(p,t)

 See also: pcw_dmo, pcw_rpt, pcw_gss'''

    a = p[0]
    t0 = p[1]
    cd = [p[2]]
    
    var = 0.445268*t/t0
    
    
    s,d = dls(cd, var)
    
    ss = []
    dd = []
    
    for i in range(0,len(s)):
        ss.append(0.868589*a*s[i])
    
    
    for i in range(0,len(d)):
        dd.append(0.868589*a*d[i])
    
    
    
    return ss
    


def drw():
    t = np.logspace(-2, 6)
    for C in [0.001,0.01,0.1,100,1000]:
        s,ds = dls([C], t)
        dt,ds1 = ldiffs(t,s)
        plt.loglog(t, s, label=f'C={C}')
        plt.loglog(dt, ds1,'-.', label= 'Derivative' +' '+ f'C={C}')
    plt.xlabel('t_D')
    plt.ylabel('s_D')
    plt.legend()
    plt.grid(True)
    plt.title('Papadopoulos type curves')
    plt.show()
    



###function gss ###

def gss(t,s):
    '''PCW_GSS - First guess for the parameters of the Papadopulos Cooper solution

 Syntax: p = pcw.gss(t,s)

   p(1) = a   = slope of Jacob straight line for late time
   p(2) = t0  = intercept of the Jacob straight line for late time
   p(3) = Cd  = Dimensionless coefficient (1/2alpha)

   t    = time
   s    = drawdown

 Description:
   First guess for the parameters of Papadopulos Cooper solution'''

    t = np.asarray(t)
    s = np.asarray(s)

    td, d = ldiffs(t, s)
    d = np.asarray(d)  # Assure que c’est bien un array

    if d[-1] > 0:
        a = np.log(10) * d[-1]
        t0 = t[-1] * np.exp(-s[-1] / d[-1])
    else:
        return ths.gss(t, s)

    if t0 <= 0:
        t0 = 1e-5

    condition = (np.greater(t, 0) & np.greater(s, 0))
    sp = np.extract(condition, s)
    tp = np.extract(condition, t)

    if len(tp) == 0:
        print('HYTOOL: Error in pcw_gss - the vector t and s do not contain positive data')
        return float('NaN')

    cd = 0.8905356 * d[-1] / sp[0] * tp[0] / t0

    return [a, t0, cd]

    
###function rpt ###

def rpt(p, stats, t, s, d, npoint,
        name='Papadopoulos-Cooper',
        title='Papadopoulos-Cooper with wellbore storage',
        author='Author', report='Report',
        filetype='pdf',
        ax=None,
        color=None):
    """
    Generate report for Papadopoulos-Cooper with wellbore storage.
    Can be used standalone or combined with other models (multi-plot).

    Parameters
    ----------
    p : list
        Model parameters [a, t0,cd]
    stats : dict
        Fit statistics {"r2": value, "rmse": value}
    t : array_like
        Measured time (s)
    s : array_like
        Measured drawdown (m)
    d : tuple
        (q, r) pumping rate (m³/s) and distance to observation well (m)
    npoint : int
        Number of points for derivative smoothing
    filetype : str
        "pdf" or "png"
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on. If None, creates a new figure.
    color : str, optional
        Line color for this model
    
    Reference: Papadopulos, I.S., and H.H.J. Cooper. 1967. Drawdown in a 
    well of large diameter. Water Resources Research 3, no. 1: 241-244. """
    
    # == Nettoyage ==
    t, s = hyclean(t, s)
    q,rw,rc = d
    a,t0,cd = p
    
    
    #Compute the transmissivity
    T = 0.1832339*q/a
    
   
    # Courbe du modèle sur grille régulière
    tplot = np.logspace(np.log10(t[0]), np.log10(t[-1]), base=10.0, num=100)
    sc = dim(p, tplot)
    tc, sc = hyclean(tplot, sc)

    
    #calculate the derivative of the data
    td, sd = ldiffs(t,s, npoint)
    #keep only positive derivatives
    td, sd = hyclean(td,sd)
    
    #compute the derivative of the model
    tdc,sdc = ldiffs(tc,sc,npoint)
    #keep only positive derivatives
    tdc,sdc = hyclean(tdc,sdc)
    
    # === Statistiques ===
    r2 = stats["r2"]
    rmse = stats["rmse"]

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # Color by default
    if color is None:
        color = next(iter(plt.rcParams['axes.prop_cycle']))['color']
    
    # Color model for each curve and data
    data_style = dict(color='blue', marker='o', linestyle='None')
    model_style = dict(color='green', linestyle='-')
    deriv_data_style = dict(color='red', marker='x', linestyle='None', alpha=0.6)
    deriv_model_style = dict(color='purple', linestyle='--', alpha=0.6)
    
    ax.loglog(t, s, 'bo', label='Data')
    ax.loglog(tc, sc, 'r-', label='Papadopoulos Cooper model')
    ax.loglog(td, sd, 'gx', label='Derivative')
    ax.loglog(tdc, sdc, 'm--', label='Model derivative')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Drawdown (m)')
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()

    # === Rapport texte en dessous ===
    text_report = (
        f" Paramètres de l'essai :\n"
        f"   q = {q:.2f} m³/s, rw = {rw:.3f} m, rc = {rc:.3f} m\n\n"
        f" Paramètres hydrauliques :\n"
        f"   T = {T:.2e} m²/s\n"
        f"   C_D = {cd:.2e} \n\n"
        f" Qualité de l’ajustement :\n"
        f"   R² = {r2:.3f}, RMSE = {rmse:.3f}"
    )

    fig.text(0.1, -0.2, text_report, ha="left", fontsize=10, family="arial",
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

    fig.tight_layout()

    # Sauvegarde
    if filetype == 'pdf':
        fig.savefig('Papadopoulos_Cooper_report.pdf', bbox_inches='tight')
    else:
        fig.savefig('Papadopoulos_Cooper_report.png', bbox_inches='tight')

    return fig,ax







