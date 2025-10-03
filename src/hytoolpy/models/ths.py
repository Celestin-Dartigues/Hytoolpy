"""The Theis (1935) solution aquifer is confined,homogeneous,
isotropic, and infinite.

This module provides:
    - Dimensionless time solution (dls)
    - Dimensional solution (dim)
    - Type curves (drw)
    - Jacobian matrix resolution for Theis solution (jac)
    - Initial parameter guess (gss)
    - Reporting and plotting (rpt)

Reference
Theis, Charles V. (1935) The relation between the lowering of the Piezometric
surface and the rate and duration of discharge of a well
using ground-water storage. Transactions, American Geophysical Union.
    
"""

import scipy as sp
import math
import numpy as np
import matplotlib.pyplot as plt

from hytoolpy.tools.laplace import stefhest
from hytoolpy.tools.derivative import ldiffs, ldiff
from hytoolpy.tools.hyclean import hyclean
from hytoolpy.models import jcb



# =========================
# 1. Dimensional solution
# =========================



def dim(p,t):
    '''THS_DIM - Compute drawdown with the Theis (1935) solution

 Syntax: s = ths.dim( p, t) 

   p(1) = a  = slope of Jacob Straight Line in meters
   p(2) = t0 = intercept with the horizontal axis for s = 0
   t = measured time
   s = measured drawdown

 Description:
   The Theis (1935) solution assumes that the aquifer is confined,
   homogeneous, isotropic, and infinite. The well as radius that is
   negligible. It is pumped at a constant rate Q and is 100 percent
   efficient.   

   Under these assumptions, Theis solution can be expressed as: 

       s(r,t) = Q/(4 pi T) E1( r2S / 4Tt)

   where Q is the pumping rate, T the transmissivity, r the radial 
   distance between the pumping well and the observation well, 
   S the storativity coefficient and t the time.  

   To interpret field data with the Theis solution, it is expressed as a
   function of two fitting parameters a and to which are defined as:   

   a = 0.183 Q /T
   t0 =  r2 S / 2.25 T 

 Example:
   s = ths.dim( p,t )

 See also: ths_dmo, ths_gss, ths_rpt'''

    t = np.array(t, dtype=np.float64)
    t = np.where(t <= 0, 1e-10, t)  # protection contre log(0) ou division par 0
    u = 0.5628 * p[1] / t
    s = p[0] / np.log(10) * sp.special.expn(1, u)
    return s


# =========================
# 2. Dimensionless solution
# =========================
def dls(td):
    '''THS_DLS - Dimensionless drawdown of the Theis model

 Syntax: sd,dd = ths.dls(td)

 Description:
   Calculates the dimensionless drawdown sd and the dimensionless 
   derivative dd for a given dimensionless reduced time td/rd^2

 See also: ths_lap'''
    td2 = []
    dd = []
    for i in range(0, len(td)):
        td2.append((0.25/td[i]))
        dd.append(0.5*np.exp(-1.0*td2[i]))

    sd = 0.5*sp.special.expn(1,td2)
    
    return sd

# =========================
# 3. Type curves
# =========================
def drw():
    t = np.logspace(-2, 6, 500)  # équivalent à 10.^([-1:0.05:4])

    # --- FIGURE 1 : Theis + dérivée ---
    s = dls(t)
    td, ds = ldiffs(t, s)

    plt.figure(1)
    plt.clf()
    plt.loglog(t, s, label='Theis')
    plt.loglog(td, ds, '-.', label='Derivative')
    plt.xlabel(r'$t_D / r_D^2 = Tt / Sr^2$', fontsize=14)
    plt.ylabel(r'$s_D = 2\pi s / Q$', fontsize=14)
    plt.legend(loc='upper left')
    plt.axis([1e-1, 1e4, 1e-2, 10])
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.title('Theis type curve and derivative')

    # --- FIGURE 2 : Theis vs Jacob ---
    t = np.logspace(-2, 6, 500)
    sj = jcb.dls(t)

    plt.figure(2)
    plt.clf()
    plt.semilogx(t, s, '--', label='Theis')
    plt.semilogx(t, sj, label='Jacob')
    plt.xlabel(r'$t_D / r_D^2$', fontsize=14)
    plt.ylabel(r'$s_D$', fontsize=14)
    plt.legend(loc='upper left')
    plt.axis([1e-1, 1e2, 0, 3])
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.title('Theis vs Jacob')

    # --- FIGURE 3 : Sensibilité à T (Transmissivité) ---
    td_sens = np.logspace(-1, 1, 100)
    a, b = 2, 0.7
    yt1 = dim([a, b], td_sens)
    yt2 = dim([2 * a, 2 * b], td_sens)

    plt.figure(3)
    plt.clf()
    plt.semilogx(td_sens, yt1, '-', label='T')
    plt.semilogx(td_sens, yt2, '--', label='2T')
    plt.xlabel(r'$t / r^2$', fontsize=14)
    plt.ylabel(r'$s$', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.title('Sensibilité à T')

    # --- FIGURE 4 : Sensibilité à Q (débit) ---
    yt1 = dim([a, b], td_sens)
    yt2 = dim([2 * a, b], td_sens)

    plt.figure(4)
    plt.clf()
    plt.semilogx(td_sens, yt1, '-', label='Q')
    plt.semilogx(td_sens, yt2, '--', label='2Q')
    plt.xlabel(r'$t / r^2$', fontsize=14)
    plt.ylabel(r'$s$', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.title('Sensibilité à Q')

    # --- FIGURE 5 : Sensibilité à S (emmagasinement) ---
    yt1 = dim([a, b], td_sens)
    yt2 = dim([a, 2 * b], td_sens)

    plt.figure(5)
    plt.clf()
    plt.semilogx(td_sens, yt1, '-', label='S')
    plt.semilogx(td_sens, yt2, '--', label='2S')
    plt.xlabel(r'$t / r^2$', fontsize=14)
    plt.ylabel(r'$s$', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.title('Sensibilité à S')

    plt.show()


# =========================
# 4. Initial guess
# =========================
def gss(t,s):
    '''THS_GSS - First guess for the parameters of the Theis model.

 Syntax:  p = ths.gss(t,s)

   p(1) = a  = slope of Jacob straight line for late time
   p(2) = t0 = intercept with the horizontal axis for s = 0
   t    = time
   s    = drawdown

 Description:
   First guess for the parameters of theis solution
'''

    if np.shape(t) == 1:
        t = np.transpose(t)
        s = np.transpose(s) #contrôler si c'est ce qui est souhaité
    
    n = round(len(t)/3)
    t = t[n:len(t)]
    s = s[n:len(s)]
    p = jcb.gss(t,s)
    
    return p



# =========================
# 5. Solving the matrix
# =========================
def jac(p,t):
    '''THS_JAC - Jacobian matrix of the Theis function

 Syntax: j = ths.jac( p, t)

    j(1,:) = ds / dp(1) 
    j(2,:) = ds / dp(2)
'''
    td = []
    for i in range(0, len(t)):
        td.append(0.5625*p[1]/t[i])
    j1 = []
    for i in range(0, len(td)):
        j1.append(sp.special.expn(1,td[i])/math.log(10))
       
    tdn = [-x for x in td]     
    j2 = []
    for i in range(0, len(td)):
        j2.append(p[0]*np.exp(tdn[i])/math.log(10)/p[1])

    j = [j1,j2]   

    return j 

# =========================
# 6. Reporting
# =========================
def rpt(p, stats, t, s, d, npoint,
        name='Theis',
        title='Theis model',
        author='Author', report='Report',
        filetype='pdf',
        ax=None,
        color=None):
    """
    Generate report for Theis solution.
    Can be used standalone or combined with other models (multi-plot).

    Parameters
    ----------
    p : list
        Model parameters [a, t0]
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
    """
    # Clean data
    t, s = hyclean(t, s)
    a,t0 = p
    q,r = d
    
    # Hydraulic parameters
    T = 0.1832339 * q / a
    S = 2.458394 * T * t0 / r**2
    Ri = 2 * np.sqrt(T * t[-1] / S)

    # Model curve
    tplot = np.logspace(np.log10(t[0]), np.log10(t[-1]), base=10.0, num=100)
    sc = dim(p, tplot)
    tc, sc = hyclean(tplot, sc)

    # Derivatives
    td, sd = ldiffs(t, s, npoint)
    td, sd = hyclean(td, sd)
    tdc, sdc = ldiffs(tc, sc, npoint)
    tdc, sdc = hyclean(tdc, sdc)

    # Stats
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
    ax.loglog(tc, sc, 'r-', label='Theis model')
    ax.loglog(td, sd, 'gx', label='Derivative')
    ax.loglog(tdc, sdc, 'm--', label='Theis model derivative')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Drawdown (m)')
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()

    # Report text
    text_report = (
        f" Test parameters :\n"
        f"   q = {q:.2f} m³/s, r = {r:.2f} m\n\n"
        f" Hydraulic parameters :\n"
        f"   T = {T:.2e}  m²/s\n"
        f"   S = {S:.2e} \n"
        f"   Radius of investigation = {Ri:.2f} m\n\n"
        f" Fit quality :\n"
        f"   R² = {r2:.3f}, RMSE = {rmse:.3f}"
    )

    fig.text(0.1, -0.2, text_report, ha="left", fontsize=10, family="arial",
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

    fig.tight_layout()

    # Sauvegarde
    if filetype == 'pdf':
        fig.savefig('Theis_report.pdf', bbox_inches='tight')
    else:
        fig.savefig('Theis_report.png', bbox_inches='tight')

    return fig,ax
