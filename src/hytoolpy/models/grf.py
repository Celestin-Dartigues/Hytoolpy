"""
Barker (1988) general radial flow model This solution is
a generalisation of flow equation in 1D, 2D, 3D and non integer flow dimension.

This module provides:
    - Global parameter definition (pre)
    - Laplace domain solution (lap)
    - Dimensionless time solution (dls)
    - Dimensional solution (dim)
    - Initial parameter guess (gss)
    - Reporting and plotting (rpt)

References:
Barker, J.A. 1988. A Generalized radial flow model for hydraulic tests
in fractured rock. Water Resources Research 24, no. 10: 1796-1804
"""

import numpy as np
from scipy.special import kv, gamma
import matplotlib.pyplot as plt

from hytoolpy.tools.laplace import stefhest
from hytoolpy.tools.derivative import ldiffs, ldiff
from hytoolpy.tools.hyclean import hyclean
from hytoolpy.models import ths

# ============================================================
# Pre-processing function to set global well/test parameters
# ============================================================
def pre(r, rw):
    """
    Define global parameters used in dimensional scaling.

    Parameters
    ----------
    rw : float
        Well radius (m)
    r : float
        Observation well distance (m)
    """
    global GRFRADIUS, GRFWELLRADIUS, GRFWELLDIMENSIONLESSRADIUS
    GRFRADIUS = r
    GRFWELLRADIUS = rw
    GRFWELLDIMENSIONLESSRADIUS = r / rw


# =========================
# 1. Laplace domain solution
# =========================
def lap(x, p):
    """
    Laplace domain solution of the General Radial flow model.

    Parameters
    ----------
    x : list [n, rd]
        n : flow dimension
        rd : dimensionless radius
    p : float
        Laplace variable

    Returns
    -------
    val : float or ndarray
        Laplace-space solution value
    """
    n, rd = x
    sp = np.sqrt(p)
    term1 = rd**(2 - n)
    term2 = (rd**2 * p / 4)**(n/4 - 0.5)
    term3 = kv(n/2 - 1, rd * sp)
    term4 = p * gamma(n/2)
    sd = term1 * term2 * term3 / term4
    return sd


# ============================================================
# 2. Numerical Laplace inversion
# ============================================================
def dls(x, t):
    """
    Perform Laplace inversion using the Stefhest algorithm.

    Parameters
    ----------
    x : list [n, rd]
        Model parameters
    t : array-like
        Time values (must be > 0)

    Returns
    -------
    array
        Time-domain solution values
    """
    return np.array([stefhest(lap, x, ti) for ti in t])

# ============================================================
# 3. Dimensional scaling (real-world units)
# ============================================================
def dim(p, t):
    """
    Convert dimensionless General radiam flow solution to dimensional drawdown.

    Parameters
    ----------
    p : list [a, t0, n]
        a : slope of late-time derivative (1/m)
        t0 : intercept time
        n : flow dimension
    t : array
        Time values (s)

    Returns
    -------
    array
        Drawdown values (m)
    """
    a, t0, n = p
    td = t /2.2458/ t0
    sd = dls([n, GRFWELLDIMENSIONLESSRADIUS], td)
    return 0.868589 * a * sd

# ============================================================
# 4. Type-curve generation
# ============================================================
def drw():
    """
    Plot general radial flow dimensionless type curves.

    Parameters
    ----------
    """
    t = np.logspace(-2, 6)
    for n in [0.5, 1, 1.5, 2]:
        s = dls([n], t)
        plt.loglog(t, s, label=f'n={n}')
    plt.xlabel('t_D')
    plt.ylabel('s_D')
    plt.legend()
    plt.grid(True)
    plt.title('General Radial Flow type curves')
    plt.show()

# ============================================================
# 5. Initial parameter estimation
# ============================================================
def gss(t, s):
    """
    Estimate initial parameters [a, t0, n] for nonlinear fitting.

    Parameters
    ----------
    t : array
        Time values
    s : array
        Drawdown values

    Returns
    -------
    list
        Initial guess [a, t0, n]
    """
    
    # If necessary
    """n = int(len(t)/3)
    dt,ds = ldiffs(t[n:-1],s[n:-1])
    # Linear fit of the derivative
    a,t0 = np.polyfit(dt,ds,1)
    n = 2-2*a
    if n>0 and n<2 :
        te = dt[-1]
        se = np.interp(te,t,s)
        B = np.log10(se/ds[-1])
        alpha = 1
        t0 = te*(alpha*2**(1-n)/ds[-1]/gamma(n/2))**(-2/(n-2))
        a = alpha"""
    n = 2
    a,t0 = ths.gss(t,s)
    t0 = t0*(GRFWELLRADIUS/GRFRADIUS)**2        
    return [a, t0, n]

# ============================================================
# 6. Report generation (plots + statistics)
# ============================================================
def rpt(p,stats, t, s, d, npoint,
        name='General Radial Flow',
        title='General Radial Flow model',
        author='Author', report='Report',
        filetype='pdf',
        ax=None,
        color=None):
    """
    Generate final report with fitted model results.

    Parameters
    ----------
    p : list [a, t0, n]
        Model parameters
    stats : dict
        Fitting statistics {r2, rmse}
    t, s : arrays
        Observed time and drawdown data
    d : list [q, r]
        Test configuration
    name : str
        Model name
    title : str
        Plot title
    author, report : str
        Metadata
    filetype : str
        'pdf' or 'png'
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on. If None, creates a new figure.
    color : str, optional
        Line color for this model 
    """
    
    # --- Data cleanup ---
    t, s = hyclean(t, s)
    a, t0, n = p
    q, r = d
    
    # --- Hydraulic parameters ---
    T = 0.1832339 * q / a
    S = 2.2458394 * T * t0 / r**2
    Ri = 2 * np.sqrt(T * t[-1] / S)


    # --- Model curves ---
    tplot = np.logspace(np.log10(t[0]), np.log10(t[-1]), base=10.0, num=100)
    sc = dim(p, tplot)
    tc, sc = hyclean(tplot, sc)
    
    td, sd = ldiffs(t, s, npoint)
    td, sd = hyclean(td, sd)
    
    tdc, sdc = ldiffs(tc, sc, npoint)
    tdc, sdc = hyclean(tdc, sdc)

    # --- Stats ---
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
    ax.loglog(tc, sc, 'r-', label='General Radial flow model')
    ax.loglog(td, sd, 'gx', label='Derivative')
    ax.loglog(tdc, sdc, 'm--', label='GRF model derivative')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Drawdown (m)')
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()

    # --- Text report ---
    text_report = (
        f" Test parameters :\n"
        f"   q = {q:.2f} m³/s, r = {r:.2f} m\n\n"
        f" Hydraulic parameters :\n"
        f"   T = {T:.2e} m²/s\n"
        f"   S = {S:.2e} \n"
        f"   n (flow dimension) = {n:.2f} \n\n"
        f" Fit quality :\n"
        f"   R² = {r2:.3f}, RMSE = {rmse:.3f}"
    )

    fig.text(0.1, -0.2, text_report, ha="left", fontsize=10, family="arial",
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

    fig.tight_layout()

    # Sauvegarde
    if filetype == 'pdf':
        fig.savefig('GRF_report.pdf', bbox_inches='tight')
    else:
        fig.savefig('GRF_report.png', bbox_inches='tight')

    return fig,ax
