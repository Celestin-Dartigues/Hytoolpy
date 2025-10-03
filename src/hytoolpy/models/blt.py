"""
Boulton (1963) model for unconfined aquifer pumping tests.

This module provides:
    - Laplace domain solution (lap)
    - Dimensionless time solution (dls)
    - Dimensional solution (dim)
    - Initial parameter guess (gss)
    - Reporting and plotting (rpt)

References:
    Boulton, N. S. (1963). Analysis of data from non-equilibrium pumping tests
    allowing for delayed yield from storage. Proc. Inst. Civil Eng., 26(4): 469–482.
"""

import numpy as np
from scipy.special import kv
import matplotlib.pyplot as plt
from hytoolpy.tools.laplace import stefhest
from hytoolpy.tools.derivative import ldiffs
from hytoolpy.tools.hyclean import hyclean
from hytoolpy.models import jcb


# =========================
# 1. Laplace domain solution
# =========================

def lap(x, p):
    """Boulton (1963) Laplace-domain dimensionless solution.

    Parameters
    ----------
    x : list
        Model parameters in Laplace domain
        x[0] = sigma = S / Sy
        x[1] = phi   = (alpha * r^2 * S) / T
    p : float
        Laplace parameter

    Returns
    -------
    float
        Dimensionless drawdown in Laplace domain.
    """
    return kv(0, np.sqrt(p + x[1]*p/(x[0]*(p + x[1])))) / p


# =========================
# 2. Dimensionless solution
# =========================

def dls(x, t):
    """Dimensionless drawdown in time domain (Boulton 1963).

    Parameters
    ----------
    x : list
        [sigma, phi]
    t : array_like
        Dimensionless time

    Returns
    -------
    list
        Dimensionless drawdown values.
    """
    return [stefhest(lap, x, ti) for ti in t]


# =========================
# 3. Dimensional solution
# =========================

def dim(p, t):
    """Boulton (1963) dimensional solution.

    Parameters
    ----------
    p : list
        Model parameters:
        p[0] = a   (slope of Jacob straight line at late time, m)
        p[1] = t0  (intercept time for early asymptote, s)
        p[2] = t1  (intercept time for late asymptote, s)
        p[3] = phi (empirical parameter triggering the delay)
    t : array_like
        Time (s)

    Returns
    -------
    list
        Drawdown values (m).
    """
    td = 0.445268 * t / p[1]
    sds = dls([p[1] / (p[1] + p[2]), 2 * p[3] * p[1]], td)
    return [0.868589 * p[0] * sdi for sdi in sds]


# =========================
# 4. Initial guess
# =========================

def gss(t, s):
    """Initial guess for Boulton (1963) model parameters.

    Parameters
    ----------
    t : array_like
        Time (s)
    s : array_like
        Drawdown (m)

    Returns
    -------
    list
        Initial parameter guess:
        [a, t0, t1, phi]
    """
    t = np.asarray(t)
    s = np.asarray(s)

    p = [1, 2, 3, 4]
    p[1] = t[0]

    # Use last 2/3 of data for Jacob straight-line guess
    n = round(len(t) / 3)
    t1, s1 = t[n-1:], s[n-1:]

    pj = jcb.gss(t1, s1)

    p[0] = pj[0]
    p[2] = pj[1]
    p[3] = 1e-4

    return p


# =========================
# 5. Reporting and plotting
# =========================

def rpt(p, stats, t, s, d, npoint,
        name="Boulton", title="Interference test",
        author="Author",
        report="Report",
        filetype="pdf",
        ax=None,
        color=None):
    """Generate a report for Boulton (1963) model interpretation.

    Parameters
    ----------
    p : list
        Model parameters [a, t0, t1, phi]
    stats : dict
        Fit statistics {"r2": value, "rmse": value}
    t : array_like
        Measured time (s)
    s : array_like
        Measured drawdown (m)
    d : tuple
        (q, r) where
        q = pumping rate (m³/s)
        r = distance between pumping well and piezometer (m)
    npoint : int
        Smoothing parameter for derivative
    name : str, optional
        Model name (default "Boulton")
    title : str, optional
        Plot title
    author : str, optional
        Report author
    report : str, optional
        Report name
    filetype : str, optional
        Output file type: "png" or "pdf"
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on. If None, creates a new figure.
    color : str, optional
        Line color for this model
    Returns
    -------
    None
    """
    # Clean input data
    t, s = hyclean(t, s)
    a, t0, t1, phi = p
    q, r = d

    # Hydraulic parameters
    T = 0.1832339 * q / a
    S = 2.2458394 * T * t0 / r**2
    omegad = 2.2458394 * T * t1 / r**2 - S
    Ri = 2 * np.sqrt(T * t[-1] / omegad)

    # Model curve on log-spaced grid
    tplot = np.logspace(np.log10(t[0]), np.log10(t[-1]), num=100)
    sc = dim(p, tplot)
    tc, sc = hyclean(tplot, sc)

    # Log-derivative of data
    td, sd = ldiffs(t, s, npoint)
    td, sd = hyclean(td, sd)

    # Log-derivative of model
    tdc, sdc = ldiffs(tc, sc, npoint)
    tdc, sdc = hyclean(tdc, sdc)

    # Statistics
    r2, rmse = stats["r2"], stats["rmse"]

    #  Plot
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
    

    ax.loglog(t, s, "bo", label="Data")
    ax.loglog(tc, sc, "r-", label="Boulton model")
    ax.loglog(td, sd, "gx", label="Derivative")
    ax.loglog(tdc, sdc, "m--", label="Boulton model derivative")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Drawdown (m)")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()

    # Text report
    text_report = (
        f" Test parameters:\n"
        f"   q = {q:.2f} m³/s, r = {r:.2f} m \n\n"
        f" Hydraulic parameters:\n"
        f"   T = {T:.2e} m²/s\n"
        f"   S = {S:.2e} \n"
        f"   Omega = {omegad:.2f} \n"
        f"   Radius of investigation = {Ri:.2e} m \n\n"
        f" Fit quality:\n"
        f"   R² = {r2:.3f}, RMSE = {rmse:.3f}"
    )

    fig.text(0.1, -0.2, text_report, ha="left", fontsize=10, family="arial",
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
    fig.tight_layout()

    #  Sauvegarde
    if filetype == 'pdf':
        fig.savefig('Boulton_report.pdf', bbox_inches='tight')
    else:
        fig.savefig('Boulton_report.png', bbox_inches='tight')

    return fig,ax
