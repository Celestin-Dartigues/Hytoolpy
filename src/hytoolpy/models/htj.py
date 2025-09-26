# -*- coding: utf-8 -*-
"""
Hantush-Jacob model for pumping tests in leaky aquifers.

This module provides:
    - Laplace domain solution (lap)
    - Dimensionless time solution (dls)
    - Dimensional solution (dim)
    - Type-curve plotting (drw)
    - Initial parameter guess (gss)
    - Reporting and plotting (rpt)

Reference:
    Hantush, M.S. & Jacob, C.E. (1955).
    Non-steady radial flow in an infinite leaky aquifer.
    Transactions, American Geophysical Union, 36(1), 95–100.
"""

import numpy as np
from scipy.special import k0
import matplotlib.pyplot as plt

from hytoolpy.tools.laplace import stefhest
from hytoolpy.tools.derivative import ldiffs
from hytoolpy.tools.hyclean import hyclean
from hytoolpy.models import ths


# =========================
# 1. Laplace domain solution
# =========================

def lap(x, p):
    """Laplace-domain solution for Hantush-Jacob model.

    Parameters
    ----------
    x : list
        Model parameters: [B] where B = r / leakage factor
    p : float
        Laplace parameter

    Returns
    -------
    float
        Dimensionless drawdown in Laplace domain
    """
    B = x[0]
    return (1.0 / p) * k0(np.sqrt(B**2 + p))


# =========================
# 2. Dimensionless solution
# =========================

def dls(x, t):
    """Dimensionless drawdown in time domain.

    Parameters
    ----------
    x : list
        [B] with B = r / leakage factor
    t : array_like
        Dimensionless time

    Returns
    -------
    np.ndarray
        Dimensionless drawdown values
    """
    return np.array([stefhest(lap, x, ti) for ti in t])


# =========================
# 3. Dimensional solution
# =========================

def dim(p, t):
    """Dimensional solution for Hantush-Jacob model.

    Parameters
    ----------
    p : list
        Model parameters:
        p[0] = a   (slope of Jacob line at late time, m)
        p[1] = t0  (time intercept for s = 0, s)
        p[2] = B   (leakage factor, dimensionless)
    t : array_like
        Time (s)

    Returns
    -------
    list
        Drawdown values (m)
    """
    a, t0, B = p
    td = 0.445268 * t / t0
    sd = dls([B], td)
    return [0.868589 * a * si for si in sd]


# =========================
# 4. Type curves
# =========================

def drw():
    """Draw Hantush-Jacob type curves."""
    t = np.logspace(-2, 6)
    for B in [0.01, 0.1, 1]:
        s = dls([B], t)
        dt, ds = ldiffs(t, s)
        plt.loglog(t, s, label=f'B={B}')
        plt.loglog(dt, ds, '-.', label=f'Derivative B={B}')
    plt.xlabel('t_D')
    plt.ylabel('s_D')
    plt.legend()
    plt.grid(True)
    plt.title('Hantush-Jacob type curves')
    plt.show()


# =========================
# 5. Initial guess
# =========================

def gss(t, s):
    """Estimate initial parameters for Hantush-Jacob model.

    Parameters
    ----------
    t : array_like
        Time (s)
    s : array_like
        Drawdown (m)

    Returns
    -------
    list
        Initial parameter guess [a, t0, B]
    """
    from hytoolpy.tools.fitting import fit
    t = np.asarray(t)
    s = np.asarray(s)

    # Use first third of data
    n = int(np.floor(len(t) / 3))
    t_short = t[:n]
    s_short = s[:n]

    # Initial guess from Theis
    p = ths.gss(t_short, s_short)

    # Fit Theis model
    p, stats = fit(p, t_short, s_short, 'ths')

    # Estimate r/B
    sm = s[-1]
    x = np.exp(-sm / p[0] * 2.3 / 2 + 0.1)
    if x > 1:
        x = -np.log(sm / p[0] * 2.3 / 2)

    return np.append(p, x)


# =========================
# 6. Reporting
# =========================

def rpt(p, stats, t, s, d, npoint,
        name='Hantush', title='Hantush-Jacob model',
        author='Author', report='Report', filetype='pdf'):
    """Generate report for Hantush-Jacob model interpretation.

    Parameters
    ----------
    p : list
        Model parameters [a, t0, B]
    stats : dict
        Fit statistics {"r2": value, "rmse": value}
    t : array_like
        Measured time (s)
    s : array_like
        Measured drawdown (m)
    d : tuple
        (q, r, e) where:
        q = pumping rate (m³/s)
        r = radial distance (m)
        e = aquitard thickness (m)
    npoint : int
        Number of points for derivative smoothing
    filetype : str, optional
        "pdf" or "png" (default = "pdf")
    """
    # Clean data
    t, s = hyclean(t, s)
    a, t0, rB = p
    q, r, e = d

    # Hydraulic parameters
    T = 0.1832339 * q / a
    S = 2.2458394 * T * t0 / r**2
    B = r / rB
    Ka = T * e / B**2

    # Model curve on log-spaced grid
    tplot = np.logspace(np.log10(t[0]), np.log10(t[-1]), num=100)
    sc = dim(p, tplot)
    tc, sc = hyclean(tplot, sc)

    # Derivatives
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
    ax.loglog(tc, sc, 'r-', label='Hantush-Jacob model')
    ax.loglog(td, sd, 'gx', label='Derivative')
    ax.loglog(tdc, sdc, 'm--', label='Model derivative')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Drawdown (m)')
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()

    # Report text
    text_report = (
        f" Test parameters:\n"
        f"   Discharge rate = {q:.2f} m³/s, Radial distance = {r:.2f} m, "
        f"Aquitard thickness = {e:.2f} m\n\n"
        f" Hydraulic parameters:\n"
        f"   T = {T:.2e} m²/s\n"
        f"   S = {S:.2e}\n"
        f"   Aquitard permeability = {Ka:.2e} m/s\n\n"
        f" Fit quality:\n"
        f"   R² = {r2:.3f}, RMSE = {rmse:.3f}"
    )

    plt.figtext(0.1, -0.25, text_report, ha="left", fontsize=12, family="arial")
    plt.tight_layout()

    # Save
    if filetype == 'pdf':
        fig.savefig('htj_report.pdf', bbox_inches='tight')
    else:
        fig.savefig('htj_report.png', bbox_inches='tight')

    plt.show()

