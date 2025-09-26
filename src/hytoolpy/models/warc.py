# -*- coding: utf-8 -*-
"""
Warren & Root (1963) dual-porosity model with wellbore storage.

This module provides:
    - Laplace domain solution (lap)
    - Dimensionless time solution (dls)
    - Dimensional solution (dim)
    - Type-curve plotting (drw)
    - Initial parameter guess (gss)
    - Reporting and plotting (rpt)

Reference:
    Warren, J.E. & Root, P.J. (1963).
    The behavior of naturally fractured reservoirs.
    Society of Petroleum Engineers Journal, 3(3), 245–255.
"""

import numpy as np
from scipy.special import kv
import matplotlib.pyplot as plt

from hytoolpy.tools.laplace import stefhest
from hytoolpy.tools.derivative import ldiffs
from hytoolpy.tools.hyclean import hyclean


# =========================
# 1. Laplace domain solution
# =========================

def lap(x, p):
    """Laplace-domain solution for Warren & Root with wellbore storage.

    Parameters
    ----------
    x : list
        [sigma, lambda_, cd] where
        sigma   = storativity ratio
        lambda_ = interporosity flow parameter
        cd      = wellbore storage coefficient
    p : float
        Laplace parameter

    Returns
    -------
    float
        Dimensionless drawdown in Laplace domain
    """
    sigma, lambda_, cd = x
    alpha = np.sqrt(p + (lambda_ * sigma * p) / (sigma * p + lambda_))
    k0 = kv(0, alpha)
    return k0 / (p * (p * cd * k0 + alpha * kv(1, alpha)))


# =========================
# 2. Dimensionless solution
# =========================

def dls(x, t):
    """Numerical Laplace inversion for Warren & Root.

    Parameters
    ----------
    x : list
        [sigma, lambda_, cd]
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
    """Dimensional solution for Warren & Root with wellbore storage.

    Parameters
    ----------
    p : list
        Model parameters:
        p[0] = a    (slope of late-time Jacob line, m)
        p[1] = t0   (early intercept time, s)
        p[2] = t1   (late intercept time, s)
        p[3] = tm   (time at minimum derivative, s)
        p[4] = cd   (wellbore storage coefficient)
    t : array_like
        Time (s)

    Returns
    -------
    np.ndarray
        Drawdown values (m)
    """
    a, t0, t1, tm, cd = p

    td = 0.445268 * t / t0
    sigma = (t1 - t0) / t0
    lambda_ = 2.2458394 * t0 * np.log(t1 / t0) / tm

    sd = dls([sigma, lambda_, cd], td)
    return 0.868589 * a * np.array(sd)


# =========================
# 4. Type curves
# =========================

def drw():
    """Plot Warren & Root type curves with wellbore storage."""
    t = np.logspace(-4, 7, 200)
    lambda_ = 0.1
    cd = 0.1
    sigmas = [10, 100, 1000]

    plt.figure(figsize=(8, 6))
    for i, sigma in enumerate(sigmas):
        s = dls([sigma, lambda_, cd], t)
        dt, ds = ldiffs(t, s)
        style = ['-', '-.', '--'][i]
        plt.loglog(t, s, f'k{style}', label=f'σ={sigma}', linewidth=2)
        plt.loglog(dt, ds, 'k:', linewidth=1.5)

    plt.xlabel('t_D/r_D²')
    plt.ylabel('s_Df')
    plt.title(f'λ = {lambda_}, C_D = {cd}')
    plt.axis([1e-4, 1e7, 1e-3, 1e1])
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(['σ=10', 'σ=10²', 'σ=10³', 'Derivatives'], loc='upper left')
    plt.tight_layout()
    plt.show()


# =========================
# 5. Initial guess
# =========================

def gss(t, s):
    """Estimate initial parameters for Warren & Root with wellbore storage.

    Parameters
    ----------
    t : array_like
        Time (s)
    s : array_like
        Drawdown (m)

    Returns
    -------
    list
        Initial parameter guess [a, t0, t1, tm, cd]
    """
    t = np.asarray(t)
    s = np.asarray(s)

    # Log-derivative
    td, d = ldiffs(t, s, npoints=40)

    # Late-time slope
    n = len(t)
    l = n // 4
    dd = np.mean(d[-l:])
    a = np.log(10) * dd

    # Intercepts
    t0 = t[0] / np.exp(s[0] / dd)
    t1 = t[-1] / np.exp(s[-1] / dd)

    # Time of minimum derivative
    i_min = np.argmin(d)
    tm = td[i_min]

    # Wellbore storage coefficient
    mask = (s > 0) & (t > 0)
    sp, tp = s[mask], t[mask]
    if len(sp) == 0:
        raise ValueError("No positive values in t and s to estimate C_D.")

    cd = 0.8905356 * d[-1] / sp[0] * tp[0] / t0

    return [a, t0, t1, tm, cd]


# =========================
# 6. Reporting
# =========================

def rpt(p, stats, t, s, d, npoint,
        name='WarrenRoot',
        title='Warren & Root model with wellbore storage',
        author='Author', report='Report', filetype='pdf'):
    """Generate report for Warren & Root with wellbore storage.

    Parameters
    ----------
    p : list
        Model parameters [a, t0, t1, tm, cd]
    stats : dict
        Fit statistics {"r2": value, "rmse": value}
    t : array_like
        Measured time (s)
    s : array_like
        Measured drawdown (m)
    d : tuple
        (q, r) where
        q = pumping rate (m³/s)
        r = distance to observation well (m)
    npoint : int
        Number of points for derivative smoothing
    filetype : str, optional
        "pdf" or "png" (default = "pdf")
    """
    t, s = hyclean(t, s)
    a, t0, t1, tm, cd = p
    q, r = d

    # Hydraulic parameters
    T = 0.1832339 * q / a
    sigma = (t1 - t0) / t0
    lambda_ = 2.2458394 * t0 * np.log(t1 / t0) / tm

    # Model curve
    tplot = np.logspace(np.log10(t[0]), np.log10(t[-1]), num=100)
    sc = dim(p, tplot)
    tc, sc = hyclean(tplot, sc)

    # Derivatives
    td, sd = ldiffs(t, s, npoint)
    td, sd = hyclean(td, sd)
    tdc, sdc = ldiffs(tc, sc, npoint)
    tdc, sdc = hyclean(tdc, sdc)

    # Statistics
    r2, rmse = stats["r2"], stats["rmse"]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.loglog(t, s, 'bo', label='Data')
    ax.loglog(tc, sc, 'r-', label='Warren & Root model')
    ax.loglog(td, sd, 'gx', label='Derivative (data)')
    ax.loglog(tdc, sdc, 'm--', label='Derivative (model)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Drawdown (m)')
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()

    # Report text
    text_report = (
        f" Test parameters:\n"
        f"   q = {q:.2f} m³/s, r = {r:.2f} m\n\n"
        f" Hydraulic parameters:\n"
        f"   T = {T:.2e} m²/s\n"
        f"   σ = {sigma:.2f}\n"
        f"   λ = {lambda_:.2e}\n\n"
        f" Fit quality:\n"
        f"   R² = {r2:.3f}, RMSE = {rmse:.3f}"
    )

    plt.figtext(0.1, -0.25, text_report, ha="left", fontsize=12, family="arial")
    plt.tight_layout()

    # Save
    if filetype == 'pdf':
        fig.savefig('warren_root_report.pdf', bbox_inches='tight')
    else:
        fig.savefig('warren_root_report.png', bbox_inches='tight')

    plt.show()
    
