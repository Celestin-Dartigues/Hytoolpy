"""
Jacob straight-line method (Cooper & Jacob, 1946).

This module provides:
    - Dimensionless solution (dls)
    - Dimensional solution (dim)
    - Initial parameter guess (gss)
    - Type-curve plotting (drw)
    - Reporting and plotting (rpt)

Reference:
    Cooper, H.H. & Jacob, C.E. (1946).
    A generalized graphical method for evaluating formation constants
    and summarizing well field history.
    Eos, Transactions American Geophysical Union, 27(4), 526–534.
"""

import numpy as np
import matplotlib.pyplot as plt
from hytoolpy.tools.derivative import ldiffs
from hytoolpy.tools.hyclean import hyclean


# =========================
# 1. Laplace domain solution
# =========================

def lap(p):
    """No Laplace domain solution exists for Jacob method."""
    return None


# =========================
# 2. Dimensionless solution
# =========================

def dls(t):
    """Dimensionless Jacob solution.

    Parameters
    ----------
    t : array_like
        Dimensionless time

    Returns
    -------
    list
        Dimensionless drawdown values
    """
    td = t
    return [0.5 * (np.log(4 * ti) - 0.5772) for ti in td]


# =========================
# 3. Dimensional solution
# =========================

def dim(p, t):
    """Dimensional Jacob solution.

    Parameters
    ----------
    p : list
        Model parameters:
        p[0] = a   (slope of Jacob straight line, m)
        p[1] = t0  (time intercept, s)
    t : array_like
        Time (s)

    Returns
    -------
    list
        Drawdown values (m)
    """
    a, t0 = p
    td = 0.445268 * t / t0
    s_d = dls(td)
    return [0.868589 * a * sd for sd in s_d]


# =========================
# 4. Initial guess
# =========================

def gss(t, s, tmin=None, tmax=None):
    """Estimate initial parameters for Jacob method.

    Parameters
    ----------
    t : array_like
        Time (s)
    s : array_like
        Drawdown (m)
    tmin : float, optional
        Start of fitting interval (default: t[0])
    tmax : float, optional
        End of fitting interval (default: t[-1])

    Returns
    -------
    list
        Initial parameter guess [a, t0]
    """
    t = np.asarray(t)
    s = np.asarray(s)

    # Remove NaNs
    mask = ~np.isnan(s)
    t, s = t[mask], s[mask]

    # Time filtering
    if tmin is None:
        tmin = t[0]
    if tmax is None:
        tmax = t[-1]

    mask_time = (t >= tmin) & (t <= tmax)
    t, s = t[mask_time], s[mask_time]

    # Linear regression on log-time
    logt = np.log10(t)
    A = np.vstack([logt, np.ones_like(logt)]).T
    a, c = np.linalg.lstsq(A, s, rcond=None)[0]

    t0 = 10 ** (-c / a)

    return [a, t0]


# =========================
# 5. Type curves
# =========================

def drw():
    """Draw Jacob type curve."""
    t = np.logspace(0, 6)
    s = dls(t)
    plt.loglog(t, s)
    plt.xlabel('t_D')
    plt.ylabel('s_D')
    plt.title('Jacob type curve')
    plt.grid(True)
    plt.show()


# =========================
# 6. Reporting
# =========================

def rpt(p, stats, t, s, d, npoint,
        name='Jacob',
        title='Jacob model',
        author='Author', report='Report',
        filetype='pdf',
        ax=None,
        color=None):
    """
    Generate report for Jacob solution.
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
    q, r = d
    a, t0 = p

    # Hydraulic parameters
    T = 0.1832339 * q / a
    S = 2.2458394 * T * t0 / r**2
    Ri = 2 * np.sqrt(T * t[-1] / S)

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
    ax.loglog(tc, sc, 'r-', label='Jacob model')
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
        f"   S = {S:.2e}\n"
        f"   Radius of investigation = {Ri:.2f} m\n\n"
        f" Fit quality:\n"
        f"   R² = {r2:.4f}, RMSE = {rmse:.4f}"
    )

    fig.text(0.1, -0.2, text_report, ha="left", fontsize=10, family="arial",
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
    fig.tight_layout()

    # Save
    if filetype == 'pdf':
        fig.savefig('jacob_report.pdf', bbox_inches='tight')
    else:
        fig.savefig('jacob_report.png', bbox_inches='tight')

    return fig,ax
