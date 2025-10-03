# -*- coding: utf-8 -*-
"""
Logarithmic derivative utilities for pumping test analysis.

This module provides several methods to compute log-log derivatives:
    - ldiff   : centered differences
    - ldiffs  : spline-based
    - ldiffb  : Bourdet's method
    - ldiffh  : Horne's method

References:
    Bourdet, D., Whittle, T.M., Douglas, A.A., & Pirard, Y.M. (1983).
    A new set of type curves simplifies well test analysis.
    World Oil, 196(6), 95–106.

    Horne, R.N. (1995).
    Modern Well Test Analysis: A Computer-Aided Approach.
    Petroway, Inc.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as spi
from hytoolpy.tools.hyclean import hyclean


# =========================
# 1. Centered differences
# =========================

def ldiff(t, s):
    """Approximate logarithmic derivative with centered differences.

    Parameters
    ----------
    t : array_like
        Time (must be > 0, increasing)
    s : array_like
        Drawdown values

    Returns
    -------
    xd, yd : arrays
        Time points and derivative values
    """
    t, s = hyclean(t, s)
    dx = np.diff(t)
    dy = np.diff(s)

    # Midpoints in logarithmic scale
    xd = [math.sqrt(t[i] * t[i+1]) for i in range(len(t)-1)]

    # Derivative estimation
    yd = np.array(xd) * (dy / dx)

    return xd, yd


# =========================
# 2. Spline-based method
# =========================

def ldiffs(t, s, npoints=20):
    """Approximate logarithmic derivative using cubic spline.

    Parameters
    ----------
    t : array_like
        Time
    s : array_like
        Drawdown
    npoints : int, optional
        Number of log-spaced interpolation points (default = 20)

    Returns
    -------
    xd, yd : arrays
        Time points and spline-based derivative values
    """
    t, s = hyclean(t, s)
    f = len(t)

    # Log-spaced interpolation points
    xi = np.logspace(np.log10(t[0]), np.log10(t[f-1]), num=npoints)

    # Cubic spline interpolation
    spl = spi.UnivariateSpline(t, s, k=5, s=0.0099)
    yi = spl(xi)

    # Derivative estimation
    xd = xi[1:-1]
    yd = xd * (yi[2:] - yi[:-2]) / (xi[2:] - xi[:-2])

    return xd, yd


# =========================
# 3. Bourdet's method
# =========================

def ldiffb(t, s, d=2):
    """Approximate logarithmic derivative using Bourdet's method.

    Parameters
    ----------
    t : array_like
        Time (must be > 0 and strictly increasing)
    s : array_like
        Drawdown
    d : int, optional
        Lag distance (default = 2)

    Returns
    -------
    xd, yd : arrays
        Time points and Bourdet derivative values
    """
    t = np.asarray(t, dtype=float).flatten()
    s = np.asarray(s, dtype=float).flatten()

    # Keep only valid finite values
    mask = (t > 0) & np.isfinite(t) & np.isfinite(s)
    t = t[mask]
    s = s[mask]

    if len(t) < 2*d + 2:
        raise ValueError(f"Not enough points ({len(t)}) for Bourdet derivative with d={d}.")

    logx = np.log(t)
    dx = np.diff(logx)
    dy = np.diff(s)

    dx1 = dx[0:len(dx)-2*d+1]
    dx2 = dx[2*d-1:len(dx)]
    dy1 = dy[0:len(dy)-2*d+1]
    dy2 = dy[2*d-1:len(dy)]

    n = min(len(dx1), len(dx2), len(dy1), len(dy2))
    if n <= 0:
        raise ValueError("Not enough valid points after index alignment.")

    dx1, dx2, dy1, dy2 = dx1[:n], dx2[:n], dy1[:n], dy2[:n]

    yd = (dx2 * dy1 / dx1 + dx1 * dy2 / dx2) / (dx1 + dx2)
    xd = t[d+1:d+1+n]  # center points

    return xd, yd


# =========================
# 4. Horne's method
# =========================

def ldiffh(t, s):
    """Approximate logarithmic derivative using Horne's method.

    Parameters
    ----------
    t : array_like
        Time
    s : array_like
        Drawdown

    Returns
    -------
    xd, yd : arrays
        Time points and Horne derivative values
    """
    t, s = hyclean(t, s)
    endt = len(t)
    ends = len(s)

    t1, t2, t3 = t[0:endt-2], t[1:endt-1], t[2:endt]
    s1, s2, s3 = s[0:ends-2], s[1:ends-1], s[2:ends]

    # d1 term
    logt2t1 = [math.log(t2[i]/t1[i]) for i in range(len(t1))]
    logt3t2 = [math.log(t3[i]/t2[i]) for i in range(len(t1))]
    logt3t1 = [math.log(t3[i]/t1[i]) for i in range(len(t1))]
    d1 = (np.array(logt2t1) * np.array(s3)) / (np.array(logt3t2) * np.array(logt3t1))

    # d2 term
    logt3t1t2 = [math.log(t3[i]*t1[i]/t2[i]**2) for i in range(len(t1))]
    d2 = (np.array(logt3t1t2) * np.array(s2)) / (np.array(logt3t2) * np.array(logt2t1))

    # d3 term
    d3 = (np.array(logt3t2) * np.array(s1)) / (np.array(logt2t1) * np.array(logt3t2))

    # Final derivative
    xd = t2
    yd = d1 + d2 - d3

    return xd, yd
