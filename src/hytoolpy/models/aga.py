import numpy as np
from scipy.special import kv
import matplotlib.pyplot as plt
import mpmath as mp

from hytoolpy.tools.laplace import stefhest, dehoog
from hytoolpy.tools.derivative import ldiffs, ldiff
from hytoolpy.tools.hyclean import hyclean
from hytoolpy.models import ths


# ============================================================
# Pre-processing function to set global well/test parameters
# ============================================================
def pre(rw, rc, r, q):
    """
    Define global parameters used in dimensional scaling.

    Parameters
    ----------
    rw : float
        Well radius (m)
    rc : float
        Casing radius (m)
    r : float
        Observation well distance (m)
    q : float
        Pumping/discharge rate (m³/s)
    """
    global AGA_RW, AGA_RC, AGA_R, AGA_Q
    AGA_RW = rw
    AGA_RC = rc
    AGA_R = r
    AGA_Q = q


# ============================================================
# 1. Analytical solution in Laplace domain (Agarwal model)
# ============================================================
def lap(x, p):
    """
    Laplace domain solution of the Agarwal model.

    Parameters
    ----------
    x : list [cd, rd, sg]
        cd : dimensionless wellbore storage coefficient
        rd : dimensionless radius
        sg : skin factor
    p : float
        Laplace variable

    Returns
    -------
    val : float or ndarray
        Laplace-space solution value
    """
    cd, rd, sg = x
    s = np.sqrt(p)
    k0 = kv(0, s)
    k1 = kv(1, s)
    numerator = kv(0, rd * s)
    denominator = p * (((1 + p * cd * sg) * s * k1) + (cd * p * k0))

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.where(np.abs(denominator) > 1e-15, numerator / denominator, 0.0)

    return val


# ============================================================
# 2. Numerical Laplace inversion
# ============================================================
def dls(x, t):
    """
    Perform Laplace inversion using the de Hoog algorithm.

    Parameters
    ----------
    x : list [cd, rd, sg]
        Model parameters
    t : array-like
        Time values (must be > 0)

    Returns
    -------
    array
        Time-domain solution values
    """
    t = np.asarray(t, dtype=float).flatten()
    if t.size == 0:
        return np.array([])
    if not np.all(np.isfinite(t)):
        bad = t[~np.isfinite(t)]
        raise ValueError(f"dehoog: 't' contains non-finite values: {bad}")
    if np.any(t <= 0):
        bad = t[t <= 0]
        raise ValueError(f"dehoog: 't' must be strictly positive. Invalid: {bad}")

    return dehoog(lap, x, t, alpha=0.0, tol=1e-9, M=20)


# ============================================================
# 3. Dimensional scaling (real-world units)
# ============================================================
def dim(p, t):
    """
    Convert dimensionless Agarwal solution to dimensional drawdown.

    Parameters
    ----------
    p : list [a, t0, sg]
        a : slope of late-time derivative (1/m)
        t0 : intercept time
        sg : skin factor
    t : array
        Time values (s)

    Returns
    -------
    array
        Drawdown values (m)
    """
    a, t0, sg = p
    global AGA_RW, AGA_RC, AGA_R, AGA_Q

    if any(v is None for v in [AGA_RW, AGA_RC, AGA_R, AGA_Q]):
        raise ValueError("AGA_RW, AGA_RC, AGA_R, AGA_Q must be defined via pre().")

    rw = AGA_RW
    rc = AGA_RC
    r = AGA_R
    q = AGA_Q 

    # Transmissivity and storativity
    T = 0.183 * q / a
    S = 2.25 * T * t0 / r**2

    # Dimensionless groups
    cd = rc**2 / (2 * rw**2 * S)
    rd = r / rw
    td = 0.445268 * t / t0 * rd**2

    # Laplace inversion
    sd = dls([cd, rd, sg], td)

    # Final dimensional drawdown
    s = (2 / np.log(10)) * a * sd
    s[s < 0] = 0
    return s


# ============================================================
# 4. Type-curve generation
# ============================================================
def drw(cd, rd, sg):
    """
    Plot Agarwal dimensionless type curves.

    Parameters
    ----------
    cd : float
        Dimensionless wellbore storage coefficient
    rd : float
        Dimensionless radius
    sg : float
        Skin factor
    """
    t = np.logspace(-2, 6, 100)
    s = dls([cd, rd, sg], t)
    plt.loglog(t, s, label=f'rd={rd}')
    plt.xlabel('t_D')
    plt.ylabel('s_D')
    plt.title('Agarwal type curves (σ=0)')
    plt.grid(True)
    plt.legend()
    plt.show()


# ============================================================
# 5. Initial parameter estimation
# ============================================================
def gss(t, s):
    """
    Estimate initial parameters [a, t0, sg] for nonlinear fitting.

    Parameters
    ----------
    t : array
        Time values
    s : array
        Drawdown values

    Returns
    -------
    list
        Initial guess [a, t0, sg]
    """
    t = np.asarray(t)
    s = np.asarray(s)
    idx = int(len(t) * 2 / 3)

    # Use Theis solution for late-time slope/intercept
    a1, t0 = ths.gss(t[idx:-1], s[idx:-1])
    td, d = ldiffs(t, s)            
    a = np.log(10) * d[-1]
    t0 = t[-1] * np.exp(-s[-1] / d[-1])
    return [a, t0, 1.0]


# ============================================================
# 6. Report generation (plots + statistics)
# ============================================================
def rpt(p, stats, t, s, d, name='Agarwal', title='Agarwal model',
        author='Author', report='Report', filetype='pdf'):
    """
    Generate final report with fitted model results.

    Parameters
    ----------
    p : list [a, t0, sg]
        Model parameters
    stats : dict
        Fitting statistics {r2, rmse}
    t, s : arrays
        Observed time and drawdown data
    d : list [q, r, rw, rc]
        Test configuration
    name : str
        Model name
    title : str
        Plot title
    author, report : str
        Metadata
    filetype : str
        'pdf' or 'png'
    """
    # --- Data cleanup ---
    t, s = hyclean(t, s)
    a, t0, sg = p
    q, r, rw, rc = d

    # --- Hydraulic parameters ---
    T = 0.183 * q / a
    S = 2.25 * T * t0 / r**2
    cd = rc**2 / (2 * rw**2 * S)
    rd = r / rw

    # --- Model curves ---
    tplot = np.logspace(np.log10(t[0]), np.log10(t[-1]), num=100)
    sc = dim(p, tplot)
    tc, sc = hyclean(tplot, sc)

    td, sd = ldiffs(t, s, npoints=30)
    td, sd = hyclean(td, sd)

    tdc, sdc = ldiffs(tc, sc)
    tdc, sdc = hyclean(tdc, sdc)

    # --- Stats ---
    r2 = stats["r2"]
    rmse = stats["rmse"]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.loglog(t, s, 'bo', label='Data')
    ax.loglog(tc, sc, 'r-', label='Agarwal model')
    ax.loglog(td, sd, 'gx', label='Derivative')
    ax.loglog(tdc, sdc, 'm--', label='Model derivative')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Drawdown (m)')
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()

    # --- Text report ---
    text_report = (
        f" Test parameters:\n"
        f"   q = {q:.2f} m³/s, r = {r:.2f} m, rw = {rw:.3f} m, rc = {rc:.3f} m\n\n"
        f" Hydraulic parameters:\n"
        f"   T = {T:.2e} m²/s \n"
        f"   S = {S:.2e} \n"
        f"   σ (skin) = {sg:.2f} \n"
        f"   C_D = {cd:.2e}\n\n"
        f" Fit quality:\n"
        f"   R² = {r2:.3f}, RMSE = {rmse:.3f}"
    )
    plt.figtext(0.1, -0.25, text_report, ha="left", fontsize=12, family="arial")

    plt.tight_layout()

    # --- Save ---
    if filetype == 'pdf':
        fig.savefig('aga_report.pdf', bbox_inches='tight')
    else:
        fig.savefig('aga_report.png', bbox_inches='tight')

    plt.show()

