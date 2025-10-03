"""
Theis (1941) model with a constant head boundary.
Constant rate pumping test in a homogeneous confined aquifer bounded
by a constant head boundary.


This module provides:
    - Laplace domain solution (lap)
    - Dimensionless time solution (dls)
    - Dimensional solution (dim)
    - Initial parameter guess (gss)
    - Reporting and plotting (rpt)

Reference: Theis, C.V., 1941. The effect of a well on the flow of a
nearby stream. Transactions of the American Geophysical Union, 22(3):
734-738.  
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
    '''THC_DIM - Theis (1941) model with a constant head boundary.

 Syntax: s = thc.dim(p,t)

   p(1) = a  = slope of Jacob Straight Line
   p(2) = t0 = intercept of the first segment of straight line
   p(3) = ti = time of intersection between the 2 straight lines

 Description:
   Computes the drawdown at time t for a constant rate pumping test in 
   a homogeneous confined aquifer bounded by a constant head boundary.

  Reference: Theis, C.V., 1941. The effect of a well on the flow of a
  nearby stream. Transactions of the American Geophysical Union, 22(3):
  734-738.  

 Example:
   s = thc.dim(p,t)
'''
    a = p[0]
    
    s = ths.dim([a,p[1]],t) - ths.dim([a,p[2]],t)
    
    return s


# =========================
# 2. Dimensionless solution
# =========================

def dls(p,t):
    '''THC_DLS - Theis dimensionless drawdown with a constant head boundary

 Syntax: s = thc.dls(p,t)

   p(0) = r1/r2  
   with r1 = radius to the pumping well 
   and  r2 = radius to the image well 

   provides the dimensionless drawdown at reduced time t
'''
    t = 0.25/t
    
    s = 0.5*sp.special.expn(1,t)-0.5*sp.special.expn(1,t*p[0]**2)
    
    return s
    

    
# =========================
# 3. Funcion f(rd) 
# =========================
def frd(rd):
    '''THC_FRD - Function f(rd)

 Syntax: frd = thc.frd(rd)

   rd  = Dimensionless radius
   frd = 2 ln(rd) / ( rd^2 -1 ) rd^(2 rd^2/(1-rd^2))'''
    
    if type(rd) == int:
        if rd == 1:
            rd = 1.00001
        
        rd2 = rd**2
        rd21 = rd2-1
    
        frd = np.log(rd2)/(rd21*np.pow(rd,-2*rd2/rd21))
        
        return frd

    if type(rd) == float:
        if rd == 1:
            rd = 1.00001
        
        rd2 = rd**2
        rd21 = rd2-1
    
        frd = np.log(rd2)/(rd21*np.pow(rd,-2*rd2/rd21))
        
        return frd        

    elif type(rd) == list :
        rd2 = np.power(rd,2)
        rd21 = []
        for i in range(0,len(rd2)):
            rd21.append(rd2[i]-1)
        
        frd = []
        
        rdpow = np.power(rd, -2*rd2/rd21)
        
        rd2log = np.log(rd2)
        frd = np.divide(rd2log,np.multiply(rd21,rdpow))
            
        
        return frd    
    else :
        print('Error, please enter a valid variable (list, int or float)')
    
# =========================
# 4. Initial guess
# =========================

def gss(t,s):
    '''THC_GSS - First guess for the parameters of the Theis model with a constant head boundary.

 Syntax:  p = thc.gss(t,s)

   p(1) = a  = slope of Jacob straight line 
   p(2) = t0 = iintercept of the first segment of straight line 
   p(3) = ti = time of intersection between the 2 straight lines

   t    = time
   s    = drawdown

 Description:
   First guess for the parameters of theis solution with a constant head
   boundary.

 See also: thc_dim, thc_dmo, thc_rpt
  '''
    t = np.asarray(t)
    s = np.asarray(s)
    
    td, d = ldiffs(t, s, npoints=40)
    
    sl = float(np.amax(s))
    du = float(np.amax(d))
    i = np.argmax(d)
    
    tu = float(td[i])
    
    # Indices where t > tu
    condition = np.greater(t, tu)
    indices = np.where(condition)[0]
    if len(indices) == 0:
        raise ValueError("Pas de valeur de temps t > tu trouvée pour calculer ta et sa.")
    
    w = indices[0]
    ta = t[w]
    sa = s[w]
    
    ts = td[-1]
    ds = d[-1]
    
    # Calculate model parameters
    a = sl / du
    rd = ird(a)
    
    b = sl / (2 * np.log10(rd))
    
    # Straight line Jacob
    if rd < 50:
        t0 = 2 * tu * np.log(rd) / (0.5625 * ((rd ** 2) - 1))
    else:
        t0 = ta * (10 ** (-sa / b))
    
    t1 = (rd ** 2) * t0
    ti = (t1 ** 2) / t0
    
    return [b, t0, ti]
    
# =========================
# 5. Inverse function of Frd
# =========================

def ird(frd1):
    '''THC_IRD - Function inverse of f(rd)

Syntax: rd = thc.ird(frd)

rd  = Dimensionless radius
frd = 2 ln(rd) / ( rd^2 -1 ) rd^(2 rd^2/(1-rd^2))'''    
    
    if type(frd1) == int :
        if frd1 < 2.71828182850322 :
            print('Problem in the inversion of Rd: thc_ird')
            rd = 1.000001
            return rd
        else :
            rd = np.exp(frd1/2)
            
            if rd < 50 :
                y1 = np.arange(1,60.005,0.05)
                y1[0] = 1.000001
                y = []
                for i in range(0,len(y1)):
                    y.append(y1[i])
                
                x = frd(y)
                rd = sp.interpolate.interp1d(x,y)(frd1)
                return rd
            else :
                return rd
            
    if type(frd1) == float :
        if frd1 < 2.71828182850322 :
            print('Problem in the inversion of Rd: thc_ird')
            rd = 1.000001
            return rd
        else :
            rd = np.exp(frd1/2)
            
            
            if rd < 50 :
                y1 = np.arange(1,60.005,0.05)
                y1[0] = 1.000001
                y = []
                for i in range(0,len(y1)):
                    y.append(y1[i])
                
                x = frd(y)
                rd = sp.interpolate.interp1d(x,y)(frd1)
                
                return rd
            else :
                return rd

    else :
        print('Error. Please enter a int or a float')


# =========================
# 6. Conversion parameters 
# =========================

def p2x(p):
    '''%THC_P2X - Conversion of parameters for constant head case

 Syntax: x = thc.p2x(p)

  p(1) = sl  = late time drawdown at the plateau
  p(2) = du  = maximum derivative 
  p(3) = tu  = time of the maximum

  x(1) = a   = slope of Jacob straight line
  x(2) = t0  = intercept of the first Jacob straight line
  x(3) = t1  = intercept of the second Jacob straight line



 See also:
'''
    sl = p[0]
    du = p[1]    
    tu = p[2]
    
    sl = np.float(sl)
    du = np.float(du)
    tu = np.float(tu)
    
    
    a = sl/du
    
    rd = ird(a)
    
    x = []
    
    x.append(0.5*sl/np.log(rd))
    x.append(2/0.5626*tu*np.log(rd)/(np.pow(rd,2)-1))
    x.append(np.pow(rd,2)*x[1])
    
    return x
    
# =========================
# 7. Discharge computation
# =========================

def std(l,T,r0):
    '''THC_STD - Computes discharge rate for a well close to a constant head boundary

 Syntax: q = thc.std( l, T, r0 )

   l  = Distance to the hydraulic boundary
   T  = Transmissivity
   r0 = radius of the well

   Nota : the head difference is supposed to be l

 Description:
   The flow field is a half infinite space with a constant head boundary.
   The aquifer is supposed to be confined, homogeneous and the well fully
   penetrating the aquifer.

   The calculation is based on the Dupuit's solution with an image well.
  This equation is also known as the Goodman formula (1965).

 Reference:
   Goodman, R., 1965. Groundwater inflows during tunnel driving.
   Engineering Geology, 2(2): 39-56. 

 Example:
   q=thc.std(100,1e-2,10)

'''

    q = 2*np.pi*T*l/(np.log(2*l/r0))
    
    return q


# =========================
# 8. Reporting
# =========================

def rpt(p, stats, t, s, d, npoint,
        name='Theis',
        title='Theis model',
        author='Author', report='Report',
        filetype='pdf',
        ax=None,
        color=None):
    """
    Generate report for Theis with a constant head boundary solution.
    Can be used standalone or combined with other models (multi-plot).

    Parameters
    ----------
    p : list
        Model parameters [a, t0, ti]
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
          
          Reference: Theis, C.V., 1941. The effect of a well on the flow of a 
          nearby stream. Transactions of the American Geophysical Union, 22(3): 734-738."""

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

    td, sd = ldiffs(t, s, npoints=30)
    td, sd = hyclean(td, sd)

    tdc, sdc = ldiffs(tc, sc)
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
    ax.loglog(tc, sc, 'r-', label='Theis model with constant head boundary')
    ax.loglog(td, sd, 'gx', label='Derivative')
    ax.loglog(tdc, sdc, 'm--', label='Model derivative')
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
        f"   Distance to boundary = {Ri:.2f}m \n\n"
        f" Fit quality :\n"
        f"   R² = {r2:.3f}, RMSE = {rmse:.3f}"
    )

    fig.text(0.1, -0.2, text_report, ha="left", fontsize=10, family="arial",
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

    fig.tight_layout()

    # Save
    if filetype == 'pdf':
        fig.savefig('Theis_constant_boundary_report.pdf', bbox_inches='tight')
    else:
        fig.savefig('Theis_constant_boundary_report.png', bbox_inches='tight')

    return fig,ax
