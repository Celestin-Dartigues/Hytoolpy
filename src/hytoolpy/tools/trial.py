import matplotlib.pyplot as plt
from hytoolpy.tools.derivative import ldiffs
from hytoolpy.tools.hyclean import hyclean
from hytoolpy.models import ths,jcb,grf,pcc,pcw,aga,blt,htj,thn,thc,war,warc
import numpy as np

def trial(p,t,s, name,npoints):
    '''TRIAL Display data and calculated solution together
       Syntax:
           trial(x, t, s, 'name')          
   
          name = name of the solution
          x    = vector of parameters
          t,s  = data set
       Description:
           The function trial allows to produce a graph that superposes data
           and a model. This can be used to test graphically the quality of a
           fit, or to adjust manually the parameters of a model until a
           satisfactory fit is obtained.
       Example:
           trial(p,t,s,'ths')
           trial([0.1,1e-3],t,s, 'cls')
       See also: ldf, diagnostic, fit, ths_dmo'''
    
    t,s = hyclean(t,s)
    td,sd = ldiffs(t,s, npoints)
    
    tplot = np.logspace(np.log10(t[0]), np.log10(t[len(t)-1]),  endpoint = True, base = 10.0, dtype = np.float64)
    
    
    string = name+'.dim(p,tplot)'
        
    sc = eval(string)
    
    
    tdc,dsc = ldiffs(tplot,sc,npoints)

    
    if np.mean(sd) < 0 :
        sd = [ -x for x in sd]
        dsc = [ -x for x in dsc]
    condition = np.greater(sd,0)
    td = np.extract(condition,td)
    sd = np.extract(condition,sd)
    
    condition2 = np.greater(dsc,0)
    tdc = np.extract(condition2,tdc)
    dsc = np.extract(condition2,dsc)    
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('t')
    ax1.set_ylabel('s')
    ax1.set_title('Log Log diagnostic plot')
    ax1.loglog(t, s, c='b', marker = 'o', linestyle = '')
    ax1.loglog(td,sd, c = 'r', marker = 'x', linestyle = '')
    ax1.loglog(tplot,sc, c = 'g')
    ax1.loglog(tdc,dsc, c = 'y')
    
    ax1.grid(True)

    plt.show()
    

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('t')
    ax1.set_ylabel('s')
    ax1.set_title('Semi Log diagnostic plot')
    ax1.semilogx(t, s, c='b', marker = 'o', linestyle = '')
    ax1.semilogx(td,sd, c = 'r', marker = 'x', linestyle = '')
    ax1.semilogx(tplot,sc, c = 'g')
    ax1.semilogx(tdc,dsc, c = 'y')
    
    ax1.grid(True)
    
    plt.show()

def trial2(p, t, s, name, npoints):
    """TRIAL - display data and calculated solution (log-log + semi-log)"""
    t,s = hyclean(t,s)
    td,sd = ldiffs(t,s, npoints)
    
    tplot = np.logspace(np.log10(t[0]), np.log10(t[len(t)-1]),  endpoint = True, base = 10.0, dtype = np.float64)
    
    string = name+'.dim(p,tplot)'
        
    sc = eval(string)
    
    tdc,dsc = ldiffs(tplot,sc,npoints)
    
    # correction signe si nécessaire
    if np.mean(sd) < 0 :
        sd = [ -x for x in sd]
        dsc = [ -x for x in dsc]
    condition = np.greater(sd,0)
    td = np.extract(condition,td)
    sd = np.extract(condition,sd)
    
    condition2 = np.greater(dsc,0)
    tdc = np.extract(condition2,tdc)
    dsc = np.extract(condition2,dsc) 
    
    fig, axs = plt.subplots(1,2, figsize=(12,5))
    
    # log-log plot
    axs[0].loglog(t, s, 'bo', label='data')
    axs[0].loglog(td, sd, 'rx', label='derivative')
    axs[0].loglog(tplot, sc, 'g', label='fit')
    axs[0].loglog(tdc, dsc, 'y', label='fit derivative')
    axs[0].set_xlabel('t'); axs[0].set_ylabel('s')
    axs[0].set_title('Log-Log diagnostic plot')
    axs[0].grid(True)
    axs[0].legend()
    
    # semi-log plot
    axs[1].semilogx(t, s, 'bo', label='data')
    axs[1].semilogx(td, sd, 'rx', label='derivative')
    axs[1].semilogx(tplot, sc, 'g', label='fit')
    axs[1].semilogx(tdc, dsc, 'y', label='fit derivative')
    axs[1].set_xlabel('t'); axs[1].set_ylabel('s')
    axs[1].set_title('Semi-Log diagnostic plot')
    axs[1].grid(True)
    axs[1].legend()
    
    plt.tight_layout()
    return fig
