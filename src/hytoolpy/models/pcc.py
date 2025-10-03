# Papadopoulos constant head boundary

import numpy as np
from scipy.special import kv
import matplotlib.pyplot as plt

from hytoolpy.tools.laplace import stefhest
from hytoolpy.tools.derivative import ldiffs, ldiff
from hytoolpy.tools.hyclean import hyclean

# 1. Laplace domain
def lap(x, p):
    cd, rd = x
    s = np.sqrt(p)
    num = kv(0, rd * s)
    den = p * (s * kv(1, s) + cd * p * kv(0, s))
    return num / den

# 2. Inversion
def dls(x, t):
    return np.array([stefhest(lap, x, ti) for ti in t])

# 3. Dimensionnel
def dim(p, t):
    a, t0 = p
    global AGA_RW, AGA_RC, AGA_R, AGA_Q
    rw, rc, r, q = AGA_RW, AGA_RC, AGA_R, AGA_Q
    T = 0.1832339 * q / a
    S = 2.2458394 * T * t0 / r**2
    cd = rc**2 / (2 * rw**2 * S)
    rd = r / rw
    td = 0.445268 * t / t0 * rd**2
    sd = dls([cd, rd], td)
    return [0.868589 * a * si for si in sd]

# 4. Courbes types
def drw(cd):
    t = np.logspace(-2, 6)
    s = dls([cd, 1.0], t)
    plt.loglog(t, s, label=f'cd={cd}')
    plt.xlabel('t_D')
    plt.ylabel('s_D')
    plt.title('Papadopoulos type curves')
    plt.legend()
    plt.grid(True)
    plt.show()

# 5. Guess
def gss(t, s):
    t = np.asarray(t)
    s = np.asarray(s)
    a = (s[-1] - s[0]) / (np.log10(t[-1]) - np.log10(t[0]))
    t0 = t[0]
    return [a, t0]

# 6. Rapport
def rpt(p, t, s, d, npoint, name='Papadopoulos', title='Papadopoulos model', author='Author', report='Report', filetype='pdf'):
    t,s = hyclean (t,s)
    a, t0 = p
    q, r, rw, rc = d
    
    T = 0.1832339 * q / a
    S = 2.2458394 * T * t0 / r**2
    cd = rc**2 / (2 * rw**2 * S)
    rd = r / rw

    # Courbe du modèle sur grille régulière
    tplot = np.logspace(np.log10(t[0]), np.log10(t[-1]), base=10.0, num=100)
    sc = dim(p, tplot)
    tc, sc = hyclean(tplot, sc)

    
    #calculate the derivative of the data
    td, sd = ldiffs(t,s, npoint)
    #keep only positive derivatives
    td, sd = hyclean(td,sd)
    
    #compute the derivative of the model
    tdc,sdc = ldiffs(tc,sc,npoint)
    #keep only positive derivatives
    tdc,sdc = hyclean(tdc,sdc)
    
    # Calcul des erreurs statistiques
    residuals = np.subtract(dim(p, t), s)
    mr = np.mean(residuals)
    sr = 2 * np.std(residuals)  # 2σ
    rms = np.sqrt(np.mean(residuals**2))

    # Création figure + agrandissement de la marge droite
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)

    # Graphique principal
    ax = fig.add_subplot(gs[0])
    ax.loglog(t, s, 'bo', label='Data')
    ax.loglog(tc, sc, 'r-', label='Papadopoulous cooper model')
    ax.loglog(td, sd, 'gx', label='Derivative')
    ax.loglog(tdc, sdc, 'm--', label='Model derivative')
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Drawdown (m)')
    ax.grid(True)
    ax.legend()

    # Zone à droite pour les textes
    ax_info = fig.add_subplot(gs[1])
    ax_info.axis('off')  # on désactive l'affichage des axes

    textstr = '\n'.join([
        f'Author: {author}',
        f'Report: {report}',
        '',
        f'T = {T:.2e} m²/s',
        f'S = {S:.2e}',
        f'cd = {cd:.2f}',
        f'Mean residual = {mr:.3g}',
        f'2σ = {sr:.3g}',
        f'RMS = {rms:.3g}'
    ])
    ax_info.text(0, 1, textstr, va='top', ha='left', fontsize=10)

    if filetype == 'pdf':
        fig.savefig('papadopoulos_report.pdf', bbox_inches='tight')
    else:
        fig.savefig('papadopoulos_report.png', bbox_inches='tight')
    plt.show()
