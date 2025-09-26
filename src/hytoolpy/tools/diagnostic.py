"""Diagnostic plot for pumping test analysis."""

import numpy as np
import matplotlib.pyplot as plt
from hytoolpy.tools.derivative import ldiffs, ldiff, ldiffb

def diagnostic(t, s, d=20, m='s'):
    """
    diagnostic - Crée un graphique diagnostic des données (log-log du rabattement et de sa dérivée)

    Paramètres :
    -----------
    t : array-like
        Temps mesuré (en s)
    s : array-like
        Rabattement mesuré (en m)
    d : int, optionnel
        Paramètre utilisé par certaines méthodes :
        - nombre de points pour la méthode 's' (spline)
        - distance de lag pour la méthode 'b' (bourdet)
        - ignoré si méthode 'd' (directe)
    m : str, optionnel
        Méthode de calcul de la dérivée :
        - 's' : spline
        - 'd' : direct
        - 'b' : bourdet

    Exemple :
    ---------
    diagnostic(t, s)              # spline par défaut
    diagnostic(t, s, 30)          # spline avec 30 points
    diagnostic(t, s, 20, 'd')     # dérivée directe
    """

    # Nettoyage : suppression des valeurs de temps négatif ou nul
    t = np.array(t)
    s = np.array(s)
    mask = t > 0
    t = t[mask]
    s = s[mask]

    # Calcul de la dérivée
    if m == 's':
        td, sd = ldiffs(t, s, npoints=d)
    elif m == 'd':
        td, sd = ldiff(t, s)
    elif m == 'b':
        td, sd = ldiffb(t, s, d=d)
    else:
        print("❌ ERREUR: Méthode de dérivée inconnue :", m)
        print("Méthodes possibles : 's', 'd', 'b'")
        return

    # Filtrer les dérivées positives pour l'affichage
    td = np.array(td)
    sd = np.array(sd)
    mask_deriv = sd > 0
    td = td[mask_deriv]
    sd = sd[mask_deriv]

    # Tracé log-log
    fig, ax = plt.subplots()
    ax.loglog(t, s, 'o', label='Drawdown')
    ax.loglog(td, sd, '+', label='Derivative')
    ax.set_xlabel('Time')
    ax.set_ylabel('Drawdown and log derivative')
    ax.legend(loc='best')
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_title('Diagnostic plot')
    fig.tight_layout()

    return fig
