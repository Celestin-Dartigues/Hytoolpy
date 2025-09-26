import numpy as np

def birsoy_time(t, s, q):
    """
    BIRSOY_TIME - Calcule le temps équivalent de Birsoy et Summers (1980)

    Paramètres :
    - t : tableau des temps (1D)
    - s : tableau des rabattements (1D)
    - q : matrice 2D des périodes de pompage (colonnes = [temps_fin, débit])

    Retourne :
    - tb : temps équivalent (1D)
    - sb : rabattement réduit = s / qn
    """
    t = np.asarray(t)
    s = np.asarray(s)
    q = np.asarray(q)

    if q.shape[1] != 2:
        raise ValueError("La matrice q doit avoir 2 colonnes : temps et débit.")

    if q.shape[0] < 2:
        print("Avertissement : la matrice q contient une seule ligne. Test à débit variable recommandé.")

    pe = np.ones_like(t, dtype=int)
    for i in range(len(q) - 1, -1, -1):
        j = np.where(t <= q[i, 0])[0]
        pe[j] = i + 1  # car Python indexe à partir de 0

    lq = q[pe - 1, 1]  # débit associé à chaque temps
    dq = np.diff(q[:, 1], prepend=q[0, 1])
    st = np.insert(q[:-1, 0], 0, 0)

    tb = np.ones_like(t)
    for j in range(len(t)):
        n = pe[j]
        for i in range(n):
            tb[j] *= (t[j] - st[i]) ** (dq[i] / q[n - 1, 1])

    tb_sorted_idx = np.argsort(tb)
    tb = tb[tb_sorted_idx]
    sb = s[tb_sorted_idx] / lq[tb_sorted_idx]
    return tb, sb
