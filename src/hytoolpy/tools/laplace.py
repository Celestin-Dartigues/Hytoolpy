import math
import numpy as np

def stehfest_coeff(i, N=12):
    Vi = 0
    for k in range(int((i + 1) / 2), min(i, N // 2) + 1):
        num = k**(N / 2) * math.factorial(2 * k)
        denom = (
            math.factorial(N // 2 - k)
            * math.factorial(k)
            * math.factorial(k - 1)
            * math.factorial(i - k)
            * math.factorial(2 * k - i)
        )
        Vi += num / denom
    Vi *= (-1) ** (i + N // 2)
    return Vi


def stefhest(func, p, t, N=12):
    """
    Laplace inversion by stefhest method.

    Parameters:
    - func : Laplace funcion (ex: grf_lap)
    - p : model parameters
    - t : inversion time
    - N : Stefhest term number (par défaut 12)

    Return
    - Inversed Lap function
    """
    resultat = 0
    for i in range(1, N + 1):
        pi = i * math.log(2) / t
        resultat += stehfest_coeff(i, N) * func(p, pi)
    return (math.log(2) / t) * resultat




import numpy as np

def dehoog(F, x, t, alpha=0.0, tol=1e-9, M=20):
    """
    Dehoog inversion based on Hytool toolbox (de Hoog, Knight & Stokes, 1982).
    - F : callable F(x, s) where s can be complex number (return column vector)
    - x : additional parameters
    - t : time vector >0
    - alpha, tol, M : Algorithm parameters
    Retour : np.ndarray (mêmes ordres que t)
    """
    t = np.asarray(t, dtype=float).flatten()
    if t.size == 0:
        return np.array([])

    # validation initiale
    if not np.all(np.isfinite(t)):
        raise ValueError(f"dehoog: 't' contient NaN/Inf : {t[~np.isfinite(t)]}")
    if np.any(t <= 0):
        raise ValueError(f"dehoog: 't' doit être strictement positif (>0). Valeurs non valides: {t[t<=0]}")

    allt = t.copy()
    logallt = np.log10(allt)
    if np.any(~np.isfinite(logallt)):
        raise ValueError("dehoog: log10(t) contient NaN/Inf (vérifier t).")

    iminlogallt = int(np.floor(np.min(logallt)))
    imaxlogallt = int(np.ceil(np.max(logallt)))

    f_out = np.zeros_like(allt, dtype=float)
    idx_write = 0  # position où écrire dans f_out (conserve ordre original via mask)

    for ilogt in range(iminlogallt, imaxlogallt + 1):
        mask = (logallt >= ilogt) & (logallt < ilogt + 1)
        if not np.any(mask):
            continue

        t_piece = allt[mask]
        nt = len(t_piece)

        # paramètres de contour
        T = 2.0 * np.max(t_piece)
        gamma = alpha - np.log(tol) / (2.0 * T)

        # nombre de termes (MATLAB: run = 0:2*M -> 2M+1 termes)
        L = 2 * M + 1
        run = np.arange(0, L)  # 0 .. 2M

        # vecteur s = gamma + i*pi*run/T
        s = gamma + 1j * np.pi * run / T

        # évaluer F en s (obligatoirement renvoyer un vecteur de longueur L)
        try:
            a = np.array(F(x, s), dtype=complex).flatten()
        except Exception as e:
            raise RuntimeError(f"dehoog: erreur lors de l'appel F(x,s): {e}")

        # s'assurer que a a bien la longueur L ; sinon pad/tronque
        if a.size != L:
            # log utile pour debug
            # print(f"[dehoog] ajustement longueur a : {a.size} -> {L}")
            if a.size < L:
                a = np.pad(a, (0, L - a.size), mode="constant", constant_values=0+0j)
            else:
                a = a[:L]

        # a(1) est le terme 0 halved
        a[0] = a[0] / 2.0

        # initialisation des tables (indices Python 0-based)
        # e : (2*M+1) x (M+1)
        e = np.zeros((L, M + 1), dtype=complex)
        # q : (2*M) x (M+1)  (col 0 non utilisée comme dans Matlab)
        q = np.zeros((L - 1, M + 1), dtype=complex)

        # q(:,1) <-- a(2:2*M+1)/a(1:2*M) in matlab -> python q[:,1] = a[1:L]/a[0:L-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = a[0:L - 1]
            q[:, 1] = np.divide(a[1:L], denom, out=np.full_like(a[1:L], np.nan + 0j), where=np.abs(denom) > 0)

        # build e and q columns following matlab indices carefully
        # loop r = 2 : M+1 in matlab -> python r_idx runs 1 .. M
        for r_idx in range(1, M + 1):
            # n = 2*(M - r_idx) + 1  (number of elements to fill in e[:, r_idx])
            n = 2 * (M - r_idx) + 1
            if n >= 1:
                # e[0 : n, r_idx] = q[1: n+1, r_idx] - q[0:n, r_idx] + e[1:n+1, r_idx-1]
                # safeguard lengths
                left = q[1:n + 1, r_idx] if (1 + n) <= q.shape[0] else q[1:q.shape[0], r_idx]
                mid = q[0:n, r_idx]
                right = e[1:n + 1, r_idx - 1]
                # pad to same length if necessary
                Lmin = min(len(left), len(mid), len(right))
                if Lmin > 0:
                    e[0:Lmin, r_idx] = left[:Lmin] - mid[:Lmin] + right[:Lmin]

            # compute q column for rq = r+1 (only if r < M+1)
            if r_idx < M:
                rq_idx = r_idx + 1
                m = 2 * (M - r_idx)  # length for q slice
                if m > 0:
                    num = q[1:m + 1, rq_idx - 1] * e[1:m + 1, rq_idx - 1]
                    den = e[0:m, rq_idx - 1]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        q[0:m, rq_idx] = np.divide(num, den, out=np.full_like(num, np.nan + 0j), where=np.abs(den) > 0)

        # build d vector (length 2*M+1)
        d = np.zeros((L,), dtype=complex)
        d[0] = a[0]
        # d indices: d[1:2:2*M] = -q[0,1:M+1]
        # python slice: start=1, stop=2*M, step=2 (stop exclusive so stop=2*M gives last index 2*M-1)
        d[1:2 * M:2] = -q[0, 1:M + 1]
        # d[2:2:2*M+1] = -e[0,1:M+1]
        d[2:2 * M + 1:2] = -e[0, 1:M + 1]

        # Build A and B (rows = 2*M+2, cols = nt)
        A = np.zeros((2 * M + 2, nt), dtype=complex)
        B = np.zeros((2 * M + 2, nt), dtype=complex)

        # indexing mirroring matlab:
        A[1, :] = d[0]  # A(2,:) = d(1)
        B[0:2, :] = 1.0  # B(1:2,:) = ones

        # z row vector = exp(i*pi*t'/T)
        z = np.exp(1j * np.pi * t_piece / T)  # shape (nt,)

        # recurrence: for n = 3 : 2*M+2 matlab -> python n_idx = 2 .. 2*M+1 inclusive
        for n_idx in range(2, 2 * M + 2):
            # A[n_idx,:] = A[n_idx-1,:] + d[n_idx-1] * z * A[n_idx-2,:]
            A[n_idx, :] = A[n_idx - 1, :] + d[n_idx - 1] * (z * A[n_idx - 2, :])
            B[n_idx, :] = B[n_idx - 1, :] + d[n_idx - 1] * (z * B[n_idx - 2, :])

        # double acceleration (careful with indices)
        # h2M = .5 * (1 + ( d(2*M)-d(2*M+1) )*z )  (MATLAB 1-based)
        # -> python indices: d[2*M-1] and d[2*M]
        with np.errstate(invalid='ignore'):
            h2M = 0.5 * (1.0 + (d[2 * M - 1] - d[2 * M]) * z)
            sqrt_arg = 1.0 + d[2 * M] * z / (h2M ** 2)
            # force complex to avoid nan from sqrt of negative real
            sqrt_arg = sqrt_arg.astype(complex)
            R2Mz = -h2M * (1.0 - np.sqrt(sqrt_arg))

        A[2 * M + 1, :] = A[2 * M, :] + R2Mz * A[2 * M - 1, :]
        B[2 * M + 1, :] = B[2 * M, :] + R2Mz * B[2 * M - 1, :]

        # safe division A/B
        with np.errstate(divide='ignore', invalid='ignore'):
            denomAB = B[2 * M + 1, :]
            ratio = np.divide(A[2 * M + 1, :], denomAB, out=np.full_like(A[2 * M + 1, :], np.nan + 0j), where=np.abs(denomAB) > 1e-300)

        fpiece = (1.0 / T) * np.exp(gamma * t_piece) * np.real(ratio)
        # replace nan/inf by zero to be safe
        fpiece = np.nan_to_num(fpiece, nan=0.0, posinf=0.0, neginf=0.0)

        # store results in correct positions
        f_out[mask] = fpiece

    return np.array(f_out)



