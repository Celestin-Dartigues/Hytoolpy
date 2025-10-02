import numpy as np
from scipy.optimize import leastsq
import importlib
from hytoolpy.models import ths,jcb,grf,pcc,pcw,aga,blt,htj,thn,thc,war,warc


def fit(p0,t,s,name):
    """
    Least square fitting with statistiques (R², RMSE, std des paramètres).

    Parameters
    ----------
    p0 : list ou array
        initial guess of the parameters.
    t, s : array
        time and drawdown.
    name : str
        Model name (ex: 'jcb', 'grf', 'war'...).

    Return
    ------
    results : dict
        with :
        - "params" : adjusted parameters
        - "p_std" : standard deviation
        - "r2" : regression coefficient
        - "rmse" : quadratic mean errors
        - "residuals" : residuals (observé - simulé)
    """

    def residual(p, t, s, name):
        string = name + '.dim(p, t)'
        sc = eval(string)
        return s - sc  # résidus (observé - simulé)

    def fit2(residual, p0, t, s):
        p, cov_x, infodict, mesg, ier = leastsq(
            residual, p0, args=(t, s, name), full_output=True
        )
        if cov_x is not None:
            p_std = np.sqrt(np.diag(cov_x))
        else:
            p_std = np.full_like(p, np.nan)
        return p, p_std, infodict

    # Fit
    p, p_std, infodict = fit2(residual, p0, t, s)

    # Résidus
    residuals = infodict["fvec"]

    # Statistiques
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((s - np.mean(s))**2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(residuals**2))
    stats = {"p_std": p_std,
        "r2": r2,
        "rmse": rmse,}

    return p, stats



import emcee
import importlib


def fit_mcmc(p0, t, s, name, nwalkers=32, nsteps=500, burn_in=100):

    """
    Fitting with Bayesian inference.

    Parameters
    ----------
    p0 : list ou array
        Initial guess.
    t, s : array
        Time and drawdown.
    name : str
        Model name (ex: 'jcb', 'grf', 'war'...).
    nwalkers, nsteps, burn_in : int
        Paramètres MCMC.

    Return
    ------
    results : dict
        Contient les paramètres ajustés, incertitudes et stats :
        - "samples" : ndarray of previous samples
        - "params_median" : Median of the adjusted parameters
        - "params_std" : Standard deviation of the adjusted parameters
        - "r2" : regression coefficient
        - "rmse" : quadratic mean errors
        - "residuals" : residuals (observé - simulé)
    """

    # Charger dynamiquement le modèle demandé
    model = importlib.import_module("hytoolpy.models." + name)

    ndim = len(p0) + 1 # paramètres du modèle + sigma incertitude

    # Prior : log-probabilité a priori
    def log_prior(theta):
        *p, sigma = theta
        if sigma <= 0:
            return -np.inf
        if any(pi <= 0 for pi in p):  # paramètres physiques positifs
            return -np.inf
        return 0.0

    # Vraisemblance : erreurs gaussiennes avec variance sigma²
    def log_likelihood(theta):
        *p, sigma = theta
        sc = np.asarray(model.dim(p, t))
        residuals = s - sc
        return -0.5 * np.sum((residuals / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))

    # Postérieure = prior + likelihood
    def log_posterior(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    # Initialisation des walkers autour de p0
    base_params = list(p0) + [np.std(s)]   
    p0_walkers = [np.array(base_params) * (1 + 1e-4 * np.random.randn(ndim))
    for _ in range(nwalkers)]


    # Sampler MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
    sampler.run_mcmc(p0_walkers, nsteps, progress=True)

    # Echantillons postérieurs après burn-in
    samples = sampler.get_chain(discard=burn_in, flat=True)

    # Médianes et écarts-types
    p = np.median(samples, axis=0)
    p_std = np.std(samples, axis=0)

    # Simulation avec les paramètres médians
    sc = np.asarray(model.dim(p[:-1], t))
    residuals = s - sc

    # Statistiques
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((s - np.mean(s))**2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(residuals**2))
    stats = {"p_std": p_std,
        "r2": r2,
        "rmse": rmse,}

    return p[:-1], stats



