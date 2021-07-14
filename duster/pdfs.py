import numpy as np
from scipy.stats import norm


def p_dust(rho_0, b, rho_min, rho_vals):
    """Compute p_dust(rho | rho_0, b, rho_min)

    Parameters
    ----------
    rho_0 : `float`
        The value of rho_0
    b : `float`
        The value of b
    rho_min : `float`
        The value of rho_min
    rho_vals : `np.ndarray`
        Array of rho values to compute p_dust

    Returns
    -------
    p_dust : `np.ndarray`
        p_dust evaluated at rho_vals given rho0, b, rho_min.
    """
    lo = (rho_vals <= rho_min)
    hi = ~lo

    B = (b + 1)/(rho_0*((rho_min/rho_0)**b)*(b + 1 + (rho_min/rho_0)))

    p_dust = np.zeros(rho_vals.size)
    p_dust[lo] = B*((rho_vals[lo]/rho_0)**b)
    p_dust[hi] = B*((rho_min/rho_0)**b)*(np.exp(-(rho_vals[hi] - rho_min)/rho_0))

    return p_dust


def compute_normalized_rho_pars(u, b):
    """Compute normalized rho_0, rho_min given b, u=rho_min/rho_0

    Parameters
    ----------
    u : `float`
        u = rho_min/rho_0 parameter.
    b : `float`
        b parameter

    Returns
    -------
    rho_0, rho_min : (`float`, `float`)
    """
    rho_0 = ((b + 2)/(b + 1))*((u + b + 1)/(u**2. + (b + 2)*(u + 1)))
    rho_min = rho_0*u

    return rho_0, rho_min


class Pabgs(object):
    """Class to compute P(rho_obs | alpha, beta, gamma, sigma2)

    Parameters
    ----------
    rho_vals : `np.ndarray`
        Array of values for integration.
    rho_0 : `float`
        Model rho_0 parameter.
    rho_min : `float`
        Model rho_min parameter.
    b : `float`
        Model b parameter.
    """
    def __init__(self, rho_vals, rho_0, rho_min, b):
        self.rho_vals = rho_vals
        self.rho_0 = rho_0
        self.rho_min = rho_min
        self.b = b
        self.P_dust = p_dust(self.rho_0, self.b, self.rho_min, self.rho_vals)

    def __call__(self, alpha, beta, gamma, sigma2, rho_obs):
        """Compute P(rho_obs | alpha, beta, gamma, sigma2)

        Parameters
        ----------
        alpha : `float`
            Model alpha.
        beta : `float`
            Model beta.
        gamma : `float`
            Model gamma.
        sigma2 : `float`
            Model sigma**2.
        rho_obs : `float`
            Observed (and possibly filtered) rho.

        Returns
        -------
        p : `float`
            Integrated probability.
        """
        rho_obs_mean = alpha*(self.rho_vals**beta) + gamma
        P_rho_obs_rho = norm.pdf(rho_obs, loc=rho_obs_mean, scale=np.sqrt(sigma2))

        return np.trapz(P_rho_obs_rho*self.P_dust, x=self.rho_vals)
