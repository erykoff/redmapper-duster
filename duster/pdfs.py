import numpy as np


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


# Use scipy.stats.norm for p_gaussian
