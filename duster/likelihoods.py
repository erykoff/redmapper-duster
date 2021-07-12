import numpy as np

from .pdfs import p_dust, compute_normalized_rho_pars, Pabgs
from .constants import (RHO_INTEGRAL_OFFSET,
                        RHO_INTEGRAL_MIN,
                        RHO_INTEGRAL_MAX,
                        RHO_INTEGRAL_STEP2)


class RhoModelLikelihood(object):
    """Class for computing the likelihood of our rho model.

    Parameters
    ----------
    rho_map1 : `healsparse.HealSparseMap`
        Normalized rho map 1
    rho_map2 : `healsparse.HealSparseMap`
        Normalized rho map 2
    bounds : `list` [`tuple`]
        A list of tuples with low and high bounds for each
        parameters (b, u).
    minimize : `bool`, optional
        Minimize likelihood rather than default to maximize.
    """
    def __init__(self, rho_map1, rho_map2, bounds, minimize=False):
        vpix1 = rho_map1.valid_pixels
        vpix2 = rho_map2.valid_pixels

        self.yvals1 = rho_map1[vpix1]
        self.yvals2 = rho_map2[vpix2]
        self.bounds = bounds
        self.npars = len(bounds)
        if minimize:
            self.sign = -1.0
        else:
            self.sign = 1.0

    def __call__(self, pars):
        """Compute the likelihood.  Returns -inf if out of bounds.

        Parameters
        ----------
        pars : `list`
            List of parameters (u, b).

        Returns
        -------
        lnlike : `float`
            ln(likelihood) of the data given the model.
        """
        for i in range(self.npars):
            if pars[i] < self.bounds[i][0] or pars[i] > self.bounds[i][1]:
                return -self.sign*np.inf

        b = 10.0**pars[0]
        u = 10.0**pars[1]

        rho_0, rho_min = compute_normalized_rho_pars(u, b)

        p_dust_1 = p_dust(rho_0, b, rho_min, self.yvals1)
        p_dust_2 = p_dust(rho_0, b, rho_min, self.yvals2)

        lnlike = np.sum(np.log(np.clip(p_dust_1, 1e-20, None)))
        lnlike += np.sum(np.log(np.clip(p_dust_2, 1e-20, None)))

        return self.sign*lnlike


class DebiasLikelihood(object):
    """Class for computing the rho debiasing model.

    Parameters
    ----------
    rho_map_in : `healsparse.HealSparseMap`
        Biased rho map to debias.
    bounds : `list` [`list`]
        List of 2-element list of parameter bounds.
    rho_0 : `float`
        Model rho_0 parameter.
    rho_min : `float`
        Model rho_min parameter.
    b : `float`
        Model b parameter.
    minimize : `bool`, optional
        Minimize likelihood rather than default to maximize.
    """
    def __init__(self, rho_map_in, bounds, rho_0, rho_min, b, minimize=False):
        vpix = rho_map_in.valid_pixels

        self.rho_in_vals = rho_map_in[vpix]
        rho_int_vals = np.arange(RHO_INTEGRAL_MIN,
                                 RHO_INTEGRAL_MAX,
                                 RHO_INTEGRAL_STEP2) + RHO_INTEGRAL_OFFSET
        self.p_abgs = Pabgs(rho_int_vals, rho_0, rho_min, b)
        self.bounds = bounds
        self.npars = len(bounds)
        if minimize:
            self.sign = -1.0
        else:
            self.sign = 1.0

    def __call__(self, pars):
        """Compute likelihood given parameters.

        Parameters
        ----------
        pars : array-like
            Parameter array, [log10(alpha), log10(beta), gamma, log10(sigma**2)].

        Returns
        -------
        likelihood : `float`
            Will return negative likelihood if self.minimize = True
        """
        for i in range(self.npars):
            if pars[i] < self.bounds[i][0] or pars[i] > self.bounds[i][1]:
                return -self.sign*np.inf

        alpha = 10.0**pars[0]
        beta = 10.0**pars[1]
        gamma = pars[2]
        sigma2 = 10.0**pars[3]

        lnlike = 0.0
        for rho_in in self.rho_in_vals:
            lnlike += np.log(np.clip(self.p_abgs(alpha, beta, gamma, sigma2, rho_in), 1e-20, None))

        # Add a prior on sigma2
        lnlike += np.log(sigma2)

        return self.sign*lnlike
