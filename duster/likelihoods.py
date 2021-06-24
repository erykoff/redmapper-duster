import numpy as np

from .pdfs import p_dust, compute_normalized_rho_pars


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
    """
    def __init__(self, rho_map1, rho_map2, bounds):
        vpix1 = rho_map1.valid_pixels
        vpix2 = rho_map2.valid_pixels

        self.yvals1 = rho_map1[vpix1]
        self.yvals2 = rho_map2[vpix2]
        self.bounds = bounds
        self.npars = len(bounds)

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
                return -np.inf

        b = 10.0**pars[0]
        u = 10.0**pars[1]

        rho_0, rho_min = compute_normalized_rho_pars(u, b)

        p_dust_1 = p_dust(rho_0, b, rho_min, self.yvals1)
        p_dust_2 = p_dust(rho_0, b, rho_min, self.yvals2)

        lnlike = np.sum(np.log(np.clip(p_dust_1, 1e-20, None)))
        lnlike += np.sum(np.log(np.clip(p_dust_2, 1e-20, None)))

        return lnlike
