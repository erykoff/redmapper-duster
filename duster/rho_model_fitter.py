import numpy as np
import emcee
import corner

from .likelihoods import RhoModelLikelihood


class RhoModelFitter(object):
    """Class for fitting the rho model.

    Parameters
    ----------
    config : `duster.DusterConfiguration`
    """
    def __init__(self, config):
        self.config = config

    def fit_rho_model(self, rho_map1, rho_map2):
        """Fit the rho model given the rho maps.

        Parameters
        ----------
        rho_map1 : `healsparse.HealSparseMap`
           Normalized rho map 1
        rho_map2 : `healsparse.HealSparseMap`
           Normalized rho map 2

        Returns
        -------
        ???
        """
        ndim = 2
        nwalkers = 32

        bounds = [[np.log10(0.1), np.log10(20.0)],
                  [np.log10(0.01), np.log10(20.0)]]

        p0 = np.zeros((nwalkers, ndim))
        for i in range(ndim):
            p0[:, i] = (bounds[i][1] - bounds[i][0])*np.random.random_sample(nwalkers) + bounds[i][0]

        rho_like = RhoModelLikelihood(rho_map1, rho_map2, bounds)

        with Pool(processes=self.config.duster_nproc):
            sampler = emcee.EnsembleSampler(nwalkers, ndim, redlike, pool=pool)
            state = sampler.run_mcmc(p0, 500, progress=True)
            sampler.reset()

            state2 = sampler.run_mcmc(state, 1000, progress=True)

        # And here we go, we need the samples and the means, etc.

