import numpy as np
import matplotlib.pyplot as plt
import fitsio
import emcee
import corner
import os
import scipy.minimize
from multiprocessing import Pool

from .likelihoods import DebiasLikelihood


class DebiasFitter(object):
    """Class for fitting the debias model.

    Parameters
    ----------
    config : `duster.DusterConfiguration`
    """
    def __init__(self, config):
        self.config = config

    def fit_debias_model(self, rho_map_in):
        """Fit the debias model given an input rho map.

        Parameters
        ----------
        rho_map_in : `healsparse.HealSparseMap`
            Biased input rho map.
        """
        chain_fname = self.config.redmapper_filename('debias_model_chain')
        self.config.duster_debias_model_chainfile = chain_fname

        if os.path.isfile(chain_fname):
            self.config.logger.info("%s already there.  Skipping...", chain_fname)
            lnprob = fitsio.read(chain_fname, ext=0)
            chain = fitsio.read(chain_fname, ext=1)
        else:
            ndim = 4

            bounds = [[np.log10(0.1), np.log10(2.0)],
                      [np.log10(0.1), np.log10(2.0)],
                      [-2.0, 1.0],
                      [np.log10(0.001**2.), np.log10(0.2**2.)]]

            p0 = np.zeros(ndim)
            for i in range(ndim):
                p0[i] = (bounds[i][0] + bounds[i][1])/2.

            debias_like_min = DebiasLikelihood(rho_map_in,
                                               bounds,
                                               self.config.duster_rho_0,
                                               self.config.duster_rho_min,
                                               self.config.duster_b,
                                               minimize=True)

            self.config.logger.info("Finding ML debias model parameters...")
            soln = scipy.optimize.minimize(debias_like_min, p0, bounds=bounds)

            nwalkers = self.config.duster_nwalkers

            # Set the chains starting around a small ball near the ML point.
            p0 = soln.x + 1e-4*np.random.randn(nwalkers, ndim)

            debias_like = DebiasLikelihood(rho_map_in,
                                           bounds,
                                           self.config.duster_rho_0,
                                           self.config.duster_rho_min,
                                           self.config.duster_b)

            self.config.logger.info("Running debias model chain...")
            with Pool(processes=self.config.duster_nproc) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, debias_like, pool=pool)
                state = sampler.run_mcmc(p0, self.config.duster_debias_model_nsample1, progress=True)
                sampler.reset()

                _ = sampler.run_mcmc(state, self.config.duster_debias_model_nsample2, progress=True)

            lnprob = sampler.get_log_prob(flat=True).copy()

            chain = sampler.get_chain(flat=True).copy()
            chain[:, 0] = 10.0**chain[:, 0]
            chain[:, 1] = 10.0**chain[:, 1]
            chain[:, 3] = np.sqrt(10.0**chain[:, 3])

            fitsio.write(chain_fname, lnprob, clobber=True)
            fitsio.write(chain_fname, chain, clobber=True)

        # Plot the lnprobability
        lnprobfile = self.config.redmapper_filename('debias_model_lnprob',
                                                    paths=(self.config.plotpath, ),
                                                    filetype='png')
        if not os.path.isfile(lnprobfile):
            fig = plt.figure(1, figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)
            ax.plot(lnprob, 'r.')
            ax.set_xlabel('Step')
            ax.set_ylabel('ln(likelihood)')
            ax.set_title('debias model likelihood')
            fig.tight_layout()
            fig.savefig(lnprobfile)
            plt.close(fig)

        # Plot the corner plot
        cornerfile = self.config.redmapper_filename('debias_model_corner',
                                                    paths=(self.config.plotpath, ),
                                                    filetype='png')
        if not os.path.isfile(cornerfile):
            plt.clf()
            corner.corner(chain[:, :], labels=['alpha', 'beta', 'gamma', 'sigma'])
            plt.savefig(cornerfile)
