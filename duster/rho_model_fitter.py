import numpy as np
import matplotlib.pyplot as plt
import fitsio
import emcee
import corner
import os
import esutil
from multiprocessing import Pool


from .likelihoods import RhoModelLikelihood
from .pdfs import compute_normalized_rho_pars, p_dust


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

        Outputs will be reflected in the config object.

        Parameters
        ----------
        rho_map1 : `healsparse.HealSparseMap`
           Normalized rho map 1
        rho_map2 : `healsparse.HealSparseMap`
           Normalized rho map 2
        """
        chain_fname = self.config.redmapper_filename('rho_model_chain')
        self.config.duster_rho_model_chainfile = chain_fname

        if os.path.isfile(chain_fname):
            self.config.logger.info("%s already there.  Skipping...", chain_fname)
            lnprob = fitsio.read(chain_fname, ext=0)
            chain = fitsio.read(chain_fname, ext=1)
        else:
            ndim = 2
            nwalkers = self.config.duster_nwalkers

            bounds = [[np.log10(0.1), np.log10(20.0)],
                      [np.log10(0.01), np.log10(20.0)]]

            p0 = np.zeros((nwalkers, ndim))
            for i in range(ndim):
                p0[:, i] = (bounds[i][1] - bounds[i][0])*np.random.random_sample(nwalkers) + bounds[i][0]

            rho_like = RhoModelLikelihood(rho_map1, rho_map2, bounds)

            with Pool(processes=self.config.duster_nproc) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, rho_like, pool=pool)
                state = sampler.run_mcmc(p0, self.config.duster_rho_model_nsample1, progress=True)
                sampler.reset()

                _ = sampler.run_mcmc(state, self.config.duster_rho_model_nsample2, progress=True)

            lnprob = sampler.get_log_prob(flat=True)

            chain = sampler.get_chain(flat=True)
            chain[:, 0] = 10.0**chain[:, 0]
            chain[:, 1] = 10.0**chain[:, 1]

            # Save the chain and the lnprobability
            fitsio.write(chain_fname, lnprob, clobber=True)
            fitsio.write(chain_fname, chain, clobber=False)

        fname = self.config.redmapper_filename('rho_model')
        self.config.duster_rho_model_file = fname

        if os.path.isfile(fname):
            self.config.logger.info("%s already there.  Skipping...", fname)

            arr = fitsio.read(fname, ext=1)
            rho_0 = arr[0]['rho_0']
            rho_min = arr[0]['rho_min']
            b = arr[0]['b']
        else:
            # Compute the max likelihood points
            ml_ind = np.argmax(lnprob)
            b = chain[ml_ind, 0]
            u = chain[ml_ind, 1]

            rho_0, rho_min = compute_normalized_rho_pars(u, b)

            arr = np.zeros(1, [('rho_0', 'f8'),
                               ('rho_min', 'f8'),
                               ('b', 'f8')])
            arr[0]['rho_0'] = rho_0
            arr[0]['rho_min'] = rho_min
            arr[0]['b'] = b
            fitsio.write(fname, arr)

        self.config.duster_rho_0 = rho_0
        self.config.duster_rho_min = rho_min
        self.config.duster_b = b

        # Plot the lnprobability
        lnprobfile = self.config.redmapper_filename('rho_model_lnprob',
                                                    paths=(self.config.plotpath, ),
                                                    filetype='png')
        if not os.path.isfile(lnprobfile):
            fig = plt.figure(1, figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)
            ax.plot(lnprob, 'r.')
            ax.set_xlabel('Step')
            ax.set_ylabel('ln(likelihood)')
            ax.set_title('rho model likelihood')
            fig.tight_layout()
            fig.savefig(lnprobfile)
            plt.close(fig)

        # Plot the corner plot
        cornerfile = self.config.redmapper_filename('rho_model_corner',
                                                    paths=(self.config.plotpath, ),
                                                    filetype='png')
        if not os.path.isfile(cornerfile):
            plt.clf()
            corner.corner(chain[:, :], labels=['b', 'rho_min/rho_0'])
            plt.savefig(cornerfile)

        # Plot the model over the rho distribution
        rhodistfile = self.config.redmapper_filename('rho_model_dist',
                                                     paths=(self.config.plotpath, ),
                                                     filetype='png')
        if not os.path.isfile(rhodistfile):
            vpix1 = rho_map1.valid_pixels
            vpix2 = rho_map2.valid_pixels

            binsize = 0.05
            h_rho1 = esutil.stat.histogram(rho_map1[vpix1], min=0.0, max=5.0, binsize=binsize, more=True)
            h_rho2 = esutil.stat.histogram(rho_map2[vpix2], min=0.0, max=5.0, binsize=binsize, more=True)

            rho_model = np.sum(h_rho1['hist'])*p_dust(rho_0, b, rho_min, h_rho1['center'])*binsize

            fig = plt.figure(1, figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)
            lab1 = self.config.duster_label1
            lab2 = self.config.duster_label2
            ax.plot(h_rho1['center'], h_rho1['hist'], 'r-', label=lab1)
            ax.plot(h_rho2['center'], h_rho2['hist'], 'b-', label=lab2)
            ax.plot(h_rho1['center'], rho_model, 'k--', label='Model')
            ax.legend()
            ax.set_yscale('log')
            ax.set_ylim(1e-1, 2*rho_model.max())
            ax.set_xlabel(r'$\rho$')
            ax.set_ylabel('Number of pixels (nside %d)' % (self.config.duster_nside))
            fig.tight_layout()
            fig.savefig(rhodistfile)
            plt.close(fig)
