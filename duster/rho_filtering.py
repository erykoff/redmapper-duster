import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import scipy.optimize
import os
import healsparse as hsp
import fitsio


def cl_filter_fxn(x, k, eta, C_noise):
    """
    Compute filter function f(x | k, eta, C_noise)

    Parameters
    ----------
    x : array-like
    k : `float`
    eta : `float`
    C_noise : `float`

    Returns
    -------
    fxn : array-like
       function evaluated at x
    """
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings("ignore")
        fxn = k*(x/50.)**(-1.*eta) + C_noise
    return fxn


def cl_filter_fit_fxn(x, k, eta, C_noise):
    """
    Compute filter function for fit.

    Parameters
    ----------
    x : array-like
    k : `float`
    eta : `float`
    C_noise : `float`

    Returns
    -------
    lnfxn : array-like
       log(function) evaluated at x
    """
    return np.log(cl_filter_fxn(x, k, eta, C_noise))


class RhoMapFilterer(object):
    """Class to filter a rho map, fitting C_ell noise term.

    Parameters
    ----------
    config : `duster.DusterConfiguration`
    """
    def __init__(self, config):
        self.config = config

    def filter_map(self, rho_map):
        """Filter a rho map.

        Parameters
        ----------
        rho_map : `healsparse.HealSparseMap`
            Input rho map to filter.

        Returns
        -------
        ???
        """
        nside = rho_map.nside_sparse

        # Generate a regular healpix map for healpy.
        rho_map_hpmap = rho_map.generate_healpix_map(nest=False)
        # Subtract off the monopole.
        okpix, = np.where(rho_map_hpmap > hp.UNSEEN)
        rho_map_mean = np.mean(rho_map_hpmap[okpix])
        rho_map_hpmap[okpix] -= rho_map_mean

        # Compute c_ell and alm
        c_ell, alm = hp.anafast(rho_map_hpmap, alm=True)
        ell = np.arange(len(c_ell))
        lmax = len(ell) - 1

        fname = self.config.redmapper_filename('filter_model')
        self.config.duster_filter_model_file = fname

        if os.path.isfile(fname):
            self.config.logger.info("%s already there.  Skipping...", fname)

            arr = fitsio.read(fname, ext=1)
            k = arr[0]['k']
            eta = arr[0]['eta']
            C_noise = arr[0]['C_noise']
        else:
            bounds = ((1e-15, 1e-15, 1e-15),
                      (100.0, 100.0, 100.0))

            ell_use, = np.where(ell >= self.config.duster_filter_ell_min)

            pars = scipy.optimize.curve_fit(cl_filter_fit_fxn,
                                            ell[ell_use],
                                            np.log(np.abs(c_ell[ell_use])),
                                            p0=[1.0, 1.0, 1.0],
                                            bounds=bounds)

            k = pars[0][0]
            eta = pars[0][1]
            C_noise = pars[0][2]

            arr = np.zeros(1, [('k', 'f8'),
                               ('eta', 'f8'),
                               ('C_noise', 'f8')])
            arr[0]['k'] = k
            arr[0]['eta'] = eta
            arr[0]['C_noise'] = C_noise
            fitsio.write(fname, arr)

            filterplotfile = self.config.redmapper_filename('filter_model',
                                                            paths=(self.config.plotpath, ),
                                                            filetype='png')

            fig = plt.figure(1, figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)
            ax.plot(ell[ell_use], c_ell[ell_use], 'm-', label=r'$\rho_{\mathrm{in}}$')
            ax.plot(ell[ell_use], cl_filter_fxn(ell[ell_use], k, eta, C_noise), label='Filter')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('ell')
            ax.set_ylabel('C_ell')
            ax.legend()
            plt.savefig(filterplotfile)
            plt.close(fig)

        fname_filtered = self.config.redmapper_filename('rho_filtered_map',
                                                        filetype='hsp')
        self.config.duster_filtered_map_file = fname_filtered
        if os.path.isfile(fname_filtered):
            self.config.logger.info("%s already there.  Skipping...", fname_filtered)
            rho_filtered_map = hsp.HealSparseMap.read(fname_filtered)
        else:
            alm_prime = alm.copy()

            c_ell_true = np.clip(cl_filter_fxn(ell, k, eta, C_noise), 0.0, 1e50)
            for index in range(alm.size):
                l, m = hp.Alm.getlm(lmax, index)
                alm_prime[index] *= (c_ell_true[l]/(c_ell_true[l] + C_noise))

            rho_filtered = hp.alm2map(alm_prime, nside, lmax=lmax)

            rho_filtered_map = hsp.HealSparseMap.make_empty_like(rho_map)
            vpix = rho_map.valid_pixels
            rho_filtered_map[vpix] = rho_filtered[hp.nest2ring(nside, vpix)].astype(np.float32)
            rho_filtered_map[vpix] += rho_map_mean

            rho_filtered_map.write(fname_filtered)

        # Get the fracdet map
        mask = hsp.HealSparseMap.read(self.config.maskfile)
        mask_fracdet = mask.fracdet_map(nside)

        map1_comp_file = self.config.redmapper_filename('rho_filt_%s_comp' % (self.config.duster_label1),
                                                        paths=(self.config.plotpath, ),
                                                        filetype='png')
        if not os.path.isfile(map1_comp_file):
            vpix = rho_filtered_map.valid_pixels
            usepix, = np.where(mask_fracdet[vpix] > self.config.duster_min_coverage_fraction)
            vpix = vpix[usepix]

            rho1 = hsp.HealSparseMap.read(self.config.duster_rhofile1)

            fig = plt.figure(1, figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)
            ax.hexbin(rho1[vpix], rho_filtered_map[vpix], bins='log', gridsize=50)
            ax.set_xlabel(r'$\rho_{\mathrm{%s}}$' % (self.config.duster_label1))
            ax.set_ylabel(r'$\rho_{\mathrm{filtered}}$')
            ax.set_title('nside = %d, filtered' % (nside))
            fig.tight_layout()
            fig.savefig(map1_comp_file)
            plt.close(fig)

        map2_comp_file = self.config.redmapper_filename('rho_filt_%s_comp' % (self.config.duster_label2),
                                                        paths=(self.config.plotpath, ),
                                                        filetype='png')
        if not os.path.isfile(map2_comp_file):
            vpix = rho_filtered_map.valid_pixels
            usepix, = np.where(mask_fracdet[vpix] > self.config.duster_min_coverage_fraction)
            vpix = vpix[usepix]

            rho2 = hsp.HealSparseMap.read(self.config.duster_rhofile2)

            fig = plt.figure(1, figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)
            ax.hexbin(rho2[vpix], rho_filtered_map[vpix], bins='log', gridsize=50)
            ax.set_xlabel(r'$\rho_{\mathrm{%s}}$' % (self.config.duster_label2))
            ax.set_ylabel(r'$\rho_{\mathrm{filtered}}$')
            ax.set_title('nside = %d, filtered' % (nside))
            fig.tight_layout()
            fig.savefig(map2_comp_file)
            plt.close(fig)
