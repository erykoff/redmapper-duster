import os
import numpy as np
from scipy.stats import norm
import fitsio
import healsparse as hsp
import matplotlib.pyplot as plt

from .pdfs import p_dust
from .constants import (RHO_INTEGRAL_OFFSET,
                        RHO_INTEGRAL_MIN,
                        RHO_INTEGRAL_MAX,
                        RHO_INTEGRAL_STEP2)


def reconstruct_map(rho_map_in, rho_0, rho_min, b, chain):
    """Reconstruct a map from chain parameters.

    Parameters
    ----------
    rho_map_in : `healsparse.HealSparseMap`
        Input (biased) rho map.
    rho_0 : `float`
        Model rho_0 parameter.
    rho_min : `float`
        Model rho_min parameter.
    b : `float`
        Model b parameter.
    chain : `np.ndarray`
        Flat chain with alpha, beta, gamma, sigma parameters.

    Returns
    -------
    rho_map_out : `healsparse.HealSparseMap`
        Output (debiased) rho map.
    var_map_out : `healsparse.HealSparseMap`
        Variance map of the debiased rho map.
    """
    vpix = rho_map_in.valid_pixels

    rho_map_out = hsp.HealSparseMap.make_empty_like(rho_map_in)
    var_map_out = hsp.HealSparseMap.make_empty_like(rho_map_in)

    rho_int_vals = np.arange(RHO_INTEGRAL_MIN,
                             RHO_INTEGRAL_MAX,
                             RHO_INTEGRAL_STEP2) + RHO_INTEGRAL_OFFSET
    P_dust = p_dust(rho_0, b, rho_min, rho_int_vals)

    rho_mean = np.zeros(rho_int_vals.size)
    rho2_mean = np.zeros(rho_int_vals.size)

    n_point = chain.shape[0]
    for i, rho_int_val in enumerate(rho_int_vals):
        rho_mean[i] = (1./n_point)*np.sum(chain[:, 0]*rho_int_val**chain[:, 1] + chain[:, 2])
        rho2_mean[i] = (1./n_point)*np.sum((chain[:, 0]*rho_int_val**chain[:, 1] + chain[:, 2])**2. +
                                           chain[:, 3]**2.)

    rho_var = rho2_mean - rho_mean**2.

    for pix in vpix:
        rho_in = rho_map_in[pix]
        gaussian = norm.pdf(rho_in, loc=rho_mean, scale=np.sqrt(rho_var))

        numerator = np.trapz(gaussian*P_dust*rho_int_vals, x=rho_int_vals)
        denominator = np.trapz(gaussian*P_dust, x=rho_int_vals)

        rho_map_out[int(pix)] = numerator/denominator

        numerator2 = np.trapz(gaussian*P_dust*rho_int_vals*rho_int_vals, x=rho_int_vals)
        var_map_out[int(pix)] = numerator2/denominator - (numerator/denominator)**2.

    return rho_map_out, var_map_out


class RhoReconstructor(object):
    """Class to reconstruct an unbiased rho map.

    Parameters
    ----------
    config : `duster.DusterConfiguration`
    """
    def __init__(self, config):
        self.config = config

    def reconstruct_map(self, rho_map_in):
        """Reconstruct a map using the configured chain and rho model pars.

        Parameters
        ----------
        rho_map_in : `healsparse.HealSparseMap`
            Input (biased) rho map.
        """
        nside = rho_map_in.nside_sparse

        recon_fname = self.config.redmapper_filename('rho_debiased_map')
        recon_var_fname = self.config.redmapper_filename('rho_debiased_var_map')
        self.config.duster_recon_map_file = recon_fname
        self.config.duster_recon_var_map_file = recon_var_fname
        if os.path.isfile(recon_fname) and os.path.isfile(recon_var_fname):
            self.config.logger.info("%s already there.  Skipping...", recon_fname)
            rho_recon_map = hsp.HealSparseMap.read(recon_fname)
            rho_recon_var = hsp.HealSparseMap.read(recon_var_fname)
        else:
            chain = fitsio.read(self.config.duster_debias_model_chainfile, ext=1)

            self.config.logger.info("Reconstructing debiased rho map...")
            rho_recon_map, rho_recon_var = reconstruct_map(rho_map_in,
                                                           self.config.duster_rho_0,
                                                           self.config.duster_rho_min,
                                                           self.config.duster_b,
                                                           chain)

            rho_recon_map.write(recon_fname)
            rho_recon_var.write(recon_var_fname)

        # Get the fracdet map
        mask = hsp.HealSparseMap.read(self.config.maskfile)
        mask_fracdet = mask.fracdet_map(nside)

        map1_comp_file = self.config.redmapper_filename('rho_recon_%s_comp' % (self.config.duster_label1),
                                                        paths=(self.config.plotpath, ),
                                                        filetype='png')
        if not os.path.isfile(map1_comp_file):
            vpix = rho_recon_map.valid_pixels
            usepix, = np.where(mask_fracdet[vpix] > self.config.duster_min_coverage_fraction)
            vpix = vpix[usepix]

            rho1 = hsp.HealSparseMap.read(self.config.duster_rhofile1)

            fig = plt.figure(1, figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)
            ax.hexbin(rho1[vpix], rho_recon_map[vpix], bins='log', gridsize=50)
            ax.set_xlabel(r'$\rho_{\mathrm{%s}}$' % (self.config.duster_label1))
            ax.set_ylabel(r'$\rho_{\mathrm{recon}}$')
            ax.set_title('nside = %d, reconstructed' % (nside))
            fig.tight_layout()
            fig.savefig(map1_comp_file)
            plt.close(fig)

        map2_comp_file = self.config.redmapper_filename('rho_recon_%s_comp' % (self.config.duster_label2),
                                                        paths=(self.config.plotpath, ),
                                                        filetype='png')
        if not os.path.isfile(map2_comp_file):
            vpix = rho_recon_map.valid_pixels
            usepix, = np.where(mask_fracdet[vpix] > self.config.duster_min_coverage_fraction)
            vpix = vpix[usepix]

            rho1 = hsp.HealSparseMap.read(self.config.duster_rhofile2)

            fig = plt.figure(1, figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)
            ax.hexbin(rho1[vpix], rho_recon_map[vpix], bins='log', gridsize=50)
            ax.set_xlabel(r'$\rho_{\mathrm{%s}}$' % (self.config.duster_label2))
            ax.set_ylabel(r'$\rho_{\mathrm{recon}}$')
            ax.set_title('nside = %d, reconstructed' % (nside))
            fig.tight_layout()
            fig.savefig(map2_comp_file)
            plt.close(fig)
