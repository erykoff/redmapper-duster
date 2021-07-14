import numpy as np
import matplotlib.pyplot as plt
import esutil
import healpy as hp
import healsparse as hsp
import os
import fitsio
import redmapper


class RhoRawPixelComputer(object):
    """Class to compute rho_obs raw from a single pixel.

    Parameters
    ----------
    zredstr : `redmapper.RedSequenceColorPar`
        Red sequence parameterization object.
    norm : `float`
        Reddening normalization value.
    r_band : array-like
        Array of normalized reddening constants.
    sigclip : `float`, optional
        Number of sigma to clip in rho_raw computation.
    niter : `int`, optional
        Number of sigma-clip iterations.
    """
    def __init__(self, zredstr, indices, norm, r_band, sigclip=3.0, niter=3):
        self.zredstr = zredstr
        self.indices = indices
        self.r_band = r_band
        self.norm = norm
        self.sigclip = sigclip
        self.niter = niter

        self.r_col = np.array(r_band)[0: -1] - np.array(r_band)[1:]

    def compute_rho_raw(self, pixgals, use_ztrue=False):
        """Compute rho_raw for a single pixel.

        Parameters
        ----------
        pixgals : `redmapper.GalaxyCatalog`
            Galaxies from a single pixel.
        use_ztrue : `bool`, optional
            Use ztrue instead of zred_uncorr.  Use for
            synthetic sky catalog testing.

        Returns
        -------
        rho_raw : `float`
            The value of rho_raw at the pixel.
        """
        galcol = pixgals.galcol
        galcol_err = pixgals.galcol_err

        if use_ztrue:
            zuse = pixgals.ztrue
        else:
            zuse = pixgals.zred_uncorr

        zind = self.zredstr.zindex(zuse)

        c_pred = np.zeros((pixgals.size, len(self.indices)))
        err2 = np.zeros_like(c_pred)
        rhobar = np.zeros_like(c_pred)
        sig2 = np.zeros_like(c_pred)
        c_use = [[]]*len(self.indices)

        for i, index in enumerate(self.indices):
            r_col = self.r_col[index]

            c_pred[:, i] = (self.zredstr.c[zind, index] +
                            self.zredstr.slope[zind, index]*(pixgals.refmag -
                                                             self.zredstr.pivotmag[zind]))
            err2[:, i] = galcol_err[:, index]**2. + self.zredstr.sigma[index, index, zind]**2.
            rhobar[:, i] = (galcol[:, index] - c_pred[:, i])/(self.norm*r_col)
            sig2[:, i] = err2[:, i]/(self.norm*r_col)

            c_use[i] = np.arange(pixgals.size)

        for iteration in range(self.niter):
            numerator = 0.0
            denominator = 0.0

            for i in range(len(self.indices)):
                numerator += np.sum((1./sig2[c_use[i], i])*rhobar[c_use[i], i])
                denominator += np.sum(1./sig2[c_use[i], i])

            rho_raw = numerator/denominator

            for i in range(len(self.indices)):
                pull = (rhobar[:, i] - rho_raw)/np.sqrt(sig2[:, i])
                sig_pull = np.std(pull[c_use[i]])
                c_use[i] = np.where(np.abs(pull) < self.sigclip*sig_pull)[0]

        return rho_raw


class RhoRawMapper(object):
    """Class for fitting a rho_raw map.

    Parameters
    ----------
    config : `duster.DusterConfiguration`
    """
    def __init__(self, config):
        self.config = config

    def fit_rho_obs_raw_map(self, gals):
        """Fit the rho_obs raw map from a set of galaxies.

        Parameters
        ----------
        gals : `redmapper.GalaxyCatalog`

        Returns
        -------
        ???
        """
        nside = self.config.duster_nside

        raw_fname = self.config.redmapper_filename('rho_raw_map', filetype='hsp')
        self.config.duster_raw_map_file = raw_fname
        if os.path.isfile(raw_fname):
            self.config.logger.info("%s already there.  Skipping...", raw_fname)
            rho_raw_map = hsp.HealSparseMap.read(raw_fname)
        else:
            self.config.logger.info("Computing rho raw map.")
            ipnest = hp.ang2pix(nside, gals.ra, gals.dec, nest=True, lonlat=True)
            h, rev = esutil.stat.histogram(ipnest, rev=True, min=0, max=hp.nside2npix(nside))

            pixels, = np.where(h >= self.config.duster_min_gals_per_pixel)

            zredstr = redmapper.RedSequenceColorPar(self.config.parfile, fine=True)
            indices = self.config.duster_color_indices
            r_band = self.config.dered_const_norm

            if self.config.duster_norm is None:
                # Get the normalization from the rho map
                rho_hdr = fitsio.read_header(self.config.duster_rhofile1, ext=0)
                if 'MEAN' not in rho_hdr:
                    raise RuntimeError("Cannot determine normalization: MEAN must be in %s header." %
                                       (self.config.duster_rhofile1))
                ref_const = self.config.duster_dereddening_constants[self.config.duster_dereddening_ref_ind]
                norm = ref_const*rho_hdr['MEAN']
            else:
                # Get the normalization from the configuration
                norm = self.config.duster_norm

            rhoComputer = RhoRawPixelComputer(zredstr, indices, norm, r_band)

            rho_raw_map = hsp.HealSparseMap.make_empty(16, nside, np.float32)

            for ii, pix in enumerate(pixels):
                if ((ii % 100) == 0):
                    self.config.logger.info("Working on pixel %d of %d" % (ii, len(pixels)))
                i1a = rev[rev[pix]: rev[pix + 1]]

                rho_raw_map[pix] = rhoComputer.compute_rho_raw(gals[i1a])

            # Write out the map
            rho_raw_map.write(raw_fname)

        # Get the fracdet map
        mask = hsp.HealSparseMap.read(self.config.maskfile)
        mask_fracdet = mask.fracdet_map(nside)

        map1_comp_file = self.config.redmapper_filename('rho_raw_%s_comp' % (self.config.duster_label1),
                                                        paths=(self.config.plotpath, ),
                                                        filetype='png')
        if not os.path.isfile(map1_comp_file):
            vpix = rho_raw_map.valid_pixels
            usepix, = np.where(mask_fracdet[vpix] > self.config.duster_min_coverage_fraction)
            vpix = vpix[usepix]

            rho1 = hsp.HealSparseMap.read(self.config.duster_rhofile1)

            fig = plt.figure(1, figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)
            ax.hexbin(rho1[vpix], rho_raw_map[vpix], bins='log', gridsize=50)
            ax.set_xlabel(r'$\rho_{\mathrm{%s}}$' % (self.config.duster_label1))
            ax.set_ylabel(r'$\rho_{\mathrm{raw}}$')
            ax.set_title('nside = %d, raw' % (nside))
            fig.tight_layout()
            fig.savefig(map1_comp_file)
            plt.close(fig)

        map2_comp_file = self.config.redmapper_filename('rho_raw_%s_comp' % (self.config.duster_label2),
                                                        paths=(self.config.plotpath, ),
                                                        filetype='png')
        if not os.path.isfile(map2_comp_file):
            vpix = rho_raw_map.valid_pixels
            usepix, = np.where(mask_fracdet[vpix] > self.config.duster_min_coverage_fraction)
            vpix = vpix[usepix]

            rho2 = hsp.HealSparseMap.read(self.config.duster_rhofile2)

            fig = plt.figure(1, figsize=(8, 6))
            fig.clf()
            ax = fig.add_subplot(111)
            ax.hexbin(rho2[vpix], rho_raw_map[vpix], bins='log', gridsize=50)
            ax.set_xlabel(r'$\rho_{\mathrm{%s}}$' % (self.config.duster_label2))
            ax.set_ylabel(r'$\rho_{\mathrm{raw}}$')
            ax.set_title('nside = %d, raw' % (nside))
            fig.tight_layout()
            fig.savefig(map2_comp_file)
            plt.close(fig)
