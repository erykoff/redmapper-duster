import os
import numpy as np
import fitsio

import redmapper


class RedGalaxySelector(object):
    """Class to select red galaxies for duster.

    Parameters
    ----------
    config : `duster.DusterConfiguration`
    """
    def __init__(self, config):
        self.config = config

    def select_red_galaxies(self):
        """Select red galaxies for duster.
        """
        # We want to do this in an efficient way, as with redmagic
        redgal_fname = self.config.redmapper_filename('redgals')
        self.config.duster_redgalfile = redgal_fname

        if os.path.isfile(redgal_fname):
            self.config.logger.info("%s already there.  Skipping...", redgal_fname)
            red_gals = redmapper.GalaxyCatalog.from_galfile(redgal_fname)
        else:
            zredstr = redmapper.RedSequenceColorPar(self.config.parfile, fine=True)

            tab = redmapper.Entry.from_fits_file(self.config.galfile)

            started = False
            for i, pix in enumerate(tab.hpix):
                gals = redmapper.GalaxyCatalog.from_galfile(self.config.galfile,
                                                            zredfile=self.config.zredfile,
                                                            nside=tab.nside,
                                                            hpix=pix,
                                                            border=0.0,
                                                            truth=self.config.has_truth)
                # Select out the galaxies
                mstar = zredstr.mstar(gals.zred_uncorr)
                use, = np.where((gals.zred_uncorr > self.config.duster_zrange[0]) &
                                (gals.zred_uncorr < self.config.duster_zrange[1]) &
                                (gals.chisq < self.config.duster_chi2_max) &
                                (gals.refmag < (mstar - 2.5*np.log10(self.config.duster_lum_min))))
                red_gals = gals[use]

                if not started:
                    red_gals.to_fits_file(redgal_fname, clobber=True)
                    started = True
                else:
                    with fitsio.FITS(redgal_fname, mode='rw') as fits:
                        fits[1].append(red_gals._ndarray)
