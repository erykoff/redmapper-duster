import os
import numpy as np
import healsparse as hsp


class RhoMapMaker(object):
    """Normalize a reddening map into a rho map.

    Parameters
    ----------
    config : `duster.DusterConfiguration`
        Configuration object
    """
    def __init__(self, config):
        self.config = config

        self.mask = hsp.HealSparseMap.read(self.config.maskfile)

    def normalize_map(self, mapfile, nside):
        """Normalize an input map after applying the relevant mask.

        Parameters
        ----------
        mapfile : `str`
            Name of the map file for input.
        nside : `int`
            Nside to compute normalization.

        Returns
        -------
        rho_map : `healsparse.HealSparseMap`
        """
        map_in = hsp.HealSparseMap.read(mapfile)

        map_masked = hsp.HealSparseMap.make_empty(16, map_in.nside_sparse, np.float32)
        vpix, ra, dec = map_in.valid_pixels_pos(return_pixels=True)
        use, = np.where(self.mask.get_values_pos(ra, dec) > 0)

        map_masked[vpix[use]] = map_in[vpix[use]]

        map_dg = map_masked.degrade(nside)

        map_in = None
        map_masked = None

        vpix = map_dg.valid_pixels
        mean = np.mean(map_dg[vpix])

        map_dg /= mean

        metadata = {'MEAN': mean}
        map_dg.metadata = metadata

        return map_dg

    def make_rho_maps(self):
        """Make and save multiple rho maps."""
        nside = self.config.duster_nside

        fname1 = self.config.redmapper_filename(f'rho_{self.config.duster_label1}',
                                                filetype='hsp')
        self.config.duster_rhofile1 = fname1

        if os.path.isfile(fname1):
            self.config.logger.info("%s already there.  Skipping...", fname1)
        else:
            self.config.logger.info("Normalizing %s", self.config.duster_mapfile1)
            nmap1 = self.normalize_map(self.config.duster_mapfile1, nside)

            nmap1.write(fname1)

        fname2 = self.config.redmapper_filename(f'rho_{self.config.duster_label2}',
                                                filetype='hsp')
        self.config.duster_rhofile2 = fname2

        if os.path.isfile(fname2):
            self.config.logger.info("%s already there.  Skipping...", fname2)
        else:
            self.config.logger.info("Normalizing %s", self.config.duster_mapfile2)
            nmap2 = self.normalize_map(self.config.duster_mapfile2, nside)

            nmap2.write(fname2)
