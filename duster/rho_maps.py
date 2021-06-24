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
