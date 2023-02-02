import os
# import numpy as np
import healsparse as hsp
import redmapper
import matplotlib.pyplot as plt
import skyproj

from .configuration import DusterConfiguration
from .redgals import RedGalaxySelector
from .rho_maps import RhoMapMaker
from .rho_model_fitter import RhoModelFitter
from .rho_raw_mapper import RhoRawMapper
from .rho_filtering import RhoMapFilterer
from .debias_fitter import DebiasFitter
from .rho_reconstructor import RhoReconstructor


class DusterPipeline:
    """Run the full DustER pipeline.

    Parameters
    ----------
    configfile : `str`
        Name of config file.
    """
    def __init__(self, configfile, norm=None):
        self.config = DusterConfiguration(configfile)

        if norm is not None:
            pass

    def run(self):
        """
        Run the DustER pipeline.
        """
        # Select red galaxies
        selector = RedGalaxySelector(self.config)
        selector.select_red_galaxies()

        # Make normalized rho maps
        map_maker = RhoMapMaker(self.config)
        map_maker.make_rho_maps()

        # Fit rho parameters
        rho_model_fitter = RhoModelFitter(self.config)
        rho_map1 = hsp.HealSparseMap.read(self.config.duster_rhofile1)
        rho_map2 = hsp.HealSparseMap.read(self.config.duster_rhofile2)

        rho_model_fitter.fit_rho_model(rho_map1, rho_map2)

        # Make the raw rho map
        raw_mapper = RhoRawMapper(self.config)
        gals = redmapper.GalaxyCatalog.from_fits_file(self.config.duster_redgalfile)

        raw_mapper.fit_rho_obs_raw_map(gals)

        # Filter the raw rho map
        rho_raw_map = hsp.HealSparseMap.read(self.config.duster_raw_map_file)

        filterer = RhoMapFilterer(self.config)
        filterer.filter_map(rho_raw_map)

        # Fit the alpha/beta/gamma parameters
        rho_filt_map = hsp.HealSparseMap.read(self.config.duster_filtered_map_file)

        debias = DebiasFitter(self.config)
        debias.fit_debias_model(rho_filt_map)

        # Make a reconstructed map
        reconstructor = RhoReconstructor(self.config)
        reconstructor.reconstruct_map(rho_filt_map)

        # And plot the map.

        map_plot_file = self.config.redmapper_filename('rho_recon_map_nside%d' % (self.config.duster_nside),
                                                       paths=(self.config.plotpath, ),
                                                       filetype='png')

        rho_recon = hsp.HealSparseMap.read(self.config.duster_recon_map_file)

        fig, ax = plt.subplots(figsize=(8, 5))
        sp = skyproj.DESSkyproj(ax=ax)
        sp.draw_hspmap(rho_recon, vmin=0.0, vmax=5.0)
        sp.draw_inset_colorbar(label='Normalized Map')
        fig.savefig(map_plot_file)
        plt.close(fig)

        # Print out relevant config values...

        print('Use the following configurations for a redmapper run:')
        map_file = os.path.abspath(self.config.duster_recon_map_file)
        print(f'dereddening_mapfile: {map_file}')
        print(f'dereddening_norm: {raw_mapper.norm}')
        print(f'dereddening_constants: {list(self.config.duster_dereddening_constants)}')
        print('dereddening_apply: true')
