"""Configuration class for duster.

This is based on the redmapper configuration.
"""
import numpy as np
import os
import logging

from redmapper.configuration import (ConfigField, Configuration, read_yaml,
                                     DuplicatableConfig)
from redmapper.dereddener import Dereddener


class DusterConfiguration(Configuration):
    """Configuration class for duster.
    """
    duster_nside = ConfigField(required=True)
    duster_min_gals_per_pixel = ConfigField(default=50)
    duster_min_coverage_fraction = ConfigField(default=0.5)

    duster_mapfile1 = ConfigField(required=True)
    duster_mapfile2 = ConfigField(required=True)
    duster_rhofile1 = ConfigField()
    duster_rhofile2 = ConfigField()
    duster_label1 = ConfigField(required=True)
    duster_label2 = ConfigField(required=True)

    duster_rho_model_file = ConfigField()
    duster_rho_model_chainfile = ConfigField()
    duster_rho_0 = ConfigField()
    duster_rho_min = ConfigField()
    duster_b = ConfigField()
    duster_norm = ConfigField()

    duster_raw_map_file = ConfigField()
    duster_filtered_map_file = ConfigField()
    duster_recon_map_file = ConfigField()
    duster_reconvar_map_file = ConfigField()

    duster_filter_ell_min = ConfigField(default=10)
    duster_filter_model_file = ConfigField()

    duster_debias_model_chainfile = ConfigField()

    duster_nwalkers = ConfigField(default=32)
    duster_nproc = ConfigField(default=2)

    duster_redgalfile = ConfigField()
    duster_chi2_max = ConfigField()
    duster_lum_min = ConfigField()
    duster_zrange = ConfigField(isArray=True, array_length=2, required=True)

    duster_color_indices = ConfigField(isArray=True, default=np.array([0, 1, 2]))
    duster_dereddening_constants = ConfigField(isArray=True)
    duster_dereddening_ref_ind = ConfigField(required=True, default=1)

    duster_rho_model_nsample1 = ConfigField(default=500)
    duster_rho_model_nsample2 = ConfigField(default=1000)
    duster_debias_model_nsample1 = ConfigField(default=100)
    duster_debias_model_nsample2 = ConfigField(default=500)

    def __init__(self, configfile, outpath=None):
        self._file_logging_started = False

        self._reset_vars()

        confdict = read_yaml(configfile)

        self.configpath = os.path.dirname(os.path.abspath(configfile))
        self.configfile = os.path.basename(configfile)

        # And now set the config variables
        confdict = self._set_duster_vars_from_dict(confdict)
        self._set_vars_from_dict(confdict)

        if outpath is not None:
            self.outpath = outpath

        # Get the logger
        logname = 'duster'
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(logname)

        # We allow a blank galfile for duster, if neccessary.
        if self.galfile is None:
            self.galfile = 'no_galfile'

        # get galaxy file stats
        if self.galfile != 'no_galfile':
            gal_stats = self._galfile_stats()
            self.galfile_area = gal_stats['area']
            if (self.area is not None):
                if self.depthfile is not None:
                    self.logger.info("WARNING: You should not need to set area "
                                     "in the config file when you have a depth map.")
                if (np.abs(self.area - gal_stats['area']) > 1e-3):
                    self.logger.info("Config area is not equal to galaxy file area.  Using config area.")
                    gal_stats.pop('area')
            else:
                if self.depthfile is None and self.nside > 0 and len(self.hpix) > 0:
                    raise RuntimeError("You must set a config file area if no "
                                       "depthfile is present and you are running a sub-region")
            self._set_vars_from_dict(gal_stats, check_none=True)

        self.validate()

        # Checks ...
        if self.maskfile is not None and self.mask_mode == 0:
            raise ValueError(("A maskfile is set, but mask_mode is 0 (no mask). "
                              "Assuming this is not intended."))

        self.d = DuplicatableConfig(self)

        if self.dereddening_apply:
            self.dereddener = Dereddener(self.dereddening_mapfile, self.dereddening_norm,
                                         self.dereddening_constants, self.nmag, self.ref_ind)
        else:
            self.dereddener = None

        self.dered_const_norm = (np.array(self.duster_dereddening_constants) /
                                 self.duster_dereddening_constants[self.duster_dereddening_ref_ind])

        # Finally, once everything is here, we can make paths
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath, exist_ok=True)
        if not os.path.exists(os.path.join(self.outpath, self.plotpath)):
            os.makedirs(os.path.join(self.outpath, self.plotpath), exist_ok=True)

    def _set_duster_vars_from_dict(self, d):
        """Set duster config vars from dict.  Pop them out.
        """
        keys = list(d.keys())
        for key in keys:
            if key in DusterConfiguration.__dict__:
                try:
                    setattr(self, key, d[key])
                    d.pop(key)
                except TypeError:
                    raise TypeError("Error with type of variable %s" % (key))

        return d
