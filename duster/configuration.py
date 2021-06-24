"""Configuration class for duster.

This is based on the redmapper configuration.
"""
import numpy as np
import os
import logging

from redmapper.configuration import ConfigField, Configuration, read_yaml


class DusterConfiguration(Configuration):
    """Configuration class for duster.
    """
    duster_mapfile1 = ConfigField(required=True)
    duster_mapfile2 = ConfigField(required=True)
    duster_rhofile1 = ConfigField()
    duster_rhofile2 = ConfigField()

    duster_p_dust_file = ConfigField()
    duster_p_dust_chainfile = ConfigField()
    duster_rho_0 = ConfigField()
    duster_rho_min = ConfigField()
    duster_b = ConfigField()

    duster_nwalkers  = ConfigField(default=32)
    duster_nproc = ConfigField(default=2)

    duster_redgalfile = ConfigField()
    duster_chi2_max = ConfigField()
    duster_lum_min = ConfigField()
    duster_zrange = ConfigField(isArray=True, array_length=2, required=True)

    duster_color_indices = ConfigField(isArray=True, default=np.array([0, 1, 2]))

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

        self.validate()

        # Checks ...
        if self.maskfile is not None and self.mask_mode == 0:
            raise ValueError(("A maskfile is set, but mask_mode is 0 (no mask). "
                              "Assuming this is not intended."))

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
