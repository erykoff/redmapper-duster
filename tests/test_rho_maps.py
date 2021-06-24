import unittest
import os
import numpy.testing as testing
import numpy as np

import duster


class RhoMapsTestCase(unittest.TestCase):
    """Test for making rho maps.
    """
    def test_make_rho_map(self):
        """Test rho map.
        """
        config_file_name = 'testconfig.yaml'
        file_path = 'data_for_tests'

        config = duster.DusterConfiguration(os.path.join(file_path, config_file_name))

        map_maker = duster.RhoMapMaker(config)

        for nside in [64, 128]:
            rho_map = map_maker.normalize_map(config.duster_mapfile1, nside)

            self.assertEqual(rho_map.nside_sparse, nside)
            testing.assert_almost_equal(np.mean(rho_map[rho_map.valid_pixels]), 1.0)
