import unittest
import os
import numpy.testing as testing
import numpy as np
import tempfile
import fitsio
import shutil

import duster


class RhoModelFitterTestCase(unittest.TestCase):
    """Test for fitting the rho model.
    """
    def test_rho_model_fitter(self):
        """Test the rho model fit.
        """
        config_file_name = 'testconfig.yaml'
        file_path = 'data_for_tests'

        config = duster.DusterConfiguration(os.path.join(file_path, config_file_name))

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestDuster-')
        config.outpath = self.test_dir

        # We first need to make rho maps...
        map_maker = duster.RhoMapMaker(config)

        rho_map1 = map_maker.normalize_map(config.duster_mapfile1, 128)
        rho_map2 = map_maker.normalize_map(config.duster_mapfile2, 128)

        fitter = duster.RhoModelFitter(config)

        np.random.seed(12345)

        fitter.fit_rho_model(rho_map1, rho_map2)

        # Check that the numbers are what we expect
        testing.assert_almost_equal(config.duster_rho_0, 0.1611384927054149, 6)
        testing.assert_almost_equal(config.duster_rho_min, 0.9007161281002618, 6)
        testing.assert_almost_equal(config.duster_b, 13.294168006029823, 6)

        # Check that the output files have been made and can be read
        self.assertTrue(os.path.isfile(config.duster_rho_model_file))

        rho_model = fitsio.read(config.duster_rho_model_file, ext=1)
        testing.assert_almost_equal(rho_model[0]['rho_0'], config.duster_rho_0)
        testing.assert_almost_equal(rho_model[0]['rho_min'], config.duster_rho_min)
        testing.assert_almost_equal(rho_model[0]['b'], config.duster_b)

        self.assertTrue(os.path.isfile(config.duster_rho_model_chainfile))
        lnprob = fitsio.read(config.duster_rho_model_chainfile, ext=0)
        self.assertEqual(len(lnprob), config.duster_nwalkers*config.duster_rho_model_nsample2)
        chain = fitsio.read(config.duster_rho_model_chainfile, ext=1)
        testing.assert_array_equal(chain.shape, (lnprob.size, 2))

        # Check that the plots have been made
        lnprobfile = config.redmapper_filename('rho_model_lnprob',
                                               paths=(config.plotpath, ),
                                               filetype='png')
        self.assertTrue(os.path.isfile(lnprobfile))
        cornerfile = config.redmapper_filename('rho_model_corner',
                                               paths=(config.plotpath, ),
                                               filetype='png')
        self.assertTrue(os.path.isfile(cornerfile))
        rhodistfile = config.redmapper_filename('rho_model_dist',
                                                paths=(config.plotpath, ),
                                                filetype='png')
        self.assertTrue(os.path.isfile(rhodistfile))

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
