import unittest
import os
import numpy.testing as testing
import tempfile
import fitsio
import shutil
import healsparse as hsp

import duster


class DebiasFitterTestCase(unittest.TestCase):
    """Test for fitting the debias model.
    """
    def test_debias_fitter(self):
        """Test the debias model fit.
        """
        config_file_name = 'testconfig.yaml'
        file_path = 'data_for_tests'

        config = duster.DusterConfiguration(os.path.join(file_path, config_file_name))

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestDuster-')
        config.outpath = self.test_dir

        fname = config.redmapper_filename('rho1')
        map_maker = duster.RhoMapMaker(config)
        rho_map = map_maker.normalize_map(config.duster_mapfile1, config.duster_nside)

        rho_map.write(fname)
        config.duster_rhofile1 = fname
        config.duster_rhofile2 = fname

        config.duster_raw_map_file = os.path.join(file_path, 'test_raw_map_buzz_bigreg.hsp')

        rho_raw_map = hsp.HealSparseMap.read(config.duster_raw_map_file)

        config.duster_rho_0 = 0.1611
        config.duster_rho_min = 0.9007
        config.duster_b = 13.2942

        # We will test the fit on this unfiltered map.

        fitter = duster.DebiasFitter(config)
        fitter.fit_debias_model(rho_raw_map, testing=True)

        self.assertTrue(os.path.isfile(config.duster_debias_model_chainfile))
        lnprob = fitsio.read(config.duster_debias_model_chainfile, ext=0)
        self.assertEqual(len(lnprob), config.duster_nwalkers*config.duster_debias_model_nsample2)
        chain = fitsio.read(config.duster_debias_model_chainfile, ext=1)
        testing.assert_array_equal(chain.shape, (lnprob.size, 4))

        lnprobfile = config.redmapper_filename('debias_model_lnprob',
                                               paths=(config.plotpath, ),
                                               filetype='png')
        self.assertTrue(os.path.isfile(lnprobfile))
        cornerfile = config.redmapper_filename('debias_model_corner',
                                               paths=(config.plotpath, ),
                                               filetype='png')
        self.assertTrue(os.path.isfile(cornerfile))

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
