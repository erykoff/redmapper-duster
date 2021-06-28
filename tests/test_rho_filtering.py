import unittest
import os
import tempfile
import shutil
import healsparse as hsp
import fitsio
import numpy.testing as testing

import duster


class RhoFilteringTestCase(unittest.TestCase):
    """Test for filtering rho maps.
    """
    def test_rho_filter_map(self):
        """Test rho filtering.
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

        filterer = duster.RhoMapFilterer(config)
        filterer.filter_map(rho_raw_map)

        # Check that the mapfile is there
        mapfile = config.redmapper_filename('rho_filtered_map', filetype='hsp')
        self.assertTrue(os.path.isfile(mapfile))
        self.assertEqual(mapfile, config.duster_filtered_map_file)

        modelfile = config.redmapper_filename('filter_model')
        self.assertTrue(os.path.isfile(modelfile))
        self.assertEqual(modelfile, config.duster_filter_model_file)

        model = fitsio.read(modelfile, ext=1)
        testing.assert_almost_equal(model[0]['k'], 3.5711356e-7)
        testing.assert_almost_equal(model[0]['eta'], 2.35808614)
        testing.assert_almost_equal(model[0]['C_noise'], 2.72645362e-08)

        # Check that the plots are there
        modfile = config.redmapper_filename('filter_model',
                                            paths=(config.plotpath, ),
                                            filetype='png')
        self.assertTrue(os.path.isfile(modfile))
        comp1file = config.redmapper_filename('rho_filt_%s_comp' % (config.duster_label1),
                                              paths=(config.plotpath, ),
                                              filetype='png')
        self.assertTrue(os.path.isfile(comp1file))
        comp2file = config.redmapper_filename('rho_filt_%s_comp' % (config.duster_label2),
                                              paths=(config.plotpath, ),
                                              filetype='png')
        self.assertTrue(os.path.isfile(comp2file))

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
