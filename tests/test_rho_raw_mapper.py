import unittest
import os
import tempfile
import shutil
import healsparse as hsp

import redmapper
import duster


class RhoRawMapperTestCase(unittest.TestCase):
    """Test for making raw rho observed maps.
    """
    def test_rho_raw_map(self):
        """Test rho obs raw map.
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

        mapper = duster.RhoRawMapper(config)

        gals = redmapper.GalaxyCatalog.from_fits_file(config.duster_redgalfile)

        mapper.fit_rho_obs_raw_map(gals)

        # Check that the mapfile is there
        mapfile = config.redmapper_filename('rho_raw_map', filetype='hsp')
        self.assertTrue(os.path.isfile(mapfile))

        raw_map = hsp.HealSparseMap.read(mapfile)
        self.assertEqual(len(raw_map.valid_pixels), 63)

        # Check that the plots are there
        comp1file = config.redmapper_filename('rho_raw_%s_comp' % (config.duster_label1),
                                              paths=(config.plotpath, ),
                                              filetype='png')
        self.assertTrue(os.path.isfile(comp1file))
        comp2file = config.redmapper_filename('rho_raw_%s_comp' % (config.duster_label2),
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
