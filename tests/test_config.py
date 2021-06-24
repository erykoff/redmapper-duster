import unittest
import os

import duster


class ReadConfigTestCase(unittest.TestCase):
    """Test for reading the configuration.
    """
    def test_readconfig(self):
        """Test reading of configuration.
        """
        file_name = 'testconfig.yaml'
        file_path = 'data_for_tests'

        config = duster.DusterConfiguration(os.path.join(file_path, file_name))

        self.assertEqual(config.duster_chi2_max, 8.0)


if __name__ == '__main__':
    unittest.main()
