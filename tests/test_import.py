import unittest


class ImportTestCase(unittest.TestCase):
    def test_import(self):
        """Test trivial import."""
        import duster  # noqa: F401


if __name__ == '__main__':
    unittest.main()
