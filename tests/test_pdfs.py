import unittest
import numpy as np
import scipy.integrate
import numpy.testing as testing

from duster import pdfs
from duster.constants import RHO_INTEGRAL_OFFSET, RHO_INTEGRAL_MIN, RHO_INTEGRAL_MAX, RHO_INTEGRAL_STEP


class PdfTestCase(unittest.TestCase):
    def test_p_dust(self):
        """Test p_dust."""
        np.random.seed(12345)

        ntest = 1000

        # Compute a random selection of positive u, b values
        b = np.random.uniform(low=0.1, high=20.0, size=ntest)
        u = np.random.uniform(low=0.01, high=20.0, size=ntest)

        rho_vals = np.arange(RHO_INTEGRAL_MIN, RHO_INTEGRAL_MAX, RHO_INTEGRAL_STEP) + RHO_INTEGRAL_OFFSET

        integrals = np.zeros(ntest)

        for i in range(ntest):
            # Compute the normalization and p_dust
            rho_0, rho_min = pdfs.compute_normalized_rho_pars(u[i], b[i])
            p_dust = pdfs.p_dust(rho_0, b[i], rho_min, rho_vals)

            integrals[i] = scipy.integrate.simps(p_dust, rho_vals)

        # Ensure that the normalization is 1.0
        testing.assert_array_almost_equal(integrals, 1.0, decimal=4)


if __name__ == '__main__':
    unittest.main()
