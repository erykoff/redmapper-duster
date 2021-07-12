import unittest
import numpy as np
import scipy.integrate
import numpy.testing as testing

from duster import pdfs
from duster.constants import (RHO_INTEGRAL_OFFSET,
                              RHO_INTEGRAL_MIN,
                              RHO_INTEGRAL_MAX,
                              RHO_INTEGRAL_STEP,
                              RHO_INTEGRAL_STEP2)


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

            integrals[i] = scipy.integrate.simpson(p_dust, rho_vals)

        # Ensure that the normalization is 1.0
        testing.assert_array_almost_equal(integrals, 1.0, decimal=4)

    def test_p_abgs(self):
        """Test Pabgs."""
        np.random.seed(12345)

        ntest = 10

        # Compute a random selection of u, b values near model pars
        b = np.random.uniform(low=4.0, high=5.0, size=ntest)
        u = np.random.uniform(low=0.6, high=0.8, size=ntest)

        alpha = np.random.uniform(low=0.2, high=1.2, size=ntest)
        beta = np.random.uniform(low=0.2, high=1.2, size=ntest)
        gamma = np.random.uniform(low=-1.0, high=0.5, size=ntest)
        sigma2 = np.random.uniform(low=0.01**2., high=0.15**2., size=ntest)

        rho_vals = np.arange(RHO_INTEGRAL_MIN, RHO_INTEGRAL_MAX, RHO_INTEGRAL_STEP2) + RHO_INTEGRAL_OFFSET
        rho_twiddle_vals = np.arange(-20.0, 20.0, RHO_INTEGRAL_STEP2)

        integrals = np.zeros(ntest)

        for i in range(ntest):
            rho_0, rho_min = pdfs.compute_normalized_rho_pars(u[i], b[i])
            pabgs = pdfs.Pabgs(rho_vals, rho_0, rho_min, b[i])

            vals = np.zeros(rho_twiddle_vals.size)
            for j in range(rho_twiddle_vals.size):
                vals[j] = pabgs(alpha[i], beta[i], gamma[i], sigma2[i], rho_twiddle_vals[j])
            integrals[i] = scipy.integrate.simpson(vals, rho_twiddle_vals)

        # Ensure that the normalization is 1.0
        testing.assert_array_almost_equal(integrals, 1.0, decimal=4)


if __name__ == '__main__':
    unittest.main()
