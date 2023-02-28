import numpy as np
import matplotlib.pyplot as plt
import healsparse as hsp
import hpgeom as hpg
import redmapper


class A0SumComputer:
    """Class to compute a0 sums.

    Parameters
    ----------
    zredstr : `redmapper.RedSequenceColorPar`
        Red sequence parameterization object.
    indices : `np.ndarray`
        Indices to sum for colors.
    r_band : array-like
        Array of normalized reddening constants.
    rho_map : `healsparse.HealSparseMap`
        Normalized, debiased rho map.
    """
    def __init__(self, zredstr, indices, r_band, rho_map):
        self.zredstr = zredstr
        self.indices = indices
        self.r_band = r_band
        self.rho_map = rho_map

        self.r_col = np.array(r_band)[0: -1] - np.array(r_band)[1:]

    def compute_a0_sums(self, gals, use_ztrue=False):
        """Compute a0 sums.

        Parameters
        ----------
        gals : `redmapper.GalaxyCatalog`
            Galaxies to compute a0.

        Returns
        -------
        sum_omega_deltac : `float`
            Numerator of a0 equation.
        sum_omega : `float`
            Denomenator of a0 equation.
        """
        sum_omega_deltac = 0.0
        sum_omega = 0.0

        galcol = gals.galcol
        galcol_err = gals.galcol_err

        if use_ztrue:
            zuse = gals.ztrue
        else:
            zuse = gals.zred_uncorr

        zind = self.zredstr.zindex(zuse)

        rho = self.rho_map.get_values_pos(gals.ra, gals.dec)

        # good, = np.where(rho > hpg.UNSEEN)
        good, = np.where(rho > 1.0)

        for i, index in enumerate(self.indices):
            r_col = self.r_col[index]

            c_pred = (self.zredstr.c[zind, index] +
                      self.zredstr.slope[zind, index]*(gals.refmag -
                                                       self.zredstr.pivotmag[zind]))
            err2 = galcol_err[:, index]**2. + self.zredstr.sigma[index, index, zind]**2.

            omega_mn = (1.0857*r_col*rho[good])**2./err2[good]

            omega_deltac = omega_mn * (galcol[good, index] - c_pred[good]) / (1.0857*r_col*rho[good])

            sum_omega = np.sum(omega_mn)
            sum_omega_deltac = np.sum(omega_deltac)

            if True:
                from fgcm.fgcmUtilities import dataBinner

                y_axis = (galcol[good, index] - c_pred[good])/(r_col*rho[good])

                binstruct = dataBinner(rho[good], y_axis, 0.2, [0.0, 6.0], nTrial=10)

                plt.clf()
                plt.hexbin(rho[good], y_axis, bins='log',
                           extent=[0, 6, -0.2, 0.2])
                ok, = np.where(binstruct['Y_ERR'] > 0.0)
                plt.plot(binstruct['X'][ok], binstruct['Y'][ok], 'r.')
                plt.plot([0, 6], [0.056, 0.056], 'k--')
                a0_est = np.sum(omega_deltac)/np.sum(omega_mn)
                plt.plot([0, 6], [a0_est, a0_est], 'k-')
                plt.xlabel('rho')
                plt.ylabel(f'a0_{index}')
                plt.savefig(f'test_a0_plot_{index}.png')

            if False:
                from fgcm.fgcmUtilities import dataBinner

                binstruct = dataBinner(rho[good], galcol[good, index] - c_pred[good], 0.2, [0.0, 6.0], nTrial=10)

                plt.hexbin(rho[good], galcol[good, index] - c_pred[good], bins='log',
                           extent=[0, 6, -0.5, 0.5])
                ok, = np.where(binstruct['Y_ERR'] > 0.0)
                plt.plot(binstruct['X'][ok], binstruct['Y'][ok], 'r.')
                plt.show()

                # galcol_corr = galcol[:, index] - 1.2*0.048*r_col*rho
                # galcol_corr = galcol[:, index] - 0.0299*rho
                # galcol_corr = galcol[:, index] - 0.034783528932118384*rho

                # galcol_corr = galcol[:, index] - 0.016*rho
                # galcol_corr = galcol[:, index] - 0.0129*rho

                print(np.sum(omega_deltac[good])/np.sum(omega_mn[good]))
                galcol_corr = galcol[:, index] - (np.sum(omega_deltac[good])/np.sum(omega_mn[good]))*r_col*rho

                binstruct = dataBinner(rho[good], galcol_corr[good] - c_pred[good], 0.2, [0.0, 6.0], nTrial=10)

                plt.hexbin(rho[good], galcol_corr[good] - c_pred[good], bins='log',
                           extent=[0, 6, -0.5, 0.5])
                ok, = np.where(binstruct['Y_ERR'] > 0.0)
                plt.plot(binstruct['X'][ok], binstruct['Y'][ok], 'r.')
                plt.show()

                y_axis = (galcol[:, index] - c_pred)/(r_col*rho)

                binstruct = dataBinner(rho[good], y_axis[good], 0.2, [0.0, 6.0], nTrial=10)

                plt.hexbin(rho[good], y_axis[good], bins='log',
                           extent=[0, 6, -0.5, 0.5])
                ok, = np.where(binstruct['Y_ERR'] > 0.0)
                plt.plot(binstruct['X'][ok], binstruct['Y'][ok], 'r.')
                plt.show()


        return sum_omega_deltac, sum_omega
        # return numerator, denominator


class A0Fitter:
    """Class for fitting a0.

    Parameters
    ----------
    config : `duster.DusterConfiguration`
    """
    def __init__(self, config):
        self.config = config

    def fit_a0(self, gals, rho_map):
        """Fit a0 normalization for a set of galaxies and a rho map.

        Parameters
        ----------
        gals : `redmapper.GalaxyCatalog`
        rho_map : `healsparse.HealSparseMap`

        Returns
        -------
        a0 : `float`
        a0_err : `float`
        """
        self.config.logger.info("Computing a0...")

        zredstr = redmapper.RedSequenceColorPar(self.config.parfile, fine=True)
        indices = self.config.duster_color_indices
        r_band = self.config.dered_const_norm

        a0_computer = A0SumComputer(zredstr, indices, r_band, rho_map)

        sum_omega_deltac, sum_omega = a0_computer.compute_a0_sums(gals)

        a0 = sum_omega_deltac / sum_omega
        a0_err = 1./np.sqrt(sum_omega)

        self.config.logger.info("a0 = %.7f +/- %.7f", a0, a0_err)

        return a0, a0_err
