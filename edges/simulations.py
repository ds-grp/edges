import os
import site

import pandas as pd
import numpy as np

from . import data
site.addsitedir('/users/dlenz/software/pymodules/edges/external_code/')

from global_util import define_cosmology as def_cosmo
from global_util import define_global as def_global
from global_util import dT_global


def sigma2fwhm(sigma):
    return sigma * (2.*np.sqrt(2.*np.log(2.)))


def flat_gauss(x, amp, x0, tau, sigma):
    """Gaussian with a flat top, parametrized through the flatness tau.
    Parameters
    ----------
    x : 1D-array of floats
        Input array
    amp : Amplitude
    x0 : Center of the Gaussian
    tau : Flattening factor
    sigma : Standard deviation of the Gaussian

    Returns
    -------
    y : 1D-array of floats, same shaps as x
    """

    fwhm = sigma2fwhm(sigma)
    d_x = x - x0

    capital_b = (
        4 * (d_x/fwhm)**2. *
        np.log(-1./tau*np.log((1.+np.exp(-tau))/2.)))

    y = amp*(
        (1. - np.exp(-tau*np.exp(capital_b))) /
        (1. - np.exp(-tau)))

    return y


def get_noise_realization():
    """Based on the residuals in the original Bowman+ (2018) data set,
    generate a random realization of noise that matches the variance.
    """
    df = data.fetch_processed()
    var_in = df['dt21'].values**2

    noise_realization = np.sqrt(var_in) * np.random.uniform(
        low=-1., high=1.,
        size=len(var_in))

    return noise_realization


class Injection(object):
    def __init__(self, freqs):
        """Freqs in MHz"""
        self._freqs = freqs

    @property
    def freqs(self):
        # Ensure that pandas.Series are correctly extracted
        if isinstance(self._freqs, pd.Series):
            self._freqs = self._freqs.values
        return self._freqs

    def inject(self):
        raise NotImplementedError('Use derived classes')


class Gauss(Injection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def inject(self, *params):
        """Parameters are
        amp, freq0, sigma = params

        amp in K
        freq0 in MHz
        sigma in MHz
        """

        amp, freq0, sigma = params
        val = -amp * np.exp(-0.5*((self.freqs-freq0)/sigma)**2)
        return val


class FlatGauss(Injection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def inject(self, *params):
        """Parameters are
        amp, freq0, tau, sigma = params

        amp in K
        freq0 in MHz
        flatness tau is dimensionless
        sigma in MHz
        """

        amp, freq0, tau, sigma = params
        val = flat_gauss(self.freqs, -amp, freq0, tau, sigma)
        return val


class Physical(Injection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def inject(self):
        cosmo = def_cosmo()
        glob = def_global()

        # Need to convert from mK to K
        val = dT_global(self.freqs, cosmo, glob) / 1000.
        return val
