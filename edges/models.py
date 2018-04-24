import numpy as np


def model_fg_poly(nus, *parameters):
    """Polynomial foreground model, taken from Eq. (2) of Bowman+ (2018).

    Parameters
    ----------
    nus : 1D-array of floats
        Input frequencies in MHz

    parameters: Sequence of floats
        Polynomial coefficients for Eq. (2)

    Returns
    -------
    t_fg : 1D-array of floats, same shaps as nus
    """
    nu_c = nus/75.

    t_fg = np.zeros_like(nus, dtype=np.float64)
    for i, p in enumerate(parameters):
        t_fg += p * nu_c**(float(i) - 2.5)

    return t_fg


def model_fg_phys(nus, *parameters):
    """Physically motivated model, Eq. (1) in the paper.
    All frequencies are handled in MHz

    Parameters
    ----------
    nus : 1D-array of floats
        Input frequencies in MHz

    parameters: Sequence of floats
        Model coefficients a0 to a4 from Eq. (1)

    Returns
    -------
    temp : 1D-array of floats, same shaps as nus
    """

    a0, a1, a2, a3, a4 = parameters

    a0 *= 1.e0
    a1 *= 1.e0
    a2 *= 1.e0
    a3 *= 1.e0
    a4 *= 1.e0

    # Center of the band in MHz
    nu_c = 75.

    nu_norm = nus / nu_c
    log_nu = np.log(nu_norm)

    temp = (
        a0 * nu_norm**(-2.5) +
        a1 * nu_norm**(-2.5) * log_nu +
        a2 * nu_norm**(-2.5) * log_nu**2 +
        a3 * nu_norm**(-4.5) +
        a4 * nu_norm**(-2.)
        )

    return temp


def model_hi_trough(nus, *parameters):
    """Signal for the HI trough, Eq. after (2) in the paper.
    It is unfortunately not labelled.

    All frequencies are handled in MHz

    Parameters
    ----------
    nus : 1D-array of floats
    Input frequencies in Hz

    parameters: Sequence of floats
    Model coefficients:
    - amp: Amplitude
    - nu_0: Center frequency in MHz
    - tau: Flattening factor
    - w: FWHM of the trough in MHz

    Returns
    -------
    temp : 1D-array of floats, same shaps as nus
    """

    amp, nu_0, tau, w = parameters

    d_nu = nus - nu_0

    capital_b = 4 * (d_nu/w)**2. * np.log(-1./tau*np.log((1.+np.exp(-tau))/2.))

    t_21 = -amp*(
        (1. - np.exp(-tau*np.exp(capital_b))) /
        (1. - np.exp(-tau)))

    return t_21


def fg_phys_plus_21cm(nus, *parameters):
    """Just a combination of the two models introduced above
    """
    parameters_fg = parameters[:5]
    parameters_hi = parameters[-4:]

    t_sky = (
        model_fg_phys(nus, *parameters_fg) +
        model_hi_trough(nus, *parameters_hi)
        )

    return t_sky
