from __future__ import print_function, division, absolute_import

import numpy as np


# Fundamental constants
c = 299792458.0 # speed of light in m/s
k_B = 1.380649e-23 # Boltzmann constant in J/K
m_p = 1.672621e-27 # proton mass in kg
m_e = 9.109383e-31 # electon mass in kg
e_charge = 1.602176e-19 # electon charge in C

# Lyman-alpha (2p -> 1s transition) properties
A_Lya = 6.265e8 # Einstein A coefficient
f_Lya = 0.4162 # oscillator strength
wl_Lya = 1215.6701 # wavelength Angstrom
nu_Lya = 1e10 * c / wl_Lya # frequency in Hz

K_Lya = 1e-7 * f_Lya * np.sqrt(np.pi) * e_charge**2 * c / m_e

def deltanu_D(T, b_turb):
    """
    
    Thermally broadened frequency (in Hz), given a temperature `T` in K (see Dijkstra 2014)
    and optional turbulent velocity `b` in km/s
    
    
    """
    return nu_Lya * np.sqrt(2.0 * k_B * T / m_p + (b_turb*1e3)**2) / c

def Voigt(x, T, b_turb):
    """
    
    Approximation of Voigt function of `x` (from Tasitsiomi 2006 or Tepper-Garcia 2006),
    given a temperature `T` in K and optional turbulent velocity `b` in km/s
    
    
    """

    # Voigt parameter
    a_V = A_Lya / (4.0 * np.pi * deltanu_D(T, b_turb=b_turb))
    x_squared = x**2

    z = (x_squared - 0.855) / (x_squared + 3.42)
    q = np.where(z > 0.0, z * (1.0 + 21.0/x_squared) * a_V / (np.pi * (x_squared + 1.0)) * (0.1117 + z * (4.421 + z * (5.674 * z - 9.207))), 0.0)
    
    return np.sqrt(np.pi) * (q + np.exp(-x_squared)/1.77245385)

def sigma_alpha(x, T, b_turb=0.0):
    """
    
    Lyman-alpha scattering cross section (e.g. Dijkstra 2014) given a temperature `T` in K and optional turbulent velocity `b` in km/s;
    note that electron charge and speed of light work slightly differently in SI and cgs units
    
    
    """
    
    sigma_alpha = K_Lya / deltanu_D(T, b_turb=b_turb) * Voigt(x, T, b_turb=b_turb)
    
    # Use the linear Lee (2013) correction
    sigma_alpha *= (1.0 - 1.792 * (x * deltanu_D(T, b_turb=b_turb)) / nu_Lya)
    
    return sigma_alpha

def dla_trans(wl_emit_array, N_HI, T, b_turb=0.0):
    """
    
    Lyman-alpha optical depth given a neutral hydrogen column density `N_HI` in cm^-2,
    temperature `T` in K, and optional turbulent velocity `b` in km/s
    
    
    """
    
    # Dimensionless Doppler parameter (use the frequency definition; see Webb, Lee 2021)
    nu = 1e10 * c / wl_emit_array # frequency in Hz
    x = (nu - nu_Lya) / deltanu_D(T, b_turb=b_turb)
    
    # Convert column density from cm^-2 to m^-2 as the cross section is in m^2
    tau = N_HI * 1e4 * sigma_alpha(x, T, b_turb=b_turb)
    
    return np.exp(-tau)