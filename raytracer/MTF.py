"""Computation of the geometric and diffraction limited Modulation Transfer Function (MTF)"""
import numpy as np
from typing import Tuple
from scipy.integrate import simpson
from scipy.special import jv
import matplotlib.pyplot as plt

def line_spread_function(coords, bins, magnification=1.0, smoothen=False):

    coords = np.asarray(coords)
    coords = (coords - coords.mean())*magnification
    hist, bin_edges = np.histogram(coords, bins)
    centers = bin_edges[0:-1] + (bin_edges[1:] - bin_edges[0:-1])/2.0

    return centers, hist

def cutoff_frequency(NA: float, wavelength: float) -> float:
    """Maximum spatial frequence (in lines per mm= that can be resolved due to diffraction.
    This is the limiting resolution for an aberration-free system.
    
    Args:
        NA: numerical aperture NA=n*sin(alpha) with index of refraction an an semi-opening angle alpha
        wavelength: wavelength of the illuminating light source (in mm)

    Returns:
        nu0: the cutoff frequency for coherent illumination (i.e. fully filled entrance pupil)
    """        
    nu0 = 2*NA/wavelength
    return nu0

def geometric_MTF(point_cloud: np.ndarray, NA: float, wavelength: float, test_target: str="sinusoidal") -> Tuple:
    """The geometric MTF is obtained by convolving the imaged (=magnified) test pattern with the geometric line spread function."""
    point_cloud = np.asarray(point_cloud)
    nu0 = cutoff_frequency(NA, wavelength)
    nu = np.linspace(0, nu0, 1000)

    assert len(point_cloud.shape) == 2
    assert point_cloud.shape[1] == 2
    if test_target not in ["sinusoidal", "square"]:
        raise ValueError(f"Unknown test_target {test_target}")

    LSF = {}
    OTF = {}
    MTF = {}
    for i, orientation in enumerate(["x", "y"]):
        LSF[orientation] = line_spread_function(point_cloud[:,i], bins=1000)
        
        delta = LSF[orientation][0]
        A = LSF[orientation][1]
        Anorm = simpson(A, delta)
        Acos = simpson(A[None,:]*np.cos(2*np.pi*delta[None,:]*nu[:,None]), delta, axis=-1) / Anorm
        Asin = simpson(A[None,:]*np.sin(2*np.pi*delta[None,:]*nu[:,None]), delta, axis=-1) / Anorm

        OTF_ = Acos[:] + 1j*Asin[:]
        MTF_ = np.abs(OTF_)
        OTF = (nu, OTF_)
        MTF = (nu, MTF_)

    if test_target in ["sinusoidal"]:
        return LSF, OTF, MTF

    elif test_target in ["square"]:
        raise NotImplementedError


def diffraction_limited_MTF(NA: float, wavelength: float, aperture_shape="circular", test_target="sinusoidal", defocus=0.0):
    """Exact diffraction limited modulation transfer function (MTF) for different aperture shapes,
    target patterns and different amounts of defocus (from the paraxial focus).
    
    Args:
        NA : float - numerical aperture NA = n*sin(alpha)
        wavelength : float - optical wavelength in mm
        aperture_shape : str
        test_target : str
        defocus : float - longitudinal defocusing in mm
    
    Returns:
        nu: ndarray - frequencies
        MTF: ndarray - MTF(nu)
    """
    assert aperture_shape in ["circular", "square"]
    assert test_target in ["sinusoidal", "square"]

    nu0 = cutoff_frequency(NA, wavelength)
    nu = np.linspace(0, nu0, 1000)

    if aperture_shape == "circular":
        phi = np.arccos(wavelength*nu[:]/(2*NA))
        MTF = (2/np.pi)*(phi - np.cos(phi)*np.sin(phi))        
        if not np.isclose(defocus, 0.0):
            # Take defocus into account by a multiplicative factor.
            MTF *= jv(1,2*np.pi*defocus*NA*nu[:])/(np.pi*defocus*NA*nu[:])

    elif aperture_shape == "square":
        raise NotImplementedError

    if test_target == "sinusoidal":
        return (nu, MTF)
    elif test_target == "square":
        raise NotImplementedError


