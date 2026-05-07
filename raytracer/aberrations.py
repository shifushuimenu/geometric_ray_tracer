"""Calculation of the five monochromatic ray aberrations as well as longitudinal and lateral chromatic aberration."""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Iterable

from raytracer.lens import LensSequence

__all__ = ["Seidel3rd_aberrations"]

class Aberrations(ABC):
    pass
    # @abstractmethod
    # def lens_unit_to_waves():
    #     "Aberration coefficients in units of wavelength."
    #     pass

@dataclass
class Aberrations3rd(Aberrations):
    S1: np.ndarray
    S2: np.ndarray
    S3: np.ndarray
    S4: np.ndarray
    S5: np.ndarray
    PetzvalRadius: float

class Aberrations5th(Aberrations):
    pass
    # raise NotImplementedError


def Seidel3rd_aberrations(y_chief: Iterable, u_chief: Iterable, y_marg: Iterable, u_marg: Iterable, 
                          lens_sequence: LensSequence) -> Aberrations3rd:
    """
    Calculate Seidel coefficients for third-order monochromatic ray aberrations.
    For a derivation of the formulae see [Hopkins 1950, Wave Theory of Aberrations].

    Parameters
    ----------
    y_chief, u_chief: ray height and ray angle of the chief ray for the ray bundle *at maximum object height*
    y_marg, u_marg: ray height and ray angle for the marginal ray of the *on-axis* ray bundle
    lens_sequence: LensSequence object

    Returns
    -------
    Seidel coefficients for each surface and their sum, Petzval radius.
    """
    num_surfs = lens_sequence.num_surfs
    n = lens_sequence.n
    R = lens_sequence.R

    CRI = np.zeros(num_surfs-1)
    MRI = np.zeros(num_surfs-1)
    L = np.zeros(num_surfs-1)
    # Image surface is excluded. Instead, the last element of S1..5[:] contains the Seidel sum over all surfaces.
    S1 = np.zeros(num_surfs) # spherical
    S2 = np.zeros(num_surfs) # coma
    S3 = np.zeros(num_surfs) # astigmatism
    S4 = np.zeros(num_surfs) # field curvature
    S5 = np.zeros(num_surfs) # distortion
    PetzSum = 0.0
    for i in range(1,num_surfs-1):
        MRI[i] = n[i]*(y_marg[i]/R[i] + np.tan(u_marg[i])) # marginal ray "invariant" for the *on-axis* ray bundle (an invariant at a refracting surface but not under propagation)
        CRI[i] = n[i]*(y_chief[i]/R[i] + np.tan(u_chief[i])) # chief ray for the ray bundle *at maximum object* height 
        L[i] = n[i-1]*(y_marg[i]*np.tan(u_chief[i-1]) - y_chief[i]*np.tan(u_marg[i-1])) # Lagrange invariant for the above two rays
        S1[i] = -MRI[i]*MRI[i]*y_marg[i]*(np.tan(u_marg[i])/n[i] - np.tan(u_marg[i-1])/n[i-1])
        S2[i] = S1[i]*CRI[i]/MRI[i]
        S3[i] = S2[i]*CRI[i]/MRI[i]
        S4[i] = -L[i]*L[i]*((1/n[i]) - (1/n[i-1]))/R[i]
        S5[i] = -(S3[i] + S4[i])*CRI[i]/MRI[i]
        PetzSum += (1/R[i])*((1/n[i]) - (1/n[i-1]))

    S1sum = np.sum(S1); S2sum = np.sum(S2); S3sum = np.sum(S3); S4sum = np.sum(S4); S5sum = np.sum(S5)
    PetzvalRadius = 1.0 / PetzSum

    S1[-1] = S1sum
    S2[-1] = S2sum
    S3[-1] = S3sum
    S4[-1] = S4sum
    S5[-1] = S5sum

    return Aberrations3rd(S1, S2, S3, S4, S5, PetzvalRadius)


def chromatic_aberrations():
    raise NotImplementedError
