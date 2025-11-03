import numpy as np
import dataclasses

__about__ = ["LensSequence", "read_lens"]

@dataclasses.dataclass(frozen=True)
class LensSequence(object):
    num_surfs: int = 3  # number of surfaces including object surface 
    AS_surf: int = 1    # index of surface which is the aperture stop
    SAG: bool = True    # Should surface sag be taken into account ?
    lens_unit: str = "mm" # lens unit millimeter
    R: np.ndarray = np.array([np.inf, np.inf, np.inf])  # radii of curvate 
    t: np.ndarray = np.array([10, 0, 10])               # distances between surfaces
    zdist: np.ndarray = np.array([-10, 0, 0, 10])
    n: np.ndarray = np.array([1.0, 1.0, 1.0])           # index of refraction *after* each surface
    Vd: np.ndarray = np.array([np.inf, np.inf, np.inf]) # Abbe number of medium *after* each surface 
    phi: np.ndarray = np.array([0, 0, 0])               # surface power
    forward: bool = True                                # traverse lens system forward or backward


def read_lens(filename: str, SAG: bool = True) -> LensSequence:
    """
    Read lens description file and initialize LensSequence object.
    Determine the aperture stop surface.

    SAG = True takes surface sag into account.
    """
    lens_layout = np.loadtxt(filename, comments="#", skiprows=1)
    num_surfs = lens_layout.shape[0]

    stop_flag = np.zeros(num_surfs)
    R = np.zeros(num_surfs)
    t = np.zeros(num_surfs)
    zdist = np.zeros(num_surfs+1)
    n = np.zeros(num_surfs)
    Vd = np.zeros(num_surfs)
    phi = np.zeros(num_surfs)

    t[0] = lens_layout[0,3]
    for s in range(0, num_surfs):
        stop_flag[s] = lens_layout[s,1]
        R[s] = lens_layout[s, 2] 
        t[s] = lens_layout[s, 3]
        n[s] = lens_layout[s, 4]
        Vd[s] = lens_layout[s, 5]
        if s  > 0:
            phi[s] = (n[s] - n[s-1]) / R[s]  # surface power 

    # zdist[s] is the total distance of surface s from the first vertex of the lens system.
    # zdist[0] < 0 is the object distance.    
    zdist[0] = -t[0]
    zdist[1] = 0
    zdist[2:] = np.cumsum(t[1:])

    # Find the aperture stop and verify that there is only one aperture stop.
    AS_surf = 0 
    found_AS = False
    for s in range(1, num_surfs):    
        if stop_flag[s] == 1:
            if not found_AS:
                AS_surf = s
                found_AS = True
            else:
                raise ValueError("There can be only one aperture stop.")
    if AS_surf == 0:
        raise ValueError(f"The object surface cannot be the aperture stop. AS_surf = {AS_surf}")

    lens_sequence = LensSequence(
        num_surfs, 
        AS_surf, 
        SAG,
        "mm",
        R[:],
        t[:],
        zdist[:],
        n[:],
        Vd[:],
        phi[:],
        forward=True
    )

    return lens_sequence
