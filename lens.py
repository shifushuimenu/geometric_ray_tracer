import numpy as np
import dataclasses
from typing import Iterable, Tuple

__about__ = ["LensSequence", "read_lens"]

@dataclasses.dataclass(frozen=True)
class LensSequence(object):
    num_surfs: int = 3  # number of surfaces including object surface 
    AS_surf: int = 1    # index of surface which is the aperture stop
    stop_flag: np.ndarray = np.array([0,1,0])
    SAG: bool = True    # Should surface sag be taken into account ?
    lens_unit: str = "mm" # lens unit millimeter
    R: np.ndarray = np.array([np.inf, np.inf, np.inf])  # radii of curvate 
    t: np.ndarray = np.array([10, 0, 10])               # distances between surfaces
    zdist: np.ndarray = np.array([-10, 0, 0, 10])
    n: np.ndarray = np.array([1.0, 1.0, 1.0])           # index of refraction *after* each surface
    Vd: np.ndarray = np.array([np.inf, np.inf, np.inf]) # Abbe number of medium *after* each surface 
    phi: np.ndarray = np.array([0, 0, 0])               # surface power
    forward: bool = True                                # traverse lens system forward or backward
    y_nnint: np.ndarray = dataclasses.field(init=False)         # y-coord where neighbouring surfaces intersect
    z_nnint: np.ndarray = dataclasses.field(init=False)         # z-coord where neighbouring surfaces intersect

    def __post_init__(self):
        _z, _y = _calc_surface_intersections(self)
        super().__setattr__("y_nnint", _y)
        super().__setattr__("z_nnint", _z)


def read_lens(filename: str, SAG: bool = True, lens_unit: str ="mm") -> LensSequence:
    """
    Read lens description file and initialize LensSequence object.
    Load txt file, determine the surface powers and locate the surface which 
    is the aperture stop. Make sure there is only one aperture stop. 

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

    print("Aperture stop is surface", AS_surf, "at", np.sum(t[1:AS_surf]), 
          f"{lens_unit} from the front vertex.")

    lens_sequence = LensSequence(
        num_surfs, 
        AS_surf, 
        stop_flag,
        SAG,
        lens_unit,
        R[:],
        t[:],
        zdist[:],
        n[:],
        Vd[:],
        phi[:],
        forward=True,
    )

    return lens_sequence


def _surface_intersection_point(R_first: float, R_second: float, t: float) -> float:
    """
    Calculate the maximal clear aperture (CA) for a lens element. This is the maximal radial extent
    where, given two radii of curvature, R_first and R_second and the distance of vertices t, 
    the lens element still is "physically sound", i.e. the two spherical surfaces have not 
    intersected.

    The maximal clear apertures are used to determine whether for a given object height a chief 
    ray can be found at all that reaches the object height before leaving the maximal clear aperture 
    of some surface. This detects impossible object heights.

    Parameters
    ----------
    R_first: radius of curvature the first surface 
    R_second: radius of curvature the second surface 
    t: distance between the vertices of first and second surface

    Returns
    -------
    z_int: intersection point on the optical axis measured relative to the vertex of the first surface
           None if no intersection
    y_int: intersection point in radial direction
           None if no intersection
    """
    print(f"R_first={R_first}, R_second={R_second}, t={t}")    

    # Find intersection points of neighbouring surfaces in a plane containing the optical axis.

    # two parallel planes
    if (np.abs(R_first) == np.abs(R_second) == np.inf):
        return None, None
        
    if (R_first < 0 and R_second > 0 and t > 0): # concave lens element
        return None, None

    CONVEX_OR_PLANOCONVEX = (R_first > 0 and R_second < 0) # convex or plano-convex
    MENISCUS_CURVED_LEFT = R_first < 0 and R_second < 0 and np.abs(R_first) > np.abs(R_second) # meniscus lens, curved to the left
    MENISCUS_CURVED_RIGHT = R_first > 0 and R_second > 0 and R_second > R_first # meniscus lens, curved to the right

    # Two centered spheres have an intersection point 
    if ( CONVEX_OR_PLANOCONVEX or MENISCUS_CURVED_LEFT or MENISCUS_CURVED_RIGHT ):        
        z_int = t*(2*R_second + t) / (2*(t + R_second - R_first))
        absy_int = np.sqrt(2*z_int*R_first - z_int**2)        
        return z_int, absy_int
    else:
        # meniscus lens, but without intersecting surfaces
        return None, None
    

def _calc_surface_intersections(lens_sequence: LensSequence) -> Tuple[Iterable[float], Iterable[float]]:
    """
    Calculate maximal clear apertures for every surface such that curved surfaces do not intersect.
    
    Parameters
    ----------
    lens_sequence : LensSequence object 

    Returns
    -------
    z_int[0:num_surfs+1] : for each surface, the z-coord of the intersection with a neighbouring surface
    y_int[0:num_surfs+1] : the absolute value of the y-coord of the intersection, i.e. the maximal CA.

    For a flat surface s without intersection z_int[s] is the location of the vertex and y_int[s] = inf.
    """
    z_int = lens_sequence.zdist.copy()
    absy_int = np.empty(lens_sequence.num_surfs+1)
    absy_int.fill(np.inf)

    for i in range(1, lens_sequence.num_surfs-1): # exclude object and image surface
        z_tmp = []
        absy_tmp = []
        if (lens_sequence.R[i] < 0): # surface curved to the left
            # check intersection points with up to three surfaces to the left
            for j in range(i-1,max(i-3, 0),-1):
                z_, y_ = _surface_intersection_point(lens_sequence.R[j], lens_sequence.R[i], np.sum(lens_sequence.t[j:i]))
                if z_ is not None:
                    z_tmp.append(z_); absy_tmp.append(y_)
            # all z-coordinates of intersection points should be negative -> take closest intersection point
            if len(z_tmp) > 0:
                z_int[i] = lens_sequence.zdist[j] + np.max(z_tmp)
                absy_int[i] = absy_tmp[np.argmax(z_tmp)]
        elif (lens_sequence.R[i] > 0): # surface curved to the right
            # check intersection points with up to three surfaces to the right
            for j in range(i+1,min(i+4, lens_sequence.num_surfs),+1):
                z_, y_ = _surface_intersection_point(lens_sequence.R[i], lens_sequence.R[j], np.sum(lens_sequence.t[i:j]))
                if z_ is not None:
                    z_tmp.append(z_); absy_tmp.append(y_)
            # all z-coordinates of intersection points should be positive
            if len(z_tmp) > 0:
                z_int[i] = lens_sequence.zdist[i] + np.min(z_tmp)
                absy_int[i] = absy_tmp[np.argmin(z_tmp)]
        else:
            raise ValueError("Zero radius of curvature is not allowed")

    return z_int, absy_int