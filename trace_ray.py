import numpy as np
from utils import timer_func
from dataclasses import dataclass

@dataclass(frozen=True)
class LensSequence(object):
    num_surfs: int = 3  # number of surfaces including object surface 
    AS_surf: int = 1    # index of surface which is the aperture stop
    SAG: bool = True    # Should surface sag be taken into account ?
    lens_unit: str = "mm"
    R: np.ndarray = np.array([np.inf, np.inf, np.inf])  # radii of curvate 
    t: np.ndarray = np.array([10, 0, 10])               # distances between surfaces 
    n: np.ndarray = np.array([1.0, 1.0, 1.0])           # index of refraction *after* each surface
    Vd: np.ndarray = np.array([np.inf, np.inf, np.inf]) # Abbe number of medium *after* each surface 
    phi: np.ndarray = np.array([0, 0, 0])               # surface power

@timer_func
def trace_ray(y0, u0, lens_sequence, surf_start=0, forward=True):
    """
    Trace a batch of rays forward (left-to-right) or backwards (right-to-left) till right after the last or 
    right before the first surface of the lens system. Surface sag is considered. 
    """
    assert y0.shape == u0.shape
    assert 0 <= surf_start <= lens_sequence.num_surfs
    batch_dim = y0.shape[0:]
    y = np.zeros((lens_sequence.num_surfs+1,)+batch_dim)
    u = np.zeros_like(y)
    z_sag = np.zeros_like(y)

    y[0,...] = y0[...]
    u[0,...] = u0[...]
    y[1,...] = y[0,...] + np.tan(u[0,...])*lens_sequence.t[0]
    for i in range(surf_start+1, lens_sequence.num_surfs):
        if np.isinf(lens_sequence.R[i]) or not lens_sequence.SAG:
            u[i,...] = np.arctan((lens_sequence.n[i-1]/lens_sequence.n[i])*np.tan(u[i-1,...]) - lens_sequence.phi[i]*y[i,...]/lens_sequence.n[i])
            y[i+1,...] = y[i,...] + np.tan(u[i,...])*lens_sequence.t[i]   

        # take surface sag into account
        else:                
            y0 = y[i,...]
            u0 = u[i-1,...]
            # intersection with spherical surface
            sgnR = np.sign(lens_sequence.R[i]) # The formula depends on the sign of the radius of curvature.
            tanu0 = np.tan(u0)
            Delta = lens_sequence.R[i]**2 - 2*y0*tanu0*lens_sequence.R[i] - y0**2            
            assert np.all(Delta > 0) #, "Delta[0] < 0, %f"%(Delta[0])  
            zp = (lens_sequence.R[i] - y0*tanu0 - sgnR*np.sqrt(Delta))/(1 + tanu0**2)                     
            yp = y0 + tanu0*zp            
            theta = np.arctan(sgnR*yp/(lens_sequence.R[i]-zp))
            u_prime = sgnR*(np.arcsin((lens_sequence.n[i-1]/lens_sequence.n[i])*np.sin(theta + sgnR*u0)) - theta)            

            z_sag[i,...] = zp
            # y_intersection[i,r,f] = yp
            y[i,...] = yp # reset to value at intersection point
            u[i,...] = u_prime 
            y[i+1,...] = yp + np.tan(u_prime)*(lens_sequence.t[i] - zp)

    u[lens_sequence.num_surfs,...] = u[lens_sequence.num_surfs-1,...] # The image surface does not refract
    z_sag[lens_sequence.num_surfs,...] = 0 # Assumes that the image surface is flat.

    return y, u, z_sag