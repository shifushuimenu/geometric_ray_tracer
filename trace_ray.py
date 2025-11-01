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
def trace_ray(y_start, u_start, lens_sequence, surf_start=0, forward=True):
    """
    Trace a batch of rays forward (left-to-right) or backwards (right-to-left) till right after the last or 
    right before the first surface of the lens system. Surface sag is considered. 
    """
    y_start = np.asarray(y_start)
    u_start = np.asarray(u_start)
    assert y_start.shape == u_start.shape
    assert 0 <= surf_start <= lens_sequence.num_surfs
    batch_dim = y_start.shape[0:]
    y = np.zeros((lens_sequence.num_surfs+1,)+batch_dim) # ray height at the curved surface
    y_vertexplane = np.zeros_like(y)  # ray height at the vertex plane
    u = np.zeros_like(y)
    z_sag = np.zeros_like(y)

    # The ray height `y_start`` is meant to be measured at the vertex plane tangential to the surface with index `surf_start`.
    # This means that surface sag is not taken into account yet.
    y_vertexplane[surf_start,...] = y_start[...]
    y[surf_start,...] = y_start[...].copy()
    u[surf_start,...] = u_start[...]

    if surf_start == 0: # starting at the object surface 
        y[surf_start+1,...] = y[surf_start,...] + np.tan(u[surf_start,...])*lens_sequence.t[surf_start]
    else: # start at an intermediate, possibly curved surface -> take surface sag into account
          # This duplication of code is a bit messy and could be simplified.
        y0 = y[surf_start,...]
        u0 = u[surf_start,...]
        # intersection with spherical surface with positive or negative radius of curvature
        sgnR = np.sign(lens_sequence.R[surf_start]) # The formula depends on the sign of the radius of curvature.
        tanu0 = np.tan(u0)
        Delta = lens_sequence.R[surf_start]**2 - 2*y0*tanu0*lens_sequence.R[surf_start] - y0**2            
        assert np.all(Delta > 0) #, "Delta[0] < 0, %f"%(Delta[0])  
        zp = (lens_sequence.R[surf_start] - y0*tanu0 - sgnR*np.sqrt(Delta))/(1 + tanu0**2)                     
        yp = y0 + tanu0*zp            
        theta = np.arctan(sgnR*yp/(lens_sequence.R[surf_start]-zp))
        u_prime = sgnR*(np.arcsin((lens_sequence.n[surf_start-1]/lens_sequence.n[surf_start])*np.sin(theta + sgnR*u0)) - theta)            

        z_sag[surf_start,...] = zp
        # y_intersection[i,r,f] = yp
        y[surf_start,...] = yp # reset to value at intersection point, i.e. now surface sag is taken into account
        # u[surf_start,...] = u_prime  # u[surf_start,...] is the angle *behind* the surface.
        y[surf_start+1,...] = yp + np.tan(u[surf_start,...])*(lens_sequence.t[surf_start] - zp)
        y_vertexplane[surf_start+1,...] = y[surf_start+1,...].copy()

    for i in range(surf_start+1, lens_sequence.num_surfs):
        if np.isinf(lens_sequence.R[i]) or not lens_sequence.SAG:
            u[i,...] = np.arctan((lens_sequence.n[i-1]/lens_sequence.n[i])*np.tan(u[i-1,...]) - lens_sequence.phi[i]*y[i,...]/lens_sequence.n[i])
            y[i+1,...] = y[i,...] + np.tan(u[i,...])*lens_sequence.t[i]   

        # take surface sag into account
        else:                
            y0 = y[i,...]
            u0 = u[i-1,...]
            # intersection with spherical surface with positive or negative radius of curvature
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
            y[i,...] = yp # reset to value at intersection point, i.e. now surface sag is taken into account
            u[i,...] = u_prime 
            y[i+1,...] = yp + np.tan(u_prime)*(lens_sequence.t[i] - zp)
            y_vertexplane[i+1,...] = y[i+1,...].copy()

    u[lens_sequence.num_surfs,...] = u[lens_sequence.num_surfs-1,...] # The image surface does not refract
    z_sag[lens_sequence.num_surfs,...] = 0 # Assumes that the image surface is flat.

    return y, u, z_sag, y_vertexplane