import numpy as np
from typing import Iterable, Tuple
from utils import timer_func, RayIntersectionNotFoundError
import dataclasses
from lens import LensSequence

__all__ = ["trace_tangential_ray"]

@timer_func
def trace_tangential_ray(y_start: Iterable, u_start: Iterable, lens_sequence: LensSequence, surf_start: int=0, 
                         forward: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Trace a batch of rays forward (left-to-right) or backwards (right-to-left) till right after the last or 
    right before the first surface of the lens system. Surface sag is considered. 

    When ray tracing from right to left:
       (A) Indices of refraction (before, after) are exchanged.
       (B) The radius of curvature is inverted and all quantities are computed as if for that problem.
       (C) The surface sag thus computed needs to be inverted.

    Parameters
    ----------
    y_start : array_like - ray height at the object surface (if surf_start == 0) or any other surface
    u_start : array_like - ray angle relative to the horizontal at the object surface (if surf_start == 0) or any other surface
    lens_sequence: LensSequence object
    surf_start : int, default=0
    forward : bool, default=True

    Returns
    -------
    y
    u
    z_sag
    y_vertexplane
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

    def intersection_spherical_with_sag(surf: float, y0: np.ndarray, u0: np.ndarray, 
                                        lens_sequence: LensSequence, forward: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """intersection with a spherical surface with positive or negative radius of curvature"""
        assert surf > 0
        assert lens_sequence.forward == forward
        sgnR = np.sign(lens_sequence.R[surf]) # The formula depends on the sign of the radius of curvature.
        tanu0 = np.tan(u0)
        Delta = lens_sequence.R[surf]**2 - 2*y0*tanu0*lens_sequence.R[surf] - y0**2
        if not np.all(Delta > 0):
            raise RayIntersectionNotFoundError("Delta < 0") #, "Delta[0] < 0, %f"%(Delta[0])  
        zp = (lens_sequence.R[surf] - y0*tanu0 - sgnR*np.sqrt(Delta))/(1 + tanu0**2)                     
        yp = y0 + tanu0*zp
        
        if (np.abs(yp) > lens_sequence.y_nnint[surf]).any():
            print("yp=", yp, "y_nnint=", lens_sequence.y_nnint[surf], "z_nnint=", lens_sequence.z_nnint[surf], "surf=", surf)
            pass
            # raise RayIntersectionNotFoundError(f"Ray intersection point (zp,yp)=({zp},{yp}) outside maximal clear" 
            #                                    f"aperture of the surface nr {surf}")

        theta = np.arctan(sgnR*yp/(lens_sequence.R[surf]-zp))
        if forward:
            n_ratio = lens_sequence.n[surf-1]/lens_sequence.n[surf]
        else: # modification (A)
            n_ratio = lens_sequence.n[surf]/lens_sequence.n[surf-1]
        u_prime = sgnR*(np.arcsin(n_ratio*np.sin(theta + sgnR*u0)) - theta)

        return zp, yp, u_prime


    if forward == True:

        # The ray height `y_start`` is meant to be measured at the vertex plane tangential to the surface with index `surf_start`.
        # This means that surface sag is not taken into account yet.
        y_vertexplane[surf_start,...] = y_start[...].copy()
        y[surf_start,...] = y_start[...].copy()
        u[surf_start,...] = u_start[...].copy()

        for i in range(surf_start, lens_sequence.num_surfs):
            if np.isinf(lens_sequence.R[i]) or not lens_sequence.SAG:
                if i > surf_start:
                    u[i,...] = np.arctan((lens_sequence.n[i-1]/lens_sequence.n[i])*np.tan(u[i-1,...]) - lens_sequence.phi[i]*y[i,...]/lens_sequence.n[i])
                y[i+1,...] = y[i,...] + np.tan(u[i,...])*lens_sequence.t[i]   

            # take surface sag into account
            else:
                if i == 0:
                    y[i+1,...] = y[i,...] + np.tan(u[i,...])*lens_sequence.t[i]   
                    continue
                y0 = y[i,...]
                if i == surf_start:            
                    u0 = u[i,...]
                else:
                    u0 = u[i-1,...]

                zp, yp, u_prime = intersection_spherical_with_sag(i, y0, u0, lens_sequence, forward=True)

                z_sag[i,...] = zp
                y[i,...] = yp
                if i > surf_start:
                    u[i,...] = u_prime 
                y[i+1,...] = yp + np.tan(u[i,...])*(lens_sequence.t[i] - zp)
                y_vertexplane[i+1,...] = y[i+1,...].copy()

        u[lens_sequence.num_surfs,...] = u[lens_sequence.num_surfs-1,...] # The image surface does not refract
        z_sag[lens_sequence.num_surfs,...] = 0 # Assumes that the image surface is flat.

    else: # raytrace backward

        assert surf_start > 0
        lens_sequence_bw = dataclasses.replace(lens_sequence, forward=False, R=-lens_sequence.R[:])  # modification (B)

        u[surf_start-1,...] = u_start[...].copy()
        y[surf_start,...] = y_start[...].copy()

        y[surf_start-1,...] = y[surf_start,...] + np.tan(u[surf_start-1,...])*lens_sequence.t[surf_start-1]

        for i in range(surf_start-1, 0, -1):
            if np.isinf(lens_sequence_bw.R[i]) or not lens_sequence.SAG:
                z_sag[i,...] = 0.0
                u[i-1,...] = np.arctan((lens_sequence.n[i]/lens_sequence.n[i-1])*np.tan(u[i,...]) - lens_sequence.phi[i]*y[i,...]/lens_sequence.n[i-1])
                y[i-1,...] = y[i,...] + np.tan(u[i-1,...])*lens_sequence.t[i-1]
            else:
                y0 = y[i,...]
                u0 = u[i,...]
                zp, yp, u_prime = intersection_spherical_with_sag(i, y0, u0, lens_sequence_bw, forward=False)
                z_sag[i,...] = (-1)*zp # modification (C)
                y[i,...] = yp # reset to value at intersection point
                u[i-1,...] = u_prime
                y[i-1,...] = yp + np.tan(u[i-1,...])*(lens_sequence.t[i-1] - zp)

    return y, u, z_sag, y_vertexplane