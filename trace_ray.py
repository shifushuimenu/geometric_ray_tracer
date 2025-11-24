import numpy as np
from typing import Iterable, Tuple
from utils import timer_func, RayIntersectionNotFoundError
from lens import LensSequence
from config import Config
from utils import ChiefRayNotFoundError

# __all__ = ["trace_tangential_ray"]


class MeridionalRayData(object):
    def __init__(self, num_surfs: int, num_rays: int, num_fields: int, y: np.ndarray, 
                 u: np.ndarray, z_sag: np.ndarray, vertex: np.ndarray, clear_apertures: np.ndarray): 
        self.num_surfs = num_surfs
        self.num_rays = num_rays
        self.num_fields = num_fields
        self.y = y  # ray heights, y[0:num_surfs]
        self.u = u  # ray angles,  u[0:num_surfs]
        self.z_sag = z_sag # surface sag, z_sag[0:num_surfs]
        self.vertex = vertex # z-position of the surface vertices
        self.clear_apertures = clear_apertures # the heights of the outermost rays, clear_apertures[0:num_surfs] 
        self.CHIEF_RAY_INDEX = self.num_rays//2


class RayTracer(object):
    def __init__(self, lens_sequence: LensSequence):
        self.lens_sequence = lens_sequence

    def trace_tangential_ray(self, y_start: Iterable, u_start: Iterable, surf_start: int=0, 
                             forward: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return trace_tangential_ray(y_start, u_start, self.lens_sequence, surf_start, forward)

    def find_chief_rays(self, obj_heights: Iterable) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return find_chief_rays(self.lens_sequence, obj_heights, u_max_start = np.pi/3, eps=1e-6)
    
    def calculate_clear_apertures(self, y: np.ndarray, config: Config) -> Tuple[np.ndarray, float]:
        """The heights of the outermost rays at each surface determine its clear aperture radius."""
        num_surfs = self.lens_sequence.num_surfs
        num_fields = config.num_fields
        num_rays = config.num_rays
        assert y.shape == (num_surfs, num_rays, num_fields)

        heights = np.zeros(num_surfs)
        heights[0] = 0
        for s in range(1, num_surfs):
            for f in range(num_fields):
                for r in [0,num_rays-1]: # consider only outermost rays
                    hs = np.abs(y[s,r,f])
                    if (hs > heights[s]): 
                        heights[s] = hs

        # Height of the marginal ray at the stop paraxial surface determines the stop_radius. 
        stop_radius = np.abs(y[self.lens_sequence.AS_surf, num_rays-1, config.ON_AXIS_FIELD_INDEX])

        return heights, stop_radius

    def trace_ray_bundles(self, obj_heights, u_cr0, marginal_ray_angle, num_rays=7):
        """
        Trace "fields" of height [obj_hgt, obj_hgt / sqrt(2), 0]
        with a cone of rays around each chief ray launch angle. For half the opening angle of the 
        cone of rays we choose the marginal ray angle.
        """
        assert num_rays % 2 == 1 # number of rays in a ray bundle for a given field
        num_fields = len(obj_heights)
        assert u_cr0.shape == (num_fields,)
        y_obj = np.zeros((num_rays, num_fields))
        u_obj = np.zeros((num_rays, num_fields))

        # Determine the launch angles in the object plane, centered around the chief ray
        for f in range(num_fields):
            y_obj[:,f] = obj_heights[f]
            dtheta = 2*marginal_ray_angle/num_rays
            # k=0 and k=num_rays-1 are the marginal rays, k=num_rays//2 is the chief ray of the ray bundle.
            u_obj[:,f] = np.array([-u_cr0[f] + (k-num_rays//2)*dtheta for k in range(num_rays)])

        y, u, z_sag, y_vertexplane = trace_tangential_ray(y_obj[:,:], u_obj[:,:], self.lens_sequence, surf_start=0)
        # Dimensions: y[0:num_surfs, 0:num_rays, 0:num_fields]
        return y, u, z_sag, y_vertexplane

    @timer_func
    def calculate_meridional_ray_data(self, lens_sequence: LensSequence, config: Config) -> MeridionalRayData:

        self.lens_sequence = lens_sequence
        # 1. Determine the chief rays for all object heights.
        y_cr, u_cr, z_sag_cr = self.find_chief_rays(config.obj_heights)

        # entrance pupil location (measured from the vertex of the first surface)
        self.EPL = config.obj_heights[0]/np.tan(u_cr[0,config.MAX_OBJ_HEIGHT_INDEX]) - lens_sequence.t[0]
        self.marginal_ray_angle = np.arctan((config.EPD/2.0)/(self.EPL+lens_sequence.t[0]))
        self.ObjNA = lens_sequence.n[0]*np.sin(self.marginal_ray_angle)
        self.FOV = np.arctan((config.obj_heights[config.MAX_OBJ_HEIGHT_INDEX]-y_cr[1,config.MAX_OBJ_HEIGHT_INDEX])/lens_sequence.t[0])

        # 2. Trace all ray bundles and find the clear apertures.
        y, u, z_sag, y_vertexplane = self.trace_ray_bundles(config.obj_heights, u_cr[0,:], self.marginal_ray_angle, config.num_rays)

        clear_apertures, self.stop_radius = self.calculate_clear_apertures(y, config)

        return  MeridionalRayData(num_surfs = y.shape[0], num_rays = y.shape[1], num_fields = y.shape[2], y=y, u=u, z_sag=z_sag,
                        vertex = lens_sequence.vertex, clear_apertures = clear_apertures)



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

    y = np.zeros((lens_sequence.num_surfs,)+batch_dim) # ray height at the curved surface
    y_vertexplane = np.zeros_like(y)  # ray height at the vertex plane
    u = np.zeros_like(y)
    z_sag = np.zeros_like(y)

    def intersection_spherical_with_sag(surf: float, y0: np.ndarray, u0: np.ndarray, 
                                        R: np.ndarray, n: np.ndarray, y_maxCA, 
                                        forward: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """intersection with a spherical surface with positive or negative radius of curvature"""
        assert surf > 0
        sgnR = np.sign(R[surf]) # The formula depends on the sign of the radius of curvature.
        tanu0 = np.tan(u0)
        Delta = R[surf]**2 - 2*y0*tanu0*R[surf] - y0**2
        if not np.all(Delta > 0):
            raise RayIntersectionNotFoundError("Delta < 0") #, "Delta[0] < 0, %f"%(Delta[0])  
        zp = (R[surf] - y0*tanu0 - sgnR*np.sqrt(Delta))/(1 + tanu0**2)                     
        yp = y0 + tanu0*zp
        
        if (np.abs(yp) > y_maxCA[surf]).any():
            print("yp=", yp)
            raise RayIntersectionNotFoundError(f"Ray intersection point yp={np.max(np.abs(yp))} outside maximal clear " 
                                               f"aperture {y_maxCA[surf]} of the surface nr {surf}")

        theta = np.arctan(sgnR*yp/(R[surf]-zp))
        if forward:
            n_ratio = n[surf-1]/n[surf]
        else: # modification (A)
            n_ratio = n[surf]/n[surf-1]
        u_prime = sgnR*(np.arcsin(n_ratio*np.sin(theta + sgnR*u0)) - theta)

        return zp, yp, u_prime
    

    if forward == True:

        # The ray height `y_start`` is meant to be measured at the vertex plane tangential to the surface with index `surf_start`.
        # This means that surface sag is not taken into account yet.
        y_vertexplane[surf_start,...] = y_start[...].copy()
        y[surf_start,...] = y_start[...].copy()
        u[surf_start,...] = u_start[...].copy()

        for i in range(surf_start, lens_sequence.num_surfs-1):
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

                zp, yp, u_prime = intersection_spherical_with_sag(i, y0, u0, lens_sequence.R, 
                                                                  lens_sequence.n, lens_sequence.y_nnint, forward=True)

                z_sag[i,...] = zp
                y[i,...] = yp
                if i > surf_start:
                    u[i,...] = u_prime 
                y[i+1,...] = yp + np.tan(u[i,...])*(lens_sequence.t[i] - zp)
                y_vertexplane[i+1,...] = y[i+1,...].copy()

        u[lens_sequence.num_surfs-1,...] = u[lens_sequence.num_surfs-2,...] # The image surface does not refract
        z_sag[lens_sequence.num_surfs-1,...] = 0 # Assumes that the image surface is flat.

    else: # raytrace backward

        assert surf_start > 0
        R=-lens_sequence.R[:]  # modification (B)

        u[surf_start-1,...] = u_start[...].copy()
        y[surf_start,...] = y_start[...].copy()

        y[surf_start-1,...] = y[surf_start,...] + np.tan(u[surf_start-1,...])*lens_sequence.t[surf_start-1]

        for i in range(surf_start-1, 0, -1):
            if np.isinf(R[i]) or not lens_sequence.SAG:
                z_sag[i,...] = 0.0
                u[i-1,...] = np.arctan((lens_sequence.n[i]/lens_sequence.n[i-1])*np.tan(u[i,...]) - lens_sequence.phi[i]*y[i,...]/lens_sequence.n[i-1])
                y[i-1,...] = y[i,...] + np.tan(u[i-1,...])*lens_sequence.t[i-1]
            else:
                y0 = y[i,...]
                u0 = u[i,...]
                zp, yp, u_prime = intersection_spherical_with_sag(i, y0, u0, R, lens_sequence.n, lens_sequence.y_nnint, forward=False)
                z_sag[i,...] = (-1)*zp # modification (C)
                y[i,...] = yp # reset to value at intersection point
                u[i-1,...] = u_prime
                y[i-1,...] = yp + np.tan(u[i-1,...])*(lens_sequence.t[i-1] - zp)

    return y, u, z_sag, y_vertexplane


def find_chief_rays(lens_sequence: LensSequence, obj_heights: Iterable, 
                    u_max_start: float=np.pi/3, eps: float=1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For all field positions given in `obj_height[0:num_fields]`, find the chief ray launch 
    angles at the aperture stop. The launch angles are modified using a binary search 
    and the ray is traced backwards from the center of the aperture stop until it hits the object position within `eps` tolerance
    (in lens units).

    IMPORTANT Note: It is assumed that increasing the chief ray launch angle in positive direction will monotonically 
    increase the ray height in the object plane. This is not always the case, especially for non-physical surfaces,
    i.e. lens elements outside their clear apertures where front and back surface are inverted. 
    """
    obj_heights = np.asarray(obj_heights) #
    assert (obj_heights >= 0).all()      #
    num_fields = len(obj_heights)
    y_cr = np.zeros((lens_sequence.num_surfs, num_fields))
    u_cr = np.zeros((lens_sequence.num_surfs, num_fields))
    z_sag_cr = np.zeros((lens_sequence.num_surfs, num_fields))

    # Same maximum chief ray launch angle for all field positions.
    u_min_start = -u_max_start 

    for f in range(num_fields):
        print("field=", f)  
        if obj_heights[f] == 0.0:
            # on-axis field point: y_cr and u_cr are zero
            continue
        
        u_min = u_min_start
        u_max = u_max_start

        y_cr[lens_sequence.AS_surf,f] = 0.0

        # Choose the initial cone of angles 
        INTERSECTION_FOUND = False
        while(not INTERSECTION_FOUND):
            try:
                y, u, z_sag, _ = trace_tangential_ray(0.0, u_max, lens_sequence, lens_sequence.AS_surf, forward=False)
            except RayIntersectionNotFoundError:
                # make the opening angle of the cone within which the chief ray launch angle is searched smaller
                u_max -= 0.1*u_max
                u_min -= 0.1*u_min
                print(f"RayIntersectionNotFoundError u_min={u_min}, u_max={u_max}")
                continue
            INTERSECTION_FOUND = True
            print("INTERSECTION FOUND")

        if y[0] < obj_heights[f]: 
            print("y[0]=", y[0])
            raise ChiefRayNotFoundError(f"Object height {obj_heights[f]} is too large, it cannot be reached "
                             f"from the aperture stop with any chief ray launch angle. Consider adjusting u_max.")

        CHIEF_RAY_FOUND = False
        print("determining chief ray launch angle")        
        while(not CHIEF_RAY_FOUND):
            # Update launch angle using bisection search.
            # It is assumed that increasing the chief ray launch angle will 
            # monotonically increase its height in object space.
            u_middle = 0.5*(u_max + u_min)                  
            # By definition, at the aperture stop, the chief ray intersects the optical axis.
            y, u, z_sag, _ = trace_tangential_ray(0.0, u_middle, lens_sequence, lens_sequence.AS_surf, forward=False)
            # criterion whether chief ray has been found
            CHIEF_RAY_FOUND = np.isclose(y[0], obj_heights[f], atol=eps)
            # update bracketing interval for binary search            
            if y[0] < obj_heights[f]:
                print(f"u_min = {u_min} and u_max = {u_max}")
                print(f"u_middle = {u_middle}")                          
                print(f"y_cr[0,{f}] < obj_height[{f}]")
                u_min = u_middle 
            else:
                print(f"u_min = {u_min} and u_max = {u_max}")
                print(f"u_middle = {u_middle}")
                print(f"y_cr[0,{f}] > obj_height[{f}]")
                u_max = u_middle
        y_cr[0:lens_sequence.AS_surf+1, f] = y[0:lens_sequence.AS_surf+1]
        u_cr[0:lens_sequence.AS_surf+1, f] = u[0:lens_sequence.AS_surf+1]
        z_sag_cr[0:lens_sequence.AS_surf+1, f] = z_sag[0:lens_sequence.AS_surf+1]

    return y_cr, u_cr, z_sag_cr
