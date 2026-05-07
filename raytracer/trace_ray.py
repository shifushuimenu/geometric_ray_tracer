"Raytracing of tangential and non-meridional rays, determination of chief ray using binary search"
import numpy as np
from math import cos, sin
from typing import Iterable, Tuple

from raytracer.utils import timer_func, RayIntersectionNotFoundError, ChiefRayNotFoundError
from raytracer.lens import LensSequence
from raytracer.config import Config

__all__ = ["MeridionalRayData", "NonmeridionalRayData", "RayTracer",
           "trace_tangential_ray", "trace_nonmeridional_rays", "find_chief_rays"]

class MeridionalRayData(object):
    def __init__(self, num_surfs: int, num_rays: int, num_fields: int, y: np.ndarray, 
                 u: np.ndarray, z_sag: np.ndarray, vertex: np.ndarray, clear_apertures: np.ndarray): 
        self.num_surfs = num_surfs
        self.num_rays = num_rays
        self.num_fields = num_fields
        self.y = y  # ray heights, y[0:num_surfs,0:num_rays,0:num_fields]
        self.u = u  # ray angles,  u[0:num_surfs,0:num_rays,0:num_fields]
        self.z_sag = z_sag # surface sag, z_sag[0:num_surfs,0:num_rays,0:num_fields]
        self.vertex = vertex # z-position of the surface vertices
        self.clear_apertures = clear_apertures # the heights of the outermost rays, clear_apertures[0:num_surfs] 
        self.CHIEF_RAY_INDEX = self.num_rays//2

class NonmeridionalRayData(object):
    def __init__(self, num_surfs: int, num_rays: int, num_fields: int, P_intersect, rayvecs):
        self.num_surfs = num_surfs
        self.num_rays = num_rays
        self.num_fields = num_fields
        self.P_intersect = P_intersect
        self.rayvecs = rayvecs
        self.CHIEF_RAY_INDEX = 0

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
        for s in range(0, num_surfs):
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
    
    def calculate_marginal_ray(self, lens_sequence: LensSequence, config: Config) -> np.ndarray:
        """if one only needs the marginal ray ..."""
        y_chief, u_chief, z_sag_chief = self.find_chief_rays(config.obj_heights)
        # EPL: non-paraxial entrance pupil location
        EPL = config.obj_heights[0]/np.tan(u_chief[0,config.MAX_OBJ_HEIGHT_INDEX]) - lens_sequence.t[0]
        marginal_ray_angle = np.arctan((config.EPD/2.0)/(EPL+lens_sequence.t[0]))
        return self.trace_tangential_ray([0.0], [marginal_ray_angle])


    @timer_func
    def calculate_meridional_ray_data(self, lens_sequence: LensSequence, config: Config) -> MeridionalRayData:

        self.lens_sequence = lens_sequence
        # 1. Determine the chief rays for all object heights.
        y_chief, u_chief, z_sag_chief = self.find_chief_rays(config.obj_heights)

        # IMPROVE
        print("u_chief=", u_chief[0,:])
        # Chief ray launch angle is needed when calculatin nonmeridional ray data.
        self.u_chief = u_chief
        # IMPROVE

        # entrance pupil location (measured from the vertex of the first surface)
        self.EPL = config.obj_heights[0]/np.tan(u_chief[0,config.MAX_OBJ_HEIGHT_INDEX]) - lens_sequence.t[0]
        self.marginal_ray_angle = np.arctan((config.EPD/2.0)/(self.EPL+lens_sequence.t[0]))
        self.ObjNA = lens_sequence.n[0]*np.sin(self.marginal_ray_angle)
        self.FOV = np.arctan((config.obj_heights[config.MAX_OBJ_HEIGHT_INDEX]-y_chief[1,config.MAX_OBJ_HEIGHT_INDEX])/lens_sequence.t[0])

        # 2. Trace all ray bundles and find the clear apertures.
        y, u, z_sag, y_vertexplane = self.trace_ray_bundles(config.obj_heights, u_chief[0,:], self.marginal_ray_angle, config.num_rays)

        clear_apertures, self.stop_radius = self.calculate_clear_apertures(y, config)

        return  MeridionalRayData(num_surfs = y.shape[0], num_rays = y.shape[1], num_fields = y.shape[2], y=y, u=u, z_sag=z_sag,
                        vertex = lens_sequence.vertex, clear_apertures = clear_apertures)


    def calculate_nonmeridional_ray_data(self, lens_sequence: LensSequence, config: Config) -> NonmeridionalRayData:
        self.lens_sequence = lens_sequence

        def _init_rayvecs_at_object(inclination_angles: Iterable[float], half_opening_angles: Iterable[float], 
                                    num_concentrics: int=5, num_rays_per_1st_concentric: int=7):
            """
            Parameters
            ----------
            inclination_angles[0:num_fields]
            half_opening_angles[0:num_fields]
            num_concentrics
            num_rays_per_1st_concentric

            Returns
            -------
            P_intersect[0:3,0:num_surfs,0:num_nonmeridional_rays,0:num_fields] intersection points with each spherical surface,
                with only the first surface initialized and intersection points for all other surfaces initialized to zero
            rayvecs[0:3,0:num_surfs,0:num_nonmeridional_rays,0:num_fields] unit vector indicating the direction of propagation after the 
                surface
            """
            # ====================================
            # Launch a non-meridional ray fan
            # ====================================
            # The rotation angle gamma1 is around the x-axis (positive angle means downward inclination)
            # and rotation angle gamma2 around the y-axis (positive angle means left-turning when looking in the direction of the ray).
            gamma1_field = inclination_angles  # inclination angle (in radians) of the ray bundle. IMPROVE: gamma1_field depends on the launch angle of the chief ray for each field position
            gamma2_max = half_opening_angles   # half opening angle of the ray bundle 
            dg2 = gamma2_max/num_concentrics   # angular increment

            # For every field position the number of non-meridional rays is the same.
            num_rays_per_concentric = np.zeros(num_concentrics, dtype=int)
            dg3 = np.zeros(num_concentrics)                
            for c in range(num_concentrics):
                if c==0:
                    dg3[c] = 2*np.pi/num_rays_per_1st_concentric # angular increment in the azimuthal angle   
                    num_rays_per_concentric[c] = num_rays_per_1st_concentric
                else:
                    dg3[c] = dg3[0] / (c+1)
                    num_rays_per_concentric[c] = (c+1) * num_rays_per_1st_concentric
            num_nonmeridional_rays = 1 + np.sum(num_rays_per_concentric[:]) # includes chief ray at the center of the ray bundle

            # A ray bundle at field position f is a tuple (P_intersect[0:3,0:num_surfs,0:num_rays,f], rayvecs[0:3,0:num_surfs,0:num_rays,f]). 
            # The chief ray is is labelled as the first ray: (P_intersect[0:3,0:num_surfs,0,f], rayvecs[0:3,0:num_surfs,0,f])
            P_intersect = np.zeros((3,lens_sequence.num_surfs, num_nonmeridional_rays, config.num_fields))
            rayvecs = np.zeros((3,lens_sequence.num_surfs, num_nonmeridional_rays, config.num_fields))

            for f in range(config.num_fields):                    
                r = 0 # chief ray
                P_intersect[0:3, 0, r, f] = np.array([0, config.obj_heights[f], lens_sequence.vertex[0]]) # object oriented along y-axis
                rayvecs[0:3, 0, r, f] = np.array([0, -sin(gamma1_field[f]), cos(gamma1_field[f])])
                r = 1
                for c in range(num_concentrics):
                    gamma2 = (c+1) * dg2[f]
                    for azi in range(num_rays_per_concentric[c]):
                        gamma3 = dg3[c] * azi          
                        P_intersect[0:3, 0, r, f] = np.array([0, config.obj_heights[f], lens_sequence.vertex[0]]) # object oriented along y-axis
                        rayvecs[0:3, 0, r, f] = np.array([-sin(gamma2)*cos(gamma3), 
                                                         -cos(gamma1_field[f])*sin(gamma2)*sin(gamma3) - sin(gamma1_field[f])*cos(gamma2), 
                                                         -sin(gamma1_field[f])*sin(gamma2)*sin(gamma3) + cos(gamma1_field[f])*cos(gamma2)])
                        r += 1                         
                assert r == num_nonmeridional_rays
            
            return P_intersect, rayvecs
        
        u_launch = self.u_chief[0,:]
        half_opening_angles = np.ones_like(u_launch) * self.marginal_ray_angle
        P_intersect, rayvecs = _init_rayvecs_at_object(u_launch, half_opening_angles)
        P_intersect, rayvecs = trace_nonmeridional_rays(lens_sequence.vertex, lens_sequence.R, lens_sequence.n, P_intersect, rayvecs)        

        return NonmeridionalRayData(num_surfs=rayvecs.shape[1], num_rays=rayvecs.shape[2], num_fields=rayvecs.shape[3], 
                                    P_intersect=P_intersect, rayvecs=rayvecs)


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


@timer_func
def trace_nonmeridional_rays(vertex, R, n, P_intersect, rayvecs):
    """    
    Trace rays that do not lie in the tangential plane spanned by the object height and the optical axis (= non-meridional rays).

    :param vertex: Description
    :param R: Description
    :param n: Description
    :param P_intersect: Description
    :param rayvecs: Description
    """
    assert R.shape == n.shape
    num_surfs = n.shape[0]
    assert vertex.shape == (num_surfs,)
    assert P_intersect.shape == rayvecs.shape
    # num_rays = P_intersect.shape[2]
    # num_fields = P_intersect.shape[3] 
    # in the future: num_wavelengths = p_intersect.shape[4]
    
    dims = len(P_intersect.shape)-1

    # P_intersect = np.transpose(P_intersect, axes=(0,2,3,1))
    # rayvecs = np.transpose(rayvecs, axes=(0,2,3,1))

    for i in range(1, num_surfs):
        # Calculate the intersection point with surface i
        xO, yO, zO = P_intersect[0:3,i-1,...]    
        xr, yr, zr = rayvecs[0:3,i-1,...]

        if np.abs(R[i]) != np.inf:
            # spherical surface 
            b = 2*xO*xr + 2*yO*yr + 2*(zO - (vertex[i]+R[i]))*zr
            c = xO**2 + yO**2 + (zO-vertex[i])**2 - 2*(zO - vertex[i])*R[i]
            if R[i] > 0:
                alpha = 0.5*(-b - np.sqrt(b**2 - 4*c))
            elif R[i] < 0:
                alpha = 0.5*(-b + np.sqrt(b**2 - 4*c))
        else:
            # flat surface 
            alpha = (vertex[i] - zO)/zr # zr is never zero by construction

        if i==1:
            print("vertex[1]=", vertex[i])
            print("vertex[0]=", vertex[0])
            print("zO=", zO[0,:])
            print("zr=", zr[0,:])
            print("alpha=", alpha[0,:])
        P_intersect[0:3,i,...] = P_intersect[0:3,i-1,...] + alpha*rayvecs[0:3,i-1,...]

        # Calculate the new normalized ray vector using Snell's law
        if np.abs(R[i]) != np.inf:
            N_pt = P_intersect[0:3,i,...] - np.array([0,0,vertex[i] + R[i]]).reshape((3,)+(1,)*(dims-1))
            surface_norm_pt = N_pt / np.linalg.norm(N_pt, axis=0)
            einsum_str = "ijklmno"[0:dims]+","+"ijklmno"[0:dims]+"->"+"ijklmno"[1:dims]
            rn = np.einsum(einsum_str, rayvecs[0:3,i-1,...], surface_norm_pt[0:3,...]) # dot product over first axis
            n_ratio = n[i-1]/n[i]
            pref = -( np.sign(R[i])*np.sqrt(1-(n_ratio)**2 * (1 - rn**2)) + n_ratio*rn )
            rayvecs[0:3,i,...] = pref*surface_norm_pt + n_ratio*rayvecs[0:3,i-1,...]    
            assert np.isclose(np.linalg.norm(rayvecs[0:3,i,...], axis=0), 1.0, atol=1e-8).all(), f"ray vector not normalized"
        else:
            rayvecs[0:3,i,...] = rayvecs[0:3,i-1,...]

    # undo reordering of axes
    # P_intersect = np.transpose(P_intersect, axes=(0,3,1,2))
    # rayvecs = np.transpose(rayvecs, axes=(0,3,1,2))

    return P_intersect, rayvecs


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
        # IMPROVE
        u_cr[lens_sequence.AS_surf, f] = u_middle
        y_cr[lens_sequence.AS_surf, f] = 0.0 
        z_sag_cr[lens_sequence.AS_surf, f] = 0.0

    return y_cr, u_cr, z_sag_cr
