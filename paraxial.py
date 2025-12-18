# TODO:
# - visualize: plot paraxial focus and principal planes 
# - plot Gaussian intensity profile and beam waist as a function of z together with waist and divergence rays
# - Calculate position of nodal points if the index of refraction before and after the lens system is different.
# - make n_air adjustable
import numpy as np
from typing import Tuple

from lens import LensSequence
from config import Config

from dataclasses import dataclass
import matplotlib.pyplot as plt

__all__ = ["ParaxialRaytracer", "ParaxialQuantities"]

@dataclass(init=True, repr=False)
class ParaxialQuantities(object):
    """
    Paraxial quantities for finite conjugate system
    
    Given the object distance specified in the LensSequence, the back image 
    distance (BID) is calculated to make object and image plane conjugate. Based on object
    distance and BID the conjugate ABCD matrix is calculated. The original distance of the image 
    plane in the LensSequence is not altered.
    """
    # def __init__(self, EFL, BFL, FFL, V1H1, V2H2, 
    #              EPP, EPD, EP_is_virtual, marginal_ray_angle,
    #              XPP, XPD, XP_is_virtual,
    #              object_dist, image_dist,
    #              magnification, angular_magnification,
    #              stop_radius,
    #              BID, 
    #              y_chief, u_chief, y_marg, u_marg,
    #              ABCD_system, ABCD_conjugate
    #              ):
    #     self.EFL = EFL
    #     self.BFL = BFL
    #     self.FFL = FFL
    #     self.V1H1 = V1H1
    #     self.V2H2 = V2H2
    #     self.EPP = EPP
    #     self.EPD = EPD
    #     self.EP_is_virtual = EP_is_virtual
    #     self.marginal_ray_angle = marginal_ray_angle
    #     self.XPP = XPP
    #     self.XPD = XPD
    #     self.XP_is_virtual = XP_is_virtual
    #     self.object_dist = object_dist
    #     self.image_dist = image_dist
    #     self.magnification = magnification
    #     self.angular_magnification = angular_magnification
    #     self.stop_radius = stop_radius
    #     self.BID = BID
    #     self.y_chief = y_chief
    #     self.u_chief = u_chief
    #     self.y_marg = y_marg
    #     self.u_marg = u_marg
    #     self.ABCD_system = ABCD_system
    #     self.ABCD_conjugate = ABCD_conjugate
    EFL: float
    BFL: float
    FFL: float
    V1H1: float
    V2H2: float
    EPP: float
    EPD: float
    EP_is_virtual: bool
    marginal_ray_angle: float
    XPP: float
    XPD: float
    XP_is_virtual: bool
    object_dist: float
    image_dist: float
    magnification: float
    angular_magnification: float
    stop_radius: float
    BID: float
    y_chief: np.ndarray
    u_chief: np.ndarray
    y_marg: np.ndarray
    u_marg: np.ndarray
    ABCD_system: np.ndarray
    ABCD_conjugate: np.ndarray

    def __repr__(self):
        kws = [f"{key}={value}" for key, value in self.__dict__.items()]
        return "\nParaxial quantities: \n====================\n{}".format("\n".join(kws))
                  


class ParaxialRaytracer(object):
    """Paraxial (ynu) ray tracer using ABCD matrices
        (yk   )  = (A B)  (y0   )
        (nk*uk)    (C D)  (n0*u0)
    """
    version = "0.0.1"

    def __init__(self, LS: LensSequence) -> None:
        self.forward = LS.forward
        self.num_surfs = LS.num_surfs
        self.n = LS.n
        self.AS_surf = LS.AS_surf
        self.vertex = LS.vertex
        self.t = LS.t

        self.ABCD_matrices = []
        self.ABCD_front_group = [] # all lens elements before the aperture stop
        self.ABCD_rear_group = [] # all lens elements behind the aperture stop
        # exclude object (surf=0) and image surface (surf=num_surfs-1) )when constructing the system matrix
        for i in range(1,LS.num_surfs-1):
            if i < LS.num_surfs-2:
                # refracting surface followed by translation
                self.ABCD_matrices.extend((self._Rmat(LS.phi[i]), self._Tmat(LS.t[i], LS.n[i])))
            elif i == LS.num_surfs-2:
                # The last refracting surface is NOT followed by translation.
                self.ABCD_matrices.extend((self._Rmat(LS.phi[i]),))
            if i < self.AS_surf:
                self.ABCD_front_group.extend((self._Rmat(LS.phi[i]), self._Tmat(LS.t[i], LS.n[i])))
            # Note: The refraction through the aperture stop is omitted: It is contained neither in the front group nor the rear group.
            elif i > self.AS_surf:
                self.ABCD_rear_group.extend((self._Tmat(LS.t[i-1], LS.n[i-1]), self._Rmat(LS.phi[i])))
 
        self.system_matrix = self._make_system_matrix(forward=True)

        self.rear_group_matrix = np.eye(2)
        for m in self.ABCD_rear_group:
            self.rear_group_matrix = np.matmul(m, self.rear_group_matrix)

        # The front group (in front of the aperture stop) is traced in reverse order.
        self.front_group_matrix = np.eye(2)
        for m in self.ABCD_front_group[::-1]:
            self.front_group_matrix = np.matmul(np.linalg.inv(m), self.front_group_matrix)
 
    def _make_system_matrix(self, forward=True):        
        M = np.eye(2)
        if forward:
            for m in self.ABCD_matrices:
                M = np.matmul(m, M)
            self.forward = True
        else:
            for m in self.ABCD_matrices[::-1]:
                M = np.matmul(np.linalg.inv(m), M)
            self.forward = False
        return M
    
    def _make_conjugate_matrix(self, object_dist, image_dist):
        return self._Tmat(image_dist) @ self.system_matrix @ self._Tmat(-object_dist)
            
    def _Tmat(self, t, n=1.0):
        """Return ABCD translation matrix
        d - float:  distance to the right of the optical element"
        n - float:  index of refraction"""
        return np.array([[1.0, t/n],
                         [0.0, 1.0]], dtype=np.float64)
    def _Rmat(self, phi):
        """Return ABCD refraction matrix
        phi = 1/f - float: refractive power"""
        return np.array([[1.0,  0.0],
                         [-phi, 1.0]], dtype=np.float64)
    
    def _Mmat(self, R, n=1.0):
        """Return ABCD mirror matrix"""
        return self. _Rmat(phi=-2.0*n/R)

    def trace_ray_V1toV2(self, y0: float, u0: float) -> Tuple[float, float]:
        """ynu trace a ray through the lens system from the first to the last vertex.
        This excludes object and image distance."""
        ynu = self.system_matrix @ np.array([y0, u0*self.n[0]])
        y1 = ynu[0]
        u1 = ynu[1]/self.n[-2] # n[-1] would be the index of refraction *after* the image surface, which is not used.
        return y1, u1
    
    def trace_ray_paraxially_object_to_image(self, y0: float, u0: float, forward: bool=True) -> np.ndarray:
        ynu = np.zeros((self.num_surfs, 2))
        ynu[0,:] = np.array([y0, u0*self.n[0]])
        if forward:
            # 1. object plane to right before first vertex
            ynu_before_V1 = self._Tmat(self.t[0], self.n[0]) @ ynu[0,:]
            # 2. right before first vertex to right after last vertex
            ynu[1:self.num_surfs-1,:] = self.trace_ray_paraxially(ynu_before_V1[0], ynu_before_V1[1]/self.n[0], start_surf=1, stop_surf=self.num_surfs-2, forward=True)[1:self.num_surfs-1,:]
            # 3. right after last vertex to image plane
            ynu[self.num_surfs-1,:] = self._Tmat(self.t[self.num_surfs-2], self.n[self.num_surfs-2]) @ ynu[self.num_surfs-2,:]
        else:
            raise NotImplementedError
        return ynu

    def trace_ray_paraxially(self, y0: float, u0: float, start_surf: int, stop_surf: int, forward: bool=True, skip_start_surf=False) -> np.ndarray:
        """
        Trace a paraxial ray (y0, u0) from *right before* surface start_surf up to *right after* surface stop_surf
        and return all ynu values. Only surfaces of the lens system can be used as start and stop surfaces,
        thus excluding object and image surface.

        Important Note: 
            By convention, a ray ynu[s] at surface s refers to y and u right *after* the surface s.
            But trace_ray_paraxially() assumes that ynu at the start surface is specified *right before* the surface
            in forward direction (or *right after* the surface in reverse direction).
            Therefore, assuming a traced ray ynu[:], in order to reproduce its trajectory starting from surface s
            and stopping at s_stop in forwar direction, the call needs to be made like this:

                trace_ray_paraxially(ynu[s,0], ynu[s-1,1], s, s_stop, forward=True)

            because ynu[s-1,1] is the paraxial angle *right before* the surface s, i.e. before refracting through surface s.

            The same trajectory in reverse direction is calculated by calling

                trace_ray_paraxially(ynu[s_stop,0], ynu[s_stop,1], s_stop, s, forward=False).

        Parameters
        ----------
        y0 - float: ray height at the start surface
        u0 - float: paraxial angle at the start surface
        start_surf - int: surface right before (forward) or right after (reverse) which (y0,u0) is given
        stop_surf - int: last surface
        forward - bool: whether to trace in forward or reverse direction
        skip_start_surf - bool: Only relevant if forward==False. Then (y0,u0) refer to the ray right before 
                                start_surf rather right after start_surf. This is useful when tracing a ray 
                                backward from the aperture stop.

        Returns
        -------
        ynu[0:num_surfs] - np.ndarray: All ynu values with np.nan entries at surfaces outside [start_surf, stop_surf].
        """
        assert 1 <= start_surf < self.num_surfs-1, "Start and stop surfaces must not include object or image surface."
        if forward: assert stop_surf > start_surf
        if not forward: assert stop_surf < start_surf, f"stop_surf={stop_surf}, start_surf={start_surf}"
        if skip_start_surf: assert not forward

        ynu = np.zeros((self.num_surfs,2)) * np.nan # Nan values are not shown by plt.plot().

        if forward:
            ynu_tmp = np.array([y0, u0*self.n[start_surf-1]]) # This is the ray vector *right before* start_surf
            i_refrac = (start_surf-1)*2
            ynu[start_surf] = self.ABCD_matrices[i_refrac] @ ynu_tmp # refract through start_surf
            # For consistency with ray tracing in the reverse direction, calculate also at start_surf-1.
            if start_surf > 1:
                ynu[start_surf-1] = np.linalg.inv(self.ABCD_matrices[i_refrac-1]) @ ynu_tmp
            else:
                # start_surf == 1:
                # The translation from object to first vertex is not part of list self.ABCD_matrices.
                ynu[0,:] = np.linalg.inv(self._Tmat(self.t[0], self.n[0])) @ ynu_tmp
            for s in range(start_surf, stop_surf):
                i_refrac = (s-1)*2
                i_transl = i_refrac+1 
                ynu_ = self.ABCD_matrices[i_transl] @ ynu[s]
                ynu[s+1] = self.ABCD_matrices[i_transl+1] @ ynu_
        else:
            ynu[start_surf] = np.array([y0, u0*self.n[start_surf]])
            if not skip_start_surf:
                i_refrac = (start_surf-1)*2
                ynu_tmp = np.linalg.inv(self.ABCD_matrices[i_refrac]) @ ynu[start_surf]
            else:
                ynu_tmp = np.array([y0, u0*self.n[start_surf-1]])
            for s in range(start_surf-1, stop_surf-1, -1):
                i_transl = (s-1)*2+1
                # translation followed by refraction
                ynu[s] = np.linalg.inv(self.ABCD_matrices[i_transl]) @ ynu_tmp
                ynu_tmp = np.linalg.inv(self.ABCD_matrices[i_transl-1]) @ ynu[s]
            ynu[stop_surf,:] = ynu_tmp

        return ynu

    def _trace_ray_paraxially_front_group_to_object(self, y0: float, u0: float) -> Tuple[float, float]:
        """Trace a ray (y0,u0) from the aperture stop till the object plane."""
        # from aperture stop till right before first vertex
        ynu = np.matmul(self.front_group_matrix, np.array([y0, u0*self.n[self.AS_surf-1]]).T)
        # from first vertex till object plane
        ynu_object = np.matmul(np.linalg.inv(self._Tmat(self.t[0], self.n[0])), ynu)
        return ynu_object[0], ynu_object[1]/self.n[0]

    def _intersection_with_oA(self, y: float, u: float, z0: float) -> float:
        """For a ray (y,u) located at z-position z0, calculate its intersection with the optical axis."""
        z_int = z0 - y/u
        return z_int
    
    def _intersection_line_segments(self, y1: float, u1: float, y2: float, u2: float, z0: float) -> Tuple[float, float]:
        """
        Intersection point between ray (y1,u1) and (y2,u2) launched at z-position z0.

        Parameters
        ----------
        y1, y2 - float: ray heights
        u1, u2 - float: paraxial ray angles
        z0 - float: position on the optical axis 
        """
        z_int = z0 - (y1 -y2)/(u1 - u2)
        y_int = y1 + u1*(z_int-z0)
        return z_int, y_int

    def get_image_distance(self, object_dist):
        """The image distance as measured from the rightmost vertex V2 of the lens system."""
        assert object_dist < 0, "object distance should be a negative number"
        A = self.system_matrix[0,0]; B = self.system_matrix[0,1]
        C = self.system_matrix[1,0]; D = self.system_matrix[1,1]
        image_dist = (object_dist*A - B) / (D - object_dist*C)
        self.conjugate_matrix = self._make_conjugate_matrix(object_dist, image_dist)
        return image_dist
    
    def is_conjugate(self, N):
        """Check whether an ABCD matrix N defines conjugate planes."""
        return np.isclose(N[0,1], 0.0, atol=1e-8)
    
    def get_BID(self):
        """back image distance"""
        return self.get_image_distance(self.vertex[0])

    def get_magnification(self):
        if self.is_conjugate(self.conjugate_matrix):
            return self.conjugate_matrix[0,0]
        else:
            return None
    
    def get_angular_magnification(self):
        if self.is_conjugate(self.conjugate_matrix):
            return self.conjugate_matrix[1,1]
        else:
            return None
        
    def get_EFL(self):
        """get effective focal length"""
        return (-1.0/self.system_matrix[1,0])

    def get_BFL(self):
        """get back focal length"""
        n2 = self.n[-2] # image space refractive index. Note: self.n[-1] is the index of refraction after the image plane and is not used.
        return (-n2*self.system_matrix[0,0]/self.system_matrix[1,0])
    
    def get_FFL(self):
        """get front focal length"""
        n1 = self.n[0] # object space refractive index
        return (-n1*self.system_matrix[1,1]/self.system_matrix[1,0])
    
    def get_V1H1(self):
        """distance of the front principal plane P1 measured from the front vertex"""
        n1 = self.n[0] # object space refractive index
        return n1*(self.system_matrix[1,1]-1.0)/self.system_matrix[1,0]

    def get_V2H2(self):
        """distance of the back principal plane P2 measured from the back vertex (Note: V2H2 is negative if P2 is inside the lens.)"""
        n2 = self.n[-2] # image space refractive index. Note: self.n[-1] is the index of refraction after the image plane and is not used.
        return (1.0-n2*self.system_matrix[0,0])/self.system_matrix[1,0]

    def _get_entrance_and_exit_pupil(self, EPD: int) -> Tuple[float, float, float, float, float, float, bool, bool]:
        """
        Assuming that the stop surface and the entrance pupil diameter are specified by the user,
        determine the location of the entrance and exit pupil, the stop radius and the diameter
        of the exit pupil.

        The position of the entrance pupil is measured relative to the first lens vertex.
        The position of the exit pupil is measured relative to the *image surface*.
        """
        # The chief ray intersects the optical axis at the aperture stop and at all conjugate planes,
        # which are the pupil planes.
        y_chief = 0; u_chief = +0.4 # angle is arbitrary
        ynu1 = np.matmul(self.front_group_matrix, np.array([y_chief, u_chief*self.n[self.AS_surf-1]]).T)
        # The position of the entrance pupil is measured with respect to the front vertex.
        position_EP = self._intersection_with_oA(ynu1[0], ynu1[1]/self.n[0], self.vertex[1])
        if position_EP > self.vertex[0]:
            EP_is_virtual = True
        else:
            EP_is_virtual = False

        # Launch the marginal ray and determine its height at the aperture stop.
        marginal_ray_angle = np.arctan((EPD/2.0)/(position_EP - self.vertex[0]))
        ynu2 = np.matmul(np.linalg.inv(self.front_group_matrix), np.array([np.tan(marginal_ray_angle)*np.abs(self.vertex[0]), marginal_ray_angle*self.n[0]]).T)
        stop_radius = ynu2[0]

        # Trace a ray from the center of the aperture stop through the rear group of the lens system.
        # The z-position where the ray leaving the rear group intersects the optical axis is the position of the exit 
        # pupil.
        y_chief = 0.0; u_chief = -0.4 # angle is arbitrary
        ynu1 = np.matmul(self.rear_group_matrix, np.array([y_chief, u_chief*self.n[self.AS_surf]]).T)
        # position_XP is measured relative to the image surface (i.e. the last surface of the LensSequence)
        position_XP = self._intersection_with_oA(ynu1[0], ynu1[1]/self.n[-2], self.vertex[-2]) - self.vertex[-1]
        if position_XP > 0: # measured relative to the image surface
            XP_is_virtual = False
        else:
            XP_is_virtual = True

        y_marg = stop_radius; u_marg = 0.0 # ray from the edge of the aperture stop, its image at the exit pupil plane gives the radius of the exit pupil
        ynu2 = np.matmul(self.rear_group_matrix, np.array([y_marg, u_marg*self.n[self.AS_surf]]).T)
        diameter_XP = 2.0*(ynu2[0] + (ynu2[1]/(self.n[-2]))*(position_XP + (self.vertex[-1]-self.vertex[-2])))

        return position_EP, EPD, marginal_ray_angle, abs(stop_radius), position_XP, abs(diameter_XP), EP_is_virtual, XP_is_virtual


    def find_chief_ray(self, obj_height: float, u_max_start: float=np.pi/3, eps: float=1e-6) -> Tuple[np.ndarray, np.ndarray]:
        
        assert obj_height > 0
        u_min = 0.0
        u_max = u_max_start

        INTERSECTION_FOUND = False
        while (not INTERSECTION_FOUND):
            u_launch_AS = 0.5*(u_min + u_max) # launch angle at the aperture stop
            y, u = self._trace_ray_paraxially_front_group_to_object(0.0, -u_launch_AS)
            assert y > 0
            if y < obj_height:
                u_min = u_launch_AS
            else:
                u_max = u_launch_AS
            if np.isclose(y, obj_height, atol=eps):
                INTERSECTION_FOUND = True
                u_launch = u # launch angle at object
                print(f"launch angle found: y={y}, u_launch={u_launch}")

        # calculate chief ray trajectory from aperture stop to object plane
        # 1. from AS to right before first vertex
        ynu_reverse = self.trace_ray_paraxially(0.0, -u_launch_AS, start_surf=self.AS_surf, stop_surf=1, forward=False, skip_start_surf=True)
        # 2. from right before first vertex to object plane
        ynu_reverse[0,:] = np.linalg.inv(self._Tmat(self.t[0], self.n[0])) @ ynu_reverse[1,:]
        # return ynu_reverse

        # # trace a ray from the object to the image plane
        ynu_object = np.array([obj_height, u_launch*self.n[0]])
        # # 1. from object to first vertex 
        ynu1 = self._Tmat(self.t[0], self.n[0]) @ ynu_object
        # # 2. from right before first vertex till right after last vertex 
        ynu_chief = self.trace_ray_paraxially(ynu1[0], ynu1[1]/self.n[0], 1, self.num_surfs-2, forward=True)
        ynu_chief[0,:] = ynu_object # overwrite
        # 3. from right after last vertex till image plane
        ynu_image = self._Tmat(self.t[self.num_surfs-2], self.n[self.num_surfs-2]) @ ynu_chief[self.num_surfs-2,:]
        ynu_chief[self.num_surfs-1,:] = ynu_image # overwrite

        return ynu_chief[:,0], ynu_chief[:,1]/self.n[:]


    def marginal_ray_from_EPD(self, EPD: float) -> Tuple[np.ndarray, np.ndarray]:
        
        if not hasattr(self, "marginal_ray_angle"):
            self.EPP, self.EPD, self.marginal_ray_angle, self.stop_radius, self.XPP, self.XPD, self.EP_is_virtual, self.XP_is_virtual=self._get_entrance_and_exit_pupil(EPD)
        y0 = 0.0; u0 = self.marginal_ray_angle
        
        ynu = self.trace_ray_paraxially_object_to_image(y0, u0, forward=True)
        return ynu[:,0], ynu[:,1]/self.n[:] 


    def marginal_ray_from_stop_radius(self, stop_radius: float) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


    def paraxial_quantities(self, config: Config):
        self.EFL=self.get_EFL()
        self.BFL=self.get_BFL()
        self.FFL=self.get_FFL()
        self.V1H1=self.get_V1H1()
        self.V2H2=self.get_V2H2()
        self.EPP, self.EPD, self.marginal_ray_angle, self.stop_radius, self.XPP, self.XPD, self.EP_is_virtual, self.XP_is_virtual=self._get_entrance_and_exit_pupil(config.EPD)
        self.object_dist = self.vertex[0]
        self.image_dist = self.get_image_distance(self.object_dist)
        self.magnification=self.get_magnification()
        self.angular_magnification=self.get_angular_magnification()
        self.BID=self.get_BID()
        self.y_chief, self.u_chief = self.find_chief_ray(config.max_obj_height)
        self.y_marg, self.u_marg = self.marginal_ray_from_EPD(self.EPD)

        return ParaxialQuantities(
            EFL=self.EFL, BFL=self.BFL, FFL=self.FFL, V1H1=self.V1H1, V2H2=self.V2H2,
            EPP=self.EPP, EPD=self.EPD, EP_is_virtual=self.EP_is_virtual, marginal_ray_angle=self.marginal_ray_angle,
            XPP=self.XPP, XPD=self.XPD, XP_is_virtual=self.XP_is_virtual,
            object_dist=self.object_dist,
            image_dist=self.image_dist,
            magnification=self.magnification, angular_magnification=self.angular_magnification,
            stop_radius=self.stop_radius,
            BID=self.BID,
            y_chief=self.y_chief,
            u_chief=self.u_chief,
            y_marg=self.y_marg,
            u_marg=self.u_marg,
            ABCD_system=self.system_matrix,
            ABCD_conjugate=self.conjugate_matrix
        )


class GaussianRaytracer(ParaxialRaytracer):
    """Propagate Gaussian beams specified by a complex parameter q using paraxial ray tracing."""
    def __init__(self, lens_sequence: LensSequence) -> None:
        super().__init__(lens_sequence)

    def propagate_Gaussian_beam(self, q0):
        assert np.iscomplex(q0), "Gaussian beam parameter must be a complex number"
        A = self.system_matrix[0,0]; B = self.system_matrix[0,1]
        C = self.system_matrix[1,0]; D = self.system_matrix[1,1]
        q = (A*q0 + B) / (C*q0 + D)
        return q
    