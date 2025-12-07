# TODO:
# - visualize: plot paraxial focus and principal planes 
# - plot Gaussian intensity profile and beam waist as a function of z together with waist and divergence rays
# - Calculate position of nodal points if the index of refraction before and after the lens system is different.
# - make n_air adjustable
import numpy as np
from typing import Tuple

from lens import LensSequence
from config import Config

import matplotlib.pyplot as plt

__all__ = ["ParaxialRaytracer", "ParaxialQuantities"]


class ParaxialQuantities(object):
    """
    Paraxial quantities for finite conjugate system
    
    Given the object distance specified in the LensSequence, the back image 
    distance (BID) is calculated to make object and image plane conjugate. Based on object
    distance and BID the conjugate ABCD matrix is calculated. The original distance of the image 
    plane in the LensSequence is not altered.
    """
    def __init__(self, EFL, BFL, FFL, V1H1, V2H2, EPP, EPD, EP_is_virtual, XPP, XPD, XP_is_virtual,
                 magnification, angular_magnification,
                 stop_radius,
                 BID, 
                 y_chief, u_chief, y_marg, u_marg,
                 ABCD_system, ABCD_conjugate
                 ):
        self.EFL = EFL
        self.BFL = BFL
        self.FFL = FFL
        self.V1H1 = V1H1
        self.V2H2 = V2H2
        self.EPP = EPP
        self.EPD = EPD
        self.EP_is_virtual = EP_is_virtual
        self.XPP = XPP
        self.XPD = XPD
        self.XP_is_virtual = XP_is_virtual
        self.magnification = magnification
        self.angular_magnification = angular_magnification
        self.stop_radius = stop_radius
        self.BID = BID
        self.y_chief = y_chief
        self.u_chief = u_chief
        self.y_marg = y_marg
        self.u_marg = u_marg
        self.ABCD_system = ABCD_system
        self.ABCD_conjugate = ABCD_conjugate


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
    
    def trace_ray_paraxially(self, y0, u0, start_surf, stop_surf, forward=True) -> Tuple[float, float]:
        """
        Trace a paraxial ray (y0, u0) from right before surface start_surf up to right after surface stop_surf
        in forward direction, which requires stop_surf > start_surf. If forward=False, then the ray
        is traced backward, which requires stop_surf < start_surf.
        """
        assert 1 <= start_surf < self.num_surfs -1, "Start and stop surfaces must not include object or image surface."
        if forward: assert stop_surf > start_surf
        if not forward: assert stop_surf < start_surf

        ynu = np.zeros((self.num_surfs,2)) * np.nan
        ynu[start_surf,:] = np.array([y0, u0*self.n[start_surf-1]]) # This is the ray vector *before* start_surf
        if forward:
            for s in range(start_surf, stop_surf):
                i_refrac = (s-1)*2
                i_transl = i_refrac+1 
                if s < self.num_surfs - 2:
                    # refraction followed by translation                                        
                    ynu_ = self.ABCD_matrices[i_refrac] @ ynu[s]
                    ynu[s+1] = self.ABCD_matrices[i_transl] @ ynu_
                if s == self.num_surfs - 2:
                    # one more refraction without translation at the last vertex
                    ynu[s+1] = self.ABCD_matrices[i_refrac] @ ynu[s]
                    ynu[s+1] = self._Tmat(self.vertex[-1]- self.vertex[-2], 1.0) @ ynu[s+1]
        else:
            for s in range(start_surf, stop_surf, -1):
                i_refrac = (s-1)*2
                # translation followed by refraction 
                ynu_ = np.linalg.inv(self.ABCD_matrices[i_refrac-1]) @ ynu[s]
                ynu[s-1] = np.linalg.inv(self.ABCD_matrices[i_refrac-2]) @ ynu_
        return ynu


    def _intersection_with_oA(self, y: float, u: float, z0: float) -> float:
        """For a ray (y,u) located at z-position z0, calculate its intersection with the optical axis."""
        z_int = z0 - y/u
        return z_int
    
    def _intersection_line_segments(self, y1: float, u1: float, y2: float, u2: float, z0: float) -> Tuple[float, float]:
        """
        Intersection point between ray (y1,u1) and (y2,u2) launched at z-position z0.
        """
        z_int = z0 - (y1 -y2)/(np.tan(u1) - np.tan(u2))
        y_int = y1 + np.tan(u1)*(z_int-z0)
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
        n2 = self.n[-1] # image space refractive index
        return (-self.system_matrix[0,0]/self.system_matrix[1,0]*n2)
    
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
        n2 = self.n[-1] # image space refractive index
        return (1.0-n2*self.system_matrix[0,0])/self.system_matrix[1,0]

    def _get_entrance_and_exit_pupil(self, EPD: int) -> Tuple[int, int]:
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
        # assert ((marginal_ray_angle > 0) and EP_is_virtual)
        ynu2 = np.matmul(np.linalg.inv(self.front_group_matrix), np.array([np.tan(marginal_ray_angle)*np.abs(self.vertex[0]), marginal_ray_angle*self.n[0]]).T)
        stop_radius = ynu2[0]

        # ynu_marginal = self.trace_ray_paraxially(np.tan(marginal_ray_angle)*np.abs(self.vertex[0]), marginal_ray_angle, 1, self.AS_surf, True)
        # print("ynu_marginal[:,0]=", ynu_marginal[:,0])

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
        diameter_XP = 2.0*(ynu2[0] + np.tan((ynu2[1]/(self.n[-2])))*(position_XP + (self.vertex[-1]-self.vertex[-2])))

        print("position_EP, marginal_ray_angle, abs(stop_radius), position_XP, abs(diameter_XP), EP_is_virtual, XP_is_virtual")
        print(position_EP, marginal_ray_angle, abs(stop_radius), position_XP, abs(diameter_XP), EP_is_virtual, XP_is_virtual)

        return position_EP, EPD, marginal_ray_angle, abs(stop_radius), position_XP, abs(diameter_XP), EP_is_virtual, XP_is_virtual


    def paraxial_quantities(self, config: Config):
        self.EFL=self.get_EFL()
        self.BFL=self.get_BFL()
        self.FFL=self.get_FFL()
        self.V1H1=self.get_V1H1()
        self.V2H2=self.get_V2H2(),
        self.EPP, self.EPD, self.marginal_ray_angle, self.stop_radius, self.XPP, self.XPD, self.EP_is_virtual, self.XP_is_virtual=self._get_entrance_and_exit_pupil(config.EPD)
        self.magnification=self.get_magnification()
        self.angular_magnification=self.get_angular_magnification(),
        self.BID=self.get_BID()


        return ParaxialQuantities(
            EFL=self.EFL, BFL=self.BFL, FFL=self.FFL, V1H1=self.V1H1, V2H2=self.V2H2,
            EPP=self.EPP, EPD=self.EPD,
            XPP=self.XPP, XPD=self.XPD,
            magnification=self.magnification, angular_magnification=self.angular_magnification,
            stop_radius=self.stop_radius,
            BID=self.BID,
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
    