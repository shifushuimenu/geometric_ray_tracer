# TODO:
# - visualize: plot paraxial focus and principal planes 
# - plot Gaussian intensity profile and beam waist as a function of z together with waist and divergence rays
# - Calculate position of nodal points if the index of refraction before and after the lens system is different.
import numpy as np
from typing import Tuple

from lens import LensSequence
from config import Config

__all__ = ["ParaxialRaytracer"]

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
        self.zvertex = LS.zdist
        self.ABCD_matrices = []
        self.ABCD_front_group = [] # all lens elements before the aperture stop
        self.ABCD_rear_group = [] # all lens elements behind the aperture stop
        # exclude object (surf=0) and image surface (surf=num_surfs-1) )when constructing the system matrix
        for i in range(1,LS.num_surfs-1):
            if i < LS.num_surfs-2:
                # refracting surface followed by translation
                self.ABCD_matrices.extend((self.Rmat(LS.phi[i]), self.Tmat(LS.t[i], LS.n[i])))
            elif i == LS.num_surfs-2:
                # The last refracting surface is NOT followed by translation.
                self.ABCD_matrices.extend((self.Rmat(LS.phi[i]),))
            if i < self.AS_surf:
                self.ABCD_front_group.extend((self.Rmat(LS.phi[i]), self.Tmat(LS.t[i], LS.n[i])))      
            elif i > self.AS_surf:
                self.ABCD_rear_group.extend((self.Tmat(LS.t[i-1], LS.n[i-1]), self.Rmat(LS.phi[i])))
 
        self.system_matrix = self.make_system_matrix(forward=True)

        self.rear_group_matrix = np.eye(2)
        for m in self.ABCD_rear_group:
            self.rear_group_matrix = np.matmul(m, self.rear_group_matrix)

        # The front group (in front of the aperture stop) is traced in reverse order.
        self.front_group_matrix = np.eye(2)
        for m in self.ABCD_front_group[::-1]:
            self.front_group_matrix = np.matmul(np.linalg.inv(m), self.front_group_matrix)
 
    def trace_ray(self, y0: float, u0: float) -> Tuple[float, float]:
        """ynu trace a ray through the system from the first to the last vertex"""
        ynu = self.system_matrix @ np.array(y0, u0*self.n[0])
        y1 = ynu[0]
        u1 = ynu[1]/self.n[-2] # n[-1] would be the index of refraction *after* the image surface, which is not used.
        return y1, u1

    def make_system_matrix(self, forward=True):        
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
    
    def make_conjugate_matrix(self, object_dist, image_dist):
        return self.Tmat(image_dist) @ self.system_matrix @ self.Tmat(-object_dist)
            
    def Tmat(self, t, n=Config.n_air):
        """Return ABCD translation matrix
        d - float:  distance to the right of the optical element"
        n - float:  index of refraction"""
        return np.array([[1.0, t/n],
                         [0.0, 1.0]], dtype=np.float64)
    def Rmat(self, phi):
        """Return ABCD refraction matrix
        phi = 1/f - float: refractive power"""
        return np.array([[1.0,  0.0],
                         [-phi, 1.0]], dtype=np.float64)
    
    def Mmat(self, R, n=Config.n_air):
        """Return ABCD mirror matrix"""
        return self. Rmat(phi=-2.0*n/R)
    
    def get_image_distance(self, object_dist):
        """The image distance as measured from the rightmost vertex V2 of the lens system."""
        assert object_dist < 0, "object distance should be a negative number"
        A = self.system_matrix[0,0]; B = self.system_matrix[0,1]
        C = self.system_matrix[1,0]; D = self.system_matrix[1,1]
        image_dist = (object_dist*A - B) / (D - object_dist*C)
        self.conjugate_matrix = self.make_conjugate_matrix(object_dist, image_dist)
        return image_dist
    
    def is_conjugate(self, N):
        """Check whether an ABCD matrix N defines conjugate planes."""
        return np.isclose(N[0,1], 0.0, atol=1e-8)
    
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

    def _intersection_with_oA(self, y: float, u: float, z0: float) -> float:
        """For a ray (y,u) located at z-position z0, calculate its intersection with the optical axis."""
        z_int = z0 - y/u
        return z_int

    def get_entrance_pupil(self):
        # IMPROVE: special cases
        y_chief = 0; u_chief = +0.4
        ynu1 = np.matmul(self.front_group_matrix, np.array([y_chief, u_chief*self.n[self.AS_surf-1]]).T)
        position_EP = self._intersection_with_oA(ynu1[0], ynu1[1]/self.n[0], self.zvertex[1])
        diameter_EP = 0
        return position_EP, diameter_EP

    def get_exit_pupil(self):
        # Trace a ray from the center of the aperture stop through the rear group of the lens system.
        # The z-position where the ray leaving the rear group intersects the optical axis is the position of the exit 
        # pupil
        y_chief = 0.0; u_chief = -0.1 # angle is arbitrary
        ynu1 = np.matmul(self.rear_group_matrix, np.array([y_chief, u_chief*self.n[self.AS_surf]]).T)
        print("ynu1=", ynu1)
        # Richtiges Ergebnis, ich verstehe aber nicht, warum.
        position_XP = self._intersection_with_oA(ynu1[0], ynu1[1]/self.n[-2], self.zvertex[-2]) - self.zvertex[-1]
        print("position_XP=", position_XP)
        print(self.rear_group_matrix)
        # measured from the image plane 
        # position_XP = position_XP - self.zvertex[-1]
        diameter_XP = 0
        return position_XP, diameter_XP


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
    