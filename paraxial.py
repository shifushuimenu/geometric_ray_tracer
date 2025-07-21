import numpy as np

class ParaxialRaytracer(object):
    """Paraxial (ynu) ray tracer using ABCD matrices
        (yk   )  = (A B)  (y0   )
        (nk*uk)    (C D)  (n0*y0)
    """
    version = 1.0

    def __init__(self, surfs):
        self.forward = True
        self.surfs = surfs
        self.num_surfs = len(surfs)
        self.ABCD_matrices = []
        self.flag = []; self.phi = []; self.t = []; self.n_d = [], self.V_d = []
        for idx, s in enumerate(self.surfs):
            (flag_, phi_, t_, n_d_, V_d_) = s
            self.flag.append(flag_); self.phi.append(phi_); self.t.append(t_); self.n_d.append(n_d_); self.V_d.append(V_d_)
            self.ABCD_matrices.extend((self.Rmat(phi_), self.Tmat(t_, n_d_)))

        self.system_matrix = self.make_system_matrix(forward=True)
        
    def make_system_matrix(self, forward=True):        
        M = np.eye(2)
        if forward:
            for m in self.ABCD_matrices:
                M = np.matmul(m, M)
            self.forward = True
        else:
            for m in self.ABCD_matrices[::-1]:
                M = np.matmul(m, M)
            self.forward = False
        return M
    
    def make_conjugate_matrix(self, object_dist, image_dist):
        return self.Tmat(image_dist) @ self.system_matrix @ self.Tmat(-object_dist)
            
    def Tmat(self, t, n=1.0):
        """translation matrix
        d - float:  distance to the right of the optical element"
        n - float:  index of refraction"""
        return np.array([[1.0, t/n],
                         [0.0, 1.0]], dtype=np.float64)
    def Rmat(self, phi):
        """refraction matrix
        phi = 1/f - float: refractive power"""
        return np.array([[1.0,  0.0],
                         [-phi, 1.0]], dtype=np.float64)
    
    def Mmat(self, R, n=1.0):
        """mirror matrix"""
        return self. Rmat(phi=-2.0*n/R)
    
    def get_image_distance(self, object_dist):
        assert object_dist < 0, "object distance should be a negative number"
        A = self.system_matrix[0,0]; B = self.system_matrix[0,1]
        C = self.system_matrix[1,0]; D = self.system_matrix[1,1]
        image_dist = (object_dist*A - B) / (D - object_dist*C)
        self.conjugate_matrix = self.make_conjugate_matrix(object_dist, image_dist)
        return image_dist
    
    def is_conjugate(self, N):
        return np.isclose(N[0,1], 0.0)
    
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
        n2 = self.n_d[-1] # image space refractive index
        return (-self.system_matrix[0,0]/self.system_matrix[1,0]*n2)
    
    def get_FFL(self):
        """get front focal length"""
        n1 = self.n_d[0] # object space refractive index
        return (-n1*self.system_matrix[1,1]/self.system_matrix[1,0])
    
    def get_V1H1(self):
        """distance of the front principal plane P1 measured from the front vertex"""
        n1 = self.n_d[0] # object space refractive index
        return n1*(self.system_matrix[1,1]-1.0)/self.system_matrix[1,0]

    def get_V2H2(self):
        """distance of the back principal plane P2 measured from the back vertex (Note: V2H2 is negative if P2 is inside the lens.)"""
        n2 = self.n_d[-1] # image space refractive index
        return (1.0-n2*self.system_matrix[0,0])/self.system_matrix[1,0]
        
    
class GaussianRaytracer(ParaxialRaytracer):
    """Propagate Gaussian beams specified by a complex parameter q using paraxial ray tracing."""
    def __init__(self, surfs):
        super().__init__(surfs)

    def propagate_Gaussian_beam(self, q):
        assert np.iscomplex(q), "Gaussian beam parameter must be a complex number"
        A = self.system_matrix[0,0]; B = self.system_matrix[0,1]
        C = self.system_matrix[1,0]; D = self.system_matrix[1,1]
        return (A*q + B) / (C*q + D)