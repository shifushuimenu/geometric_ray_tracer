import unittest

from trace_ray import RayTracer
from paraxial import *
from lens import read_lens
from plot import plot_paraxial_surfaces

import matplotlib.pyplot as plt

class Test1(unittest.TestCase):
   
   def setUp(self):
       self.lens_sequence = read_lens("lens_files/stepper_lens.txt", SAG=False, lens_unit="mm")
       self.PR = ParaxialRaytracer(self.lens_sequence)
       self.raytracer = RayTracer(self.lens_sequence)

   def test_image_distance(self):
       print("paraxial image distance=", self.PR.get_image_distance(-95.0))

   def test_ABCD_system_matrix(self):
       print("ABCD=", self.PR._make_system_matrix())

   def test_raytrace(self):
       y, u, z_sag, _ = self.raytracer.trace_tangential_ray(0.0, 0.2)
       print("shape=", self.PR.ABCD_matrices[0].shape)
       ynu = self.PR.trace_ray_paraxially(y[1], u[1], 1, self.lens_sequence.AS_surf, True)
       fig = plot_paraxial_surfaces(self.PR.vertex)
    #    fig.axes[0].plot(self.PR.vertex, ynu[:,0])
    #    fig.axes[0].plot(self.PR.vertex+z_sag, y, '--')

       start_surf = self.lens_sequence.AS_surf
       stop_surf = self.lens_sequence.num_surfs-2
       ynu = self.PR.trace_ray_paraxially(ynu[start_surf,0], ynu[start_surf,1], start_surf, stop_surf, True)
       fig.axes[0].plot(self.PR.vertex, ynu[:,0], '-', color="green")
       print("ynu[stop_surf,0], ynu[stop_surf,1]=", ynu[stop_surf,0], ynu[stop_surf,1])
       ynu_reverse = self.PR.trace_ray_paraxially(ynu[stop_surf,0], ynu[stop_surf,1], stop_surf, start_surf, False)
       fig.axes[0].plot(self.PR.vertex, ynu_reverse[:,0], '--', color="blue")
       print("ynu_reverse=", ynu_reverse[stop_surf,:])
       plt.show()       


   def tearDown(self):
       pass

if __name__ == "__main__":
    unittest.main()
