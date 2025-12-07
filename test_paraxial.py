import unittest

from trace_ray import RayTracer
from paraxial import *
from lens import read_lens
from plot import plot_paraxial_surfaces

import matplotlib.pyplot as plt
import numpy as np

class Test1(unittest.TestCase):
   
   def setUp(self):
    #    self.lens_sequence = read_lens("lens_files/test_doublet_realXP_v2.txt", SAG=False, lens_unit="mm")
       self.lens_sequence = read_lens("lens_files/lens_Kidger2004_modified.txt", SAG=False, lens_unit="mm")
       self.PR = ParaxialRaytracer(self.lens_sequence)
       self.raytracer = RayTracer(self.lens_sequence)

   def test_image_distance(self):
       print("paraxial image distance=", self.PR.get_image_distance(-95.0))

   def test_ABCD_system_matrix(self):
       print("ABCD=", self.PR._make_system_matrix())

   def test_raytrace(self):
    #    y, u, z_sag, _ = self.raytracer.trace_tangential_ray(0.0, 0.2)
    #    print("shape=", self.PR.ABCD_matrices[0].shape)
    #    ynu = self.PR.trace_ray_paraxially(y[1], u[1], 1, self.lens_sequence.AS_surf, True)
       fig = plot_paraxial_surfaces(self.PR.vertex)
    #    fig.axes[0].plot(self.PR.vertex, ynu[:,0])
    #    fig.axes[0].plot(self.PR.vertex+z_sag, y, '--')

    #    # from the aperture stop till the last surface
    #    start_surf = self.lens_sequence.AS_surf
    #    stop_surf = self.lens_sequence.num_surfs-2
    #    ynu = self.PR.trace_ray_paraxially(ynu[start_surf,0], ynu[start_surf,1], start_surf, stop_surf, True)
    #    fig.axes[0].plot(self.PR.vertex, ynu[:,0], '-', color="green")
    #    ynu_chief_reverse = self.PR.trace_ray_paraxially(0.0, 0.1, self.lens_sequence.AS_surf, 1, False)
    #    fig.axes[0].plot(self.PR.vertex, ynu_chief_reverse[:,0], '--', color="blue")
    #    ynu_chief_forward = self.PR.trace_ray_paraxially(0.0, 0.1, self.lens_sequence.AS_surf, self.lens_sequence.num_surfs, True)
    #    fig.axes[0].plot(self.PR.vertex, ynu_chief_forward[:,0], '-', color="blue")

       EPD = 2.0

       EPP, marginal_ray_angle, stop_radius, XPP, XPD, EP_is_virtual, XP_is_virtual = self.PR._get_entrance_and_exit_pupil(EPD=EPD)
       print("EPP=", EPP, "stop_radius=", stop_radius, "XPP=", XPP, "XPD=", XPD)

    #    ynu1_marginal_forward = self.PR.trace_ray_paraxially(stop_radius, 0.1, self.lens_sequence.AS_surf, self.lens_sequence.num_surfs, True)
    #    ynu2_marginal_forward = self.PR.trace_ray_paraxially(stop_radius, 0.0, self.lens_sequence.AS_surf, self.lens_sequence.num_surfs, True)
    #    fig.axes[0].plot(self.PR.vertex, ynu1_marginal_forward[:,0], '--', color="red")
    #    fig.axes[0].plot(self.PR.vertex, ynu2_marginal_forward[:,0], '-', color="red")

    #    ynu1 = np.matmul(self.PR.front_group_matrix, np.array([stop_radius, 0.0]).T)
    #    ynu2 = np.matmul(self.PR.front_group_matrix, np.array([stop_radius, 0.1]).T)

    #    z_int, y_int = self.PR._intersection_line_segments(*ynu1, *ynu2, 0.0)
    #    print("EP: z_int=", z_int, "y_int=", y_int)
    #    fig.axes[0].plot([0.0, z_int], [ynu1[0], y_int], "-o")
    #    fig.axes[0].plot([0.0, z_int], [ynu2[0], y_int], "-o")


    #    ynu1 = np.matmul(self.PR.rear_group_matrix, np.array([stop_radius, 0.0]).T)
    #    ynu2 = np.matmul(self.PR.rear_group_matrix, np.array([stop_radius, 0.1]).T)

    #    z_int, y_int = self.PR._intersection_line_segments(ynu1[0], ynu1[1]/1.0, ynu2[0], ynu2[1]/1.0, self.lens_sequence.vertex[-2])
    #    print("XP: z_int=", z_int, "y_int=", y_int)
    #    fig.axes[0].plot([self.lens_sequence.vertex[-2], z_int], [ynu1[0], y_int], "-o")
    #    fig.axes[0].plot([self.lens_sequence.vertex[-2], z_int], [ynu2[0], y_int], "-o")

    #    # vertical lines
    #    EPP_ = EPP + self.lens_sequence.vertex[1]
    #    fig.axes[0].plot([EPP_, EPP_], [EPD/2.0, 1.2*EPD/2.0], linewidth=3, color="k")
    #    fig.axes[0].plot([EPP_, EPP_], [-EPD/2.0, -1.2*EPD/2.0], linewidth=3, color="k")
    #    fig.axes[0].plot([EPP_-0.2*EPD/2.0, EPP_+0.2*EPD/2.0], [EPD/2.0, EPD/2.0], linewidth=3, color="k")
    #    fig.axes[0].plot([EPP_-0.2*EPD/2.0, EPP_+0.2*EPD/2.0], [-EPD/2.0, -EPD/2.0], linewidth=3, color="k")
    #    fig.axes[0].text(EPP_, 1.3*EPD/2.0, "EP")

    #    XPP_ = XPP + self.lens_sequence.vertex[-2]
    #    fig.axes[0].plot([XPP_, XPP_], [XPD/2.0, 1.2*XPD/2.0], linewidth=3, color="k")
    #    fig.axes[0].plot([XPP_, XPP_], [-XPD/2.0, -1.2*XPD/2.0], linewidth=3, color="k")
    #    fig.axes[0].plot([XPP_-0.2*XPD/2.0, XPP_+0.2*XPD/2.0], [XPD/2.0, XPD/2.0], linewidth=3, color="k")
    #    fig.axes[0].plot([XPP_-0.2*XPD/2.0, XPP_+0.2*XPD/2.0], [-XPD/2.0, -XPD/2.0], linewidth=3, color="k")
    #    fig.axes[0].text(XPP_, 1.3*XPD/2.0, "XP")

    #    plt.show()

   def tearDown(self):
       pass

if __name__ == "__main__":
    unittest.main()
