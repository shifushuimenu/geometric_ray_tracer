import unittest

from trace_ray import RayTracer
from config import Config
from paraxial import *
from lens import read_lens
from plot import plot_paraxial_surfaces

import matplotlib.pyplot as plt
import numpy as np

class Test1(unittest.TestCase):
   
    def setUp(self):
        # self.lens_sequence = read_lens("lens_files/test_doublet_realXP_v2.txt", SAG=False, lens_unit="mm")
        self.lens_sequence = read_lens("lens_files/lens_Kidger2004_modified.txt", SAG=False, lens_unit="mm")
        # self.lens_sequence = read_lens("lens_files/lens_Kidger2004_modified_v2.txt", SAG=False, lens_unit="mm")
        # self.lens_sequence = read_lens("lens_files/stepper_lens.txt", SAG=False, lens_unit="mm")
        self.PR = ParaxialRaytracer(self.lens_sequence)
        self.raytracer = RayTracer(self.lens_sequence)

#     def test_image_distance(self):
#         print("paraxial image distance=", self.PR.get_image_distance(-95.0))

#     def test_ABCD_system_matrix(self):
#         print("ABCD=", self.PR._make_system_matrix())

# #    def test_find_chief_ray(self):
# #        ynu = self.PR.find_chief_ray(obj_height=1.0)
# #        fig = plot_paraxial_surfaces(self.PR.vertex)
# #        fig.axes[0].plot(self.PR.vertex, ynu[:,0])
# #        plt.show()


    # def test_trace_ray_paraxially(self, plot=True):
    #     """Check that for small angles the paraxial and non-paraxial ray tracing routines give the same trajectories."""
    #     y, theta, z_sag, _ = self.raytracer.trace_tangential_ray(0.0, 0.01)
    #     ynu = self.PR.trace_ray_paraxially(y[2], theta[1], 2, self.lens_sequence.num_surfs-2, forward=True)
    #     ynu_reverse = self.PR.trace_ray_paraxially(ynu[self.lens_sequence.num_surfs-2,0], 
    #                                                 ynu[self.lens_sequence.num_surfs-2,1]/self.lens_sequence.n[self.lens_sequence.num_surfs-2], 
    #                                                 self.lens_sequence.num_surfs-2, 1, forward=False)
        
    #     if plot:
    #         fig = plot_paraxial_surfaces(self.PR.vertex)
    #         fig.axes[0].plot(self.PR.vertex, ynu[:,0], label="paraxial, forward")
    #         fig.axes[0].plot(self.PR.vertex, ynu_reverse[:,0], 'o', label="paraxial, reverse")
    #         fig.axes[0].plot(self.PR.vertex+z_sag, y, '--', label="non-paraxial")

    #         plt.legend()
    #         plt.show()

    #         plt.plot(self.PR.vertex, theta*self.PR.n, label=r"$\theta \cdot n$")
    #         plt.plot(self.PR.vertex, ynu[:,1], label=r"$u$")
    #         plt.plot(self.PR.vertex, ynu_reverse[:,1], label=r"$u$ reverse")
    #         plt.legend()
    #         plt.show()

    #     for i in range(len(ynu)):
    #         print("i=", i, " : ", ynu[i,:], " == ", ynu_reverse[i,:], f"non-paraxial: y={y[i]}, theta*n={theta[i]*self.PR.n[i]}")

    #     assert np.isclose(ynu[np.where(np.invert(np.isnan(ynu)))], ynu_reverse[np.where(np.invert(np.isnan(ynu_reverse)))]).all()
    #     assert np.isclose(ynu[:,0][np.where(np.invert(np.isnan(ynu[:,0])))], y[np.where(np.invert(np.isnan(ynu[:,0])))], atol=1e-3).all()
    #     assert np.isclose(ynu[:,1][np.where(np.invert(np.isnan(ynu[:,1])))], (theta*self.PR.n)[np.where(np.invert(np.isnan(ynu[:,1])))], atol=1e-2).all()


    def test_find_chief_ray(self):

        obj_height = 1.414
        EPD = 1.0

        fig = plot_paraxial_surfaces(self.PR.vertex)
        y, u = self.PR.find_chief_ray(obj_height=obj_height)
        print("ynu=", y)
        ynu1 = self.PR._trace_ray_paraxially_front_group_to_object(0.0, u[self.PR.AS_surf])
        print("ynu1=", ynu1)

        ynu_forward = self.PR.trace_ray_paraxially(y[1], u[0], 1, self.PR.num_surfs-2, True)

        fig.axes[0].plot(self.PR.vertex, y, '-o', label="from find chief ray, reverse")
        fig.axes[0].plot(self.PR.vertex, ynu_forward[:,0], '--', label="forward")

        # compare with non-paraxial ray tracing
        y_cr, u_cr, z_cr = self.raytracer.find_chief_rays([obj_height])
        ynu_cr_forward = self.PR.trace_ray_paraxially(y_cr[1,0], -u_cr[0,0], 1, self.PR.num_surfs-2, forward=True)
        print("AS_surf=", self.PR.AS_surf)
        print("u_cr[self.PR.AS_surf,0]=", u_cr[:,0])
        print("y_cr[self.PR.AS_surf,0]=", y_cr[:,0])
        ynu_cr_reverse = self.PR.trace_ray_paraxially(y_cr[self.PR.AS_surf,0], -u_cr[self.PR.AS_surf,0], self.PR.AS_surf, 1, forward=False, skip_start_surf=True)
        fig.axes[0].plot(self.PR.vertex, y_cr[:,0], label="chief ray (non-paraxial)") # OK
        fig.axes[0].plot(self.PR.vertex, ynu_cr_forward[:,0], label="chief ray (paraxial)") # OK
        fig.axes[0].plot(self.PR.vertex, ynu_cr_reverse[:,0], '--', label="chief ray reverse (paraxial)") # OK

        ynu_obj_to_image = self.PR.trace_ray_paraxially_object_to_image(obj_height, u[0], True)
        fig.axes[0].plot(self.PR.vertex, ynu_obj_to_image[:,0], '-o', label="obj to image")

        y_marginal_ray, u_marginal_ray = self.PR.marginal_ray_from_EPD(EPD)
        fig.axes[0].plot(self.PR.vertex, y_marginal_ray[:], '--', label="marginal ray")
        fig.axes[0].plot(self.PR.vertex, -y_marginal_ray[:], '--', label="marginal ray")

        plt.legend()
        plt.show()


    def test_pupils(self):
        EPD = 2.0
        position_EP, EPD, marginal_ray_angle, stop_radius, position_XP, diameter_XP, EP_is_virtual, XP_is_virtuall = self.PR._get_entrance_and_exit_pupil(EPD=EPD)
        print("EPP=", position_EP, "stop_radius=", stop_radius, "XPP=", position_XP, "XPD=", diameter_XP)

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

    def test_paraxial_quantities(self):

        config = Config(max_obj_height=1.414, entrance_pupil_diameter=2.0)
        PQ = self.PR.paraxial_quantities(config)

        print("paraxial quantities=", PQ)

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()
