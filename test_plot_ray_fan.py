from math import sin, cos

from trace_ray import RayTracer
from config import Config
from lens import read_lens
import matplotlib.pyplot as plt
import matplotlib as mpl
from plot import *
import numpy as np
from nonmeridional_rays import raytrace_nonmeridional_rays

lens_sequence = read_lens("lens_files/stepper_lens.txt")
config = Config(max_obj_height=80.0, entrance_pupil_diameter=40.0)
# lens_sequence = read_lens("lens_files/lens_Kidger2004_modified.txt")
# config = Config(max_obj_height=1.414, entrance_pupil_diameter=2.0)
config.num_rays = 31
ray_tracer = RayTracer(lens_sequence)

# tangential ray fan
tangential_ray_data = ray_tracer.calculate_meridional_ray_data(lens_sequence, config)

# sagittal ray fan
# Initialize a sagittal ray fan which fills the aperture stop horizontally.
P_intersect = np.zeros((3,lens_sequence.num_surfs, config.num_rays, config.num_fields))
rayvecs = np.zeros((3,lens_sequence.num_surfs, config.num_rays, config.num_fields))
alpha = ray_tracer.marginal_ray_angle
for f in range(config.num_fields):    
    gamma1_field = ray_tracer.u_chief[0,:]  # inclination angles
    gamma3 = 0
    gamma2 = [-alpha + 2*r*alpha/(config.num_rays-1) for r in range(config.num_rays)]
    for r in range(config.num_rays):        
        P_intersect[0:3, 0, r, f] = np.array([0, config.obj_heights[f], lens_sequence.vertex[0]]) # object oriented along y-axis
        rayvecs[0:3, 0, r, f] = np.array([-sin(gamma2[r])*cos(gamma3), 
                                          -cos(gamma1_field[f])*sin(gamma2[r])*sin(gamma3) - sin(gamma1_field[f])*cos(gamma2[r]), 
                                          -sin(gamma1_field[f])*sin(gamma2[r])*sin(gamma3) + cos(gamma1_field[f])*cos(gamma2[r])])

P_intersect, rayvecs = raytrace_nonmeridional_rays(lens_sequence.vertex, lens_sequence.R, lens_sequence.n, P_intersect, rayvecs)        

fig, axs = plt.subplots(nrows = 1, ncols = config.num_fields, squeeze=True, sharey=True, sharex=True)
fig.set_layout_engine("tight")

for f in range(config.num_fields)[::-1]:
    IMAG_SURF = -1

    # tangential ray fan
    CHIEF_INDEX = tangential_ray_data.CHIEF_RAY_INDEX
    y = tangential_ray_data.y[IMAG_SURF,:,f] - tangential_ray_data.y[IMAG_SURF,CHIEF_INDEX,f]
    # We take the coordinate in the aperture stop as the pupil coordinate since the aperture stop and entrance pupil are conjugate planes.
    Py = tangential_ray_data.y[lens_sequence.AS_surf,:,f] # chief ray height at aperture stop is zero by definition
    # normalize
    Py /= np.max(Py)

    # sagittal ray fan
    x = P_intersect[0,IMAG_SURF,:,f] - 0.0 # should be symmetric about x=0
    Px = P_intersect[0,lens_sequence.AS_surf,:,f]
    Px /= np.max(Px)

    axs[f].plot(Py, y, '-', label="tangential")
    axs[f].plot(Px, x, '--', label="sagittal")
    axs[f].grid(visible=True)
    axs[f].set_box_aspect(1.0)

fig.legend()
plt.show()