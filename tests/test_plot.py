import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

from pupils_and_stops import find_chief_rays
from lens import read_lens
from config import Config
from trace_ray import trace_tangential_ray
from plot import *


# SECTION 1:
# User input: lens prescription file, field of view, F/# and wavelength.
lens_file = sys.argv[1]
max_obj_height = float(sys.argv[2]) 
EPD = float(sys.argv[3]) # entrance pupil diameter

lens_sequence = read_lens(lens_file, SAG = True)
config = Config(max_obj_height, EPD)

obj_height = [max_obj_height, max_obj_height / np.sqrt(2.0), 0.0]
num_fields = len(obj_height)

y_cr, u_cr, z_sag_cr = find_chief_rays(lens_sequence, obj_height)
EPL = obj_height[0]/np.tan(u_cr[0,0]) - lens_sequence.t[0]
marginal_ray_angle = np.arctan((EPD/2.0)/(EPL+lens_sequence.t[0]))

num_rays = 55 # number of rays in a ray bundle for a given field
assert num_rays % 2 == 1
y_obj = np.zeros((num_rays, num_fields))
u_obj = np.zeros((num_rays, num_fields))

# loop over fields
for f in range(num_fields):
    y_obj[:,f] = obj_height[f]
    dtheta = 2*marginal_ray_angle/num_rays
    # k=0 and k=num_rays-1 are the marginal rays, k=num_rays//2 is the chief ray of the ray bundle.
    u_obj[:,f] = np.array([-u_cr[0,f] + (k-num_rays//2)*dtheta for k in range(num_rays)])


y, u, z_sag, y_vertexplane = trace_tangential_ray(y_obj[:,:], u_obj[:,:], lens_sequence, surf_start=0)


# The heights of the outermost rays at each surface determine its clear aperture radius.
heights = np.zeros(lens_sequence.num_surfs)
heights[0] = 0
for s in range(1, lens_sequence.num_surfs):
    for f in range(num_fields):
        for r in [0,num_rays-1]: # consider only outermost rays
            hs = np.abs(y[s,r,f])
            if (hs > heights[s]): 
                heights[s] = hs

print(heights)
fig = plot_spherical_surfaces(lens_sequence.t, lens_sequence.R, lens_sequence.n, heights, config)

plt.show()