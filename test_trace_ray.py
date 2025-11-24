from trace_ray import RayTracer
from config import Config
from lens import read_lens
import matplotlib.pyplot as plt
from plot import *
import numpy as np

lens_sequence = read_lens("lens_files/lens_Kidger2004_modified.txt")
# LS = read_lens("lens_files/stepper_lens.txt")

ray_tracer = RayTracer(lens_sequence)
config = Config(max_obj_height=1.414, entrance_pupil_diameter=2.0)

ray_data = ray_tracer.calculate_meridional_ray_data(lens_sequence, config)

fig, _, _ = plot_spherical_surfaces(lens_sequence.vertex, lens_sequence.R, lens_sequence.n, 
                              ray_data.clear_apertures, config)
fig = plot_ray_bundles(ray_data, fig)
plt.show()