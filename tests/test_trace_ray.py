from trace_ray import RayTracer
from config import Config
from lens import read_lens
import matplotlib.pyplot as plt
import matplotlib as mpl
from plot import *
import numpy as np

# lens_sequence = read_lens("lens_files/lens_Kidger2004_modified.txt")
lens_sequence = read_lens("lens_files/stepper_lens.txt")
# config = Config(max_obj_height=1.414, entrance_pupil_diameter=2.0)
config = Config(max_obj_height=80.0, entrance_pupil_diameter=40.0)

ray_tracer = RayTracer(lens_sequence)

ray_data = ray_tracer.calculate_meridional_ray_data(lens_sequence, config)

fig, _, _ = plot_spherical_surfaces(lens_sequence.vertex, lens_sequence.R, lens_sequence.n, 
                              ray_data.clear_apertures, lens_sequence.AS_surf, config)
fig = plot_ray_bundles(ray_data, fig)
plt.show()

# Check distribution of angles in the ray bundle at the object
nonmeridional_ray_data= ray_tracer.calculate_nonmeridional_ray_data(lens_sequence, config)
ax = plt.subplot(1,1,1)
ax.plot(nonmeridional_ray_data.rayvecs[0,0,:,0], nonmeridional_ray_data.rayvecs[1,0,:,0], 'o')
ax.set_aspect("equal")
plt.show()


colors = ["blue", "green", "red"] if config.num_fields == 3 else mpl.color_sequences["tab10"][0:config.num_fields]
# Plot intersection points in the image plane
# Plot off-axis ray bundles relative to the chief ray.
fig = plt.figure("fig2", figsize=(6,6), edgecolor="g")
axs = fig.subplots(1,config.num_fields,sharex=True)
for f in range(config.num_fields):
    axs[f].axis("equal")
    y_CR = nonmeridional_ray_data.P_intersect[1,-1,0,f]
    axs[f].plot(nonmeridional_ray_data.P_intersect[0,-1,:,f], nonmeridional_ray_data.P_intersect[1,-1,:,f]-y_CR, 'o', color=colors[f])
fig.tight_layout()
plt.show()