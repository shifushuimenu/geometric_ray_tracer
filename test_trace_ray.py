from trace_ray import *
from lens import *
from visualize import *
import matplotlib.pyplot as plt
import numpy as np

LS = read_lens("lens_files/test_doublet_realXP.txt")
# LS = read_lens("lens_files/stepper_lens.txt")

obj_height= [8.0, 4.0]
obj_angle = [0.0, -0.1]

# 1. Trace ray forward from the image surface.
y, u, z_sag, _ = trace_tangential_ray(obj_height, obj_angle, LS, LS.AS_surf, True)
print("y=", y)
print("u=", u)
# 2. Trace ray backward from an arbitrary surface.
# 3. Trace ray forward from an arbitrary surface. 
# 3(a). Trace ray forward from a flat surface. 
# 3(b). Trace ray forward from a curved surface.

fig = None
for b in range(y.shape[-1]):
    fig = plot_ray(LS.t, y[:,b], fig, color="r", linewidth=1)
plot_surfaces(LS.t, LS.R, 5*np.max(obj_height)*np.ones_like(LS.R), LS.n, fig)
# plt.ylim((-1.2*max(max_obj_height, max(heights)), 1.2*max(max_obj_height, max(heights))))
plt.show()