import numpy as np
from math import sin, cos
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

from nonmeridional_rays import raytrace_nonmeridional_rays
from plot import plot_ray, plot_surfaces

# SECTION 1:
# User input: lens prescription file, field of view, F/# and wavelength.
# Load txt file, determine the surface powers and locate the surface which 
# is the aperture stop. Make sure there is only one aperture stop. 

# Take surface sag into account.
SAG = True

lens_file = sys.argv[1]
max_obj_height = float(sys.argv[2])
EPD = float(sys.argv[3])

lens_layout = np.loadtxt(lens_file, comments="#", skiprows=1)
num_surfs = lens_layout.shape[0]

stop_flag = np.zeros(num_surfs)
R = np.zeros(num_surfs)
t = np.zeros(num_surfs)
n = np.zeros(num_surfs)
V_d = np.zeros(num_surfs)
phi = np.zeros(num_surfs)

zS = np.zeros(num_surfs+1)

t[0] = lens_layout[0,3]
AS_surf = 0
sum_dists = 0
zS[0] = 0
for s in range(0, num_surfs):
    stop_flag[s] = lens_layout[s,1]
    R[s] = lens_layout[s, 2] 
    t[s] = lens_layout[s, 3]
    n[s] = lens_layout[s, 4]
    V_d[s] = lens_layout[s, 5]
    if s  > 0:
        phi[s] = (n[s] - n[s-1]) / R[s]  # surface power 

    sum_dists += t[s]
    zS[s+1] = sum_dists 

# special case: first surface is the aperture stop
if stop_flag[1] == 1:
    AS_surf = 1
    pass

# Find the aperture stop and verify that there is only one aperture stop.
found_AS = False
for s in range(1, num_surfs):    
    if stop_flag[s] == 1:
        if not found_AS:
            AS_surf = s
            found_AS = True
        else:
            raise ValueError("There can be only one aperture stop.")

if AS_surf == 0:
    raise ValueError(f"The object surface cannot be the aperture stop. AS_surf = {AS_surf}")

print("Aperture stop is surface", AS_surf, "at", np.sum(t[1:AS_surf]), "mm from the front vertex.")

# SECTION 2: Calculate the chief ray piercing height on the first surface.
# The chief ray goes from the tip of the object through the center of the aperture stop.

# The specified aperture stop is assumed to be the true aperture stop, that is, all lens elements
# are taken to have inifinite radial extent such that none acts as the actual aperture stop.

# The chief ray is calculated for three object heights ("field positions" in optics jargon):
# at full object height, at 70% object height and on axis.
obj_height = [max_obj_height, max_obj_height / np.sqrt(2.0), 0.0]
num_fields = len(obj_height)

num_rays = 55 # number of rays per ray fan
num_azimuth = 5 # rotate fay fan around chief ray, split interval [0,pi] into num_azimuth+1 different angles
dg3 = np.pi/(num_azimuth+1)

# ===========================================================================================================================
P_intersect = np.zeros((3,num_surfs+1,num_rays,num_fields))
rayvecs = np.zeros((3,num_surfs+1,num_rays,num_fields))
# Launch a non-meridional ray fan 
print("raytrace non-meridional rays")
# with rotation angle gamma1 around the x-axis (positive angle means downward inclination)
# and rotation angle gamma2 around the y-axis (positive angle means left-turning when looking in the direction of the ray).
gamma1 = -0.02  # inclination angle (in radians) of the ray bundle 
gamma2_max = 0.02  # half opening angle of the ray bundle 
dg2 = gamma2_max*2/num_rays
for f in range(num_fields):
    for r, gamma2 in enumerate([(k-num_rays//2)*dg2 for k in range(num_rays)]):
        P_intersect[0:3, 0, r, f] = np.array([0, obj_height[f], 0]) # object oriented along y-axis
        rayvecs[0:3, 0, r, f] = np.array([-sin(gamma2), -sin(gamma1)*cos(gamma2), cos(gamma1)*cos(gamma2)])
P_intersect, rayvecs = raytrace_nonmeridional_rays(zS, R, n, P_intersect, rayvecs)
# ===========================================================================================================================

colors = ["blue", "green", "red"] if num_fields == 3 else mpl.color_sequences["tab10"][0:num_fields]
fig = None
for f in range(num_fields):
    for r in range(num_rays):
        fig = plot_ray(t, P_intersect[1,:,r,f], fig, z_sag=P_intersect[2,:,r,f]-zS[:], color=colors[f])

plot_surfaces(t, R, (t[0]*np.abs(gamma1)+max_obj_height)*np.ones_like(R), n, fig)

plt.show()

# Plot intersection points in the image plane
fig = plt.figure("fig2", figsize=(6,6), edgecolor="g")
axs = fig.subplots(1,num_fields,sharex="all",)
for f in range(num_fields):
    # axs[f].axis("equal")
    axs[f].plot(P_intersect[0,-1,:,f], P_intersect[1,-1,:,f], 'o', color=colors[f])
fig.tight_layout()
plt.show()