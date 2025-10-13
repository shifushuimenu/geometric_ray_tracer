import numpy as np
from math import sin, cos
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

from nonmeridional_rays import raytrace_nonmeridional_rays, calculate_OPD
from plot import plot_ray, plot_surfaces, intersection_with_surface

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

# ====================================
# Launch a non-meridional ray fan
# ====================================
num_rays_per_fan = 55 # number of rays per ray fan
# with rotation angle gamma1 around the x-axis (positive angle means downward inclination)
# and rotation angle gamma2 around the y-axis (positive angle means left-turning when looking in the direction of the ray).
gamma1_field = 0.3*np.arctan(obj_height[0:num_fields]/t[0])  # inclination angle (in radians) of the ray bundle. IMPROVE: gamma1_field depends on the launch angle of the chief ray for each field position
gamma2_max = 0.03  # half opening angle of the ray bundle 
dg2 = gamma2_max*2/num_rays_per_fan

num_azimuth = 15 # rotate fay fan around chief ray, split interval [0,pi] into num_azimuth+1 different angles
dg3 = np.pi/num_azimuth

# Note that the chief ray is not rotated around itself.
num_rays = (num_rays_per_fan-1)*num_azimuth + 1

# A ray bundle at field position f is a tuple (P_intersect[0:3,0:num_surfs+1,0:num_rays,f], rayvecs[0:3,0:num_surfs+1,0:num_rays,f]). 
# The chief ray is is labelled as the first ray: (P_intersect[0:3,0:num_surfs+1,0,f], rayvecs[0:3,0:num_surfs+1,0,f])

# ===========================================================================================================================
P_intersect = np.zeros((3,num_surfs+1,num_rays,num_fields))
rayvecs = np.zeros((3,num_surfs+1,num_rays,num_fields)) 
gamma2_list = [(k-num_rays_per_fan//2)*dg2 for k in range(num_rays_per_fan) if k != num_rays_per_fan//2] # don't include chief ray
gamma3_list = [k*dg3 for k in range(num_azimuth)]
print("raytrace non-meridional rays")
for f in range(num_fields):    
    # chief ray
    r = 0
    P_intersect[0:3, 0, r, f] = np.array([0, obj_height[f], 0]) # object oriented along y-axis
    rayvecs[0:3, 0, r, f] = np.array([0, -sin(gamma1_field[f]), cos(gamma1_field[f])])

    for gamma2 in gamma2_list:
        for gamma3 in gamma3_list:
            r += 1           
            P_intersect[0:3, 0, r, f] = np.array([0, obj_height[f], 0]) # object oriented along y-axis
            rayvecs[0:3, 0, r, f] = np.array([-sin(gamma2)*cos(gamma3), 
                                              -cos(gamma1_field[f])*sin(gamma2)*sin(gamma3) - sin(gamma1_field[f])*cos(gamma2), 
                                              -sin(gamma1_field[f])*sin(gamma2)*sin(gamma3) + cos(gamma1_field[f])*cos(gamma2)])
            
P_intersect, rayvecs = raytrace_nonmeridional_rays(zS, R, n, P_intersect, rayvecs)
# ===========================================================================================================================

test1, test2 = raytrace_nonmeridional_rays(zS, R, n, P_intersect[:,:,:,np.newaxis,np.newaxis], rayvecs[:,:,:,np.newaxis,np.newaxis])

colors = ["blue", "green", "red"]
OPD = calculate_OPD(n, P_intersect)
for f in range(num_fields):
    for r in range(num_rays):
       plt.plot(range(num_surfs), OPD[:,r,f], color=colors[f])
plt.show()

plt.plot(P_intersect[0,-5,:,f], OPD[-5,:,f], '-o')
plt.plot(P_intersect[1,-5,:,f], OPD[-5,:,f], '-o')
plt.show()

fig = plt.figure(figsize=(6,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(P_intersect[0,-3,:,f], P_intersect[1,-3,:,f], OPD[-3,:,f], marker='o')
plt.show()
exit(1)

colors = ["blue", "green", "red"] if num_fields == 3 else mpl.color_sequences["tab10"][0:num_fields]
fig = None
for f in range(num_fields):
    for r in range(num_rays):
        fig = plot_ray(t, P_intersect[1,:,r,f], fig, z_sag=P_intersect[2,:,r,f]-zS[:], color=colors[f])

plot_surfaces(t, R, (t[0]*np.abs(gamma1_field[0])+max_obj_height)*np.ones_like(R), n, fig) # IMPROVE: use actual clear aperture at each surface 

fig.axes[0].set_ylim((-36.0,10.09))
fig.axes[0].axis("equal")
plt.show()

# Plot intersection points in the image plane
# Plot off-axis ray bundles relative to the chief ray.
fig = plt.figure("fig2", figsize=(6,6), edgecolor="g")
axs = fig.subplots(1,num_fields,sharex=True)
for f in range(num_fields):
    axs[f].axis("equal")
    y_CR = P_intersect[1,-1,0,f]
    axs[f].plot(P_intersect[0,-1,:,f], P_intersect[1,-1,:,f]-y_CR, 'o', color=colors[f])
fig.tight_layout()
plt.show()

# fig = plt.figure("fig3", figsize=(6,6))
# axs = fig.subplots(1,num_fields,sharex="all",)
# for f in range(num_fields):
#     P = intersection_with_surface((P_intersect[:,1,:,f], rayvecs[:,1,:,f]), zS[1])
#     axs[f].plot(P[0,:], P[1,:], '+', color=colors[f])
# fig.tight_layout()
# plt.show()