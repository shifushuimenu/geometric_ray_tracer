# TODO:
# - speed up calculation of chief ray using binary search (DONE)
# - visualize aperture stop, entrance and exit pupil, chief rays and marginal rays
# - better selection of initial u_min and u_max which works for all lens systems
# - chromatic aberrations, Seidel coefficiens in waves, additional quantities such as paraxial working F-number 
# - input mode where the field of view is specified rather than maximum object height
# - ray fan plot at the paraxial image plane

"""
Ray tracer in paraxial approximation for finite conjugate system.

Based on Stephen Remillard's ray tracing code:
https://www.youtube.com/watch?v=5962dgvPZCk
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time

from lens import read_lens
from trace_ray import trace_tangential_ray
from pupils_and_stops import find_chief_rays, intersection_line_segments
from plot import plot_ray, plot_spherical_surfaces, plot_paraxial_surfaces
from aberrations import Seidel3rd_aberrations

plt.ion()

# SECTION 1:
# User input: lens prescription file, field of view, F/# and wavelength.
lens_file = sys.argv[1]
max_obj_height = float(sys.argv[2]) 
EPD = float(sys.argv[3]) # entrance pupil diameter

lens_sequence = read_lens(lens_file, SAG = True)

# SECTION 2: Calculate the chief ray piercing height on the first surface.
# The chief ray goes from the tip of the object through the center of the aperture stop.

# The specified aperture stop is assumed to be the true aperture stop, that is, all lens elements
# are taken to have inifinite radial extent such that none acts as the actual aperture stop.

# The chief ray is calculated for three object heights ("field positions" in optics jargon):
# at full object height, at 70% object height and on axis.
obj_height = [max_obj_height, max_obj_height / np.sqrt(2.0), 0.0]
MAX_FIELD_INDEX=0
ON_AXIS_FIELD_INDEX=-1
num_fields = len(obj_height)


# REMOVE
AS_surf = lens_sequence.AS_surf
SAG = lens_sequence.SAG
n = lens_sequence.n
phi = lens_sequence.phi
t = lens_sequence.t
R = lens_sequence.R
num_surfs = lens_sequence.num_surfs
zdist = lens_sequence.vertex
stop_flag = lens_sequence.stop_flag
Vd = lens_sequence.Vd
# REMOVE 

y_cr, u_cr, z_sag_cr = find_chief_rays(lens_sequence, obj_height)

# entrance pupil location (measured from the vertex of the first surface)
EPL = obj_height[0]/np.tan(u_cr[0,0]) - lens_sequence.t[0]

marginal_ray_angle = np.arctan((EPD/2.0)/(EPL+lens_sequence.t[0]))*0.8
ObjNA = n[0]*np.sin(marginal_ray_angle)
FOV = np.arctan((obj_height[0]-y_cr[1,0])/lens_sequence.t[0])

# Plot the chief rays
for f in range(num_fields):
    if f==0:
        fig = plot_ray(zdist, y_cr[:,f], z_sag=z_sag_cr[:,f], color="orange", linewidth=4)
    else:
        fig = plot_ray(zdist, y_cr[:,f], fig, z_sag=z_sag_cr[:,f], color="orange", linewidth=4)

# SECTION 3: Trace "fields" of height [obj_hgt, obj_hgt / sqrt(2), 0]
# with a cone of rays around each chief ray launch angle. For half the opening angle of the 
# cone of rays we choose the marginal ray angle.
num_rays = 7 # number of rays in a ray bundle for a given field
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

# plt.show()
# generate_ray_fan_plot(y, AS_surf, 1.0, num_surfs)
# exit(1)

# The height of the  marginal ray of the on-axis field at the aperture stop gives the stop radius.
stop_radius = np.abs(y[AS_surf,num_rays-1,ON_AXIS_FIELD_INDEX])

# ===============================================================
# Locate the position of the *exit pupil* and its diameter.
# ===============================================================
# Trace two rays from the edge of the aperture stop towards the image side. 
# Their intersection point as viewed from the image side gives the edge 
# of the exit pupil since it is the image of the aperture stop. The exit pupil
# can be a virtual image.
# (y_pupil, u_pupil) is the ray that goes through the edge of the aperture stop and through 
# the edge of the exit pupil.
y_pupil = np.zeros((num_surfs,2))
u_pupil = np.zeros((num_surfs,2))

# Ray 1 is just a copy of the on-axis marginal ray.
y_pupil[0:AS_surf+1,0] = y[0:AS_surf+1,num_rays-1,-1] # The last field position is the on-axis field.
u_pupil[0:AS_surf+1,0] = u[0:AS_surf+1,num_rays-1,-1]
# Ray 2 goes through the center of the next lens element. In the paraxial approximation, 
# any other ray would be just as good. 
y_pupil[AS_surf,1] = stop_radius
if np.isclose(t[AS_surf], 0):
    dist_right_of_AS = t[AS_surf+1]
else:
    dist_right_of_AS = t[AS_surf]
u_pupil[AS_surf,1] = np.arctan(-stop_radius/dist_right_of_AS)

# y_tmp, u_tmp, z_sag_tmp, _ = trace_tangential_ray(y_pupil[AS_surf,0:2], u_pupil[AS_surf,0:2], lens_sequence, surf_start=AS_surf, forward=True)
# y_tmp, u_tmp, z_sag_tmp, _ = trace_tangential_ray(y_pupil[AS_surf,0:2], u_pupil[AS_surf,0:2], lens_sequence, surf_start=AS_surf, forward=True)
y_tmp, u_tmp, z_sag_tmp, _ = trace_tangential_ray([stop_radius, stop_radius, stop_radius], [0, np.arctan(-stop_radius/dist_right_of_AS/5.0), np.arctan(-stop_radius/dist_right_of_AS/10.0)], lens_sequence, surf_start=AS_surf, forward=True)

y_tmp[0:AS_surf,0:2] = y_pupil[0:AS_surf,:]
u_tmp[0:AS_surf,0:2] = u_pupil[0:AS_surf,:]
fig = plot_ray(zdist, y_tmp[:,0], fig, z_sag_tmp[:,0], color="g")
fig = plot_ray(zdist, y_tmp[:,1], fig, z_sag_tmp[:,1], color="g")
fig = plot_ray(zdist, y_tmp[:,2], fig, z_sag_tmp[:,2], color="g")


# Intersection point of the ray segments behind the last lens element
y1 = y_tmp[-2,0] 
u1 = u_tmp[-2,0]
y2 = y_tmp[-2,1]
u2 = u_tmp[-2,1]
y3 = y_tmp[-2,2]
u3 = u_tmp[-2,2]
z_int, y_int = intersection_line_segments(y1, u1, y2, u2, zdist[-2])
ray1 = [y_int, y_tmp[-2,0]]
ray2 = [y_int, y_tmp[-2,1]]
zcoord = [z_int, zdist[-2]]
fig.axes[0].plot(zcoord, ray1, "-o", color="magenta")
fig.axes[0].plot(zcoord, ray2, "-o", color="magenta")
print("y1=", y1, "y2=", y2)
print("u1=", u1, "u2=", u2)
print("z_int=", z_int, "y_int=", y_int)
print("zdist[-1]=", zdist[-2])

# Exit pupil location (measured from the image plane)
XPL = z_int - zdist[-1]
# Exit pupil diameter
XPD = 2*np.abs(y_int)

z_int2, y_int2 = intersection_line_segments(y1, u1, y3, u3, zdist[-2])
ray3 = [y_int2, y_tmp[-2,2]]
fig.axes[0].plot([z_int2, zdist[-2]], ray3, "-o", color="magenta")
print("z_int2=", z_int2, "y_int2=", y_int2)

# The heights of the outermost rays at each surface determine its clear aperture radius.
heights = np.zeros(num_surfs)
heights[0] = 0
for s in range(1, num_surfs):
    for f in range(num_fields):
        for r in [0,num_rays-1]: # consider only outermost rays
            hs = np.abs(y[s,r,f])
            if (hs > heights[s]): 
                heights[s] = hs
    # print(f"heights[{s}] = {heights[s]}")

# Calculate the back focal length BFL, effective focal length EFL, back image distance BID, and total track length TTL.
# To this end, trace a horizontal ray coming from infinity.
y_inf = np.zeros(num_surfs+1)
u_inf = np.zeros(num_surfs)
y_inf[0] = max_obj_height
u_inf[0] = 0.0
y_inf[1] = y_inf[0] # trivial transfer
for s in range(1, num_surfs):
    # Replacing u -> tan(u) in the paraxial ray tracing equations leads to results 
    # for EFL and BFL matching perfectly with Zemax Optics Studio.
    u_inf[s] = np.arctan((n[s-1]/n[s])*np.tan(u_inf[s-1]) - phi[s]*y_inf[s]/n[s])
    if s < num_surfs-1:
        y_inf[s+1] = y_inf[s] + np.tan(u_inf[s])*t[s]

BFL = - y_inf[num_surfs-2] / np.tan(u_inf[num_surfs-1])
EFL = BFL - (y_inf[0] - y_inf[num_surfs-2])/np.tan(u_inf[num_surfs-1])

# Calculate the BID from the intersection of the marginal rays of the on-axis ray bundle.
BID = (y[num_surfs-2,num_rays-1,0] - y[num_surfs-2,0,0])/(np.tan(u[num_surfs-1,0,0]) - np.tan(u[num_surfs-1,num_rays-1,0]))
TTL = np.sum(t[1:num_surfs])

# Calculate the image space numerical aperture from the angle between the marginal ray and the optical axis in image space.
ImgNA = n[num_surfs-1]*np.abs(np.sin(u[num_surfs-1, num_rays-1, num_fields-1]))
# Calculate magnification 
magnification = y[num_surfs-1, num_rays//2, 0] / obj_height[0] # As image height we take the height of the chief ray.

if True:
    # SECTION 4: Plot
    colors = ["blue", "green", "red"] if num_fields == 3 else mpl.color_sequences["tab10"][0:num_fields]
    for f in range(num_fields):
        for r in range(num_rays):
            fig = plot_ray(zdist, y[:,r,f], fig, z_sag[:,r,f], color=colors[f])
            # fig = plot_ray(t, y[:,r,f], fig, z_sag[:,r,f], color=colors[f])
            fig = plot_ray(zdist, y_cr[:,0], fig, z_sag_cr[:,0], color="k")
            fig = plot_ray(zdist, y_cr[:,1], fig, z_sag_cr[:,1], color="k")

# # horizontal incoming ray
# fig = plot_ray(zdist, y_inf[:], fig, color="m", linewidth=1)
fig, surface_segments, edge_segments = plot_spherical_surfaces(lens_sequence.vertex, R, heights, n, fig)

for l in surface_segments[4]:
    print(l[0].set_color("g"))

fig.axes[0].set_aspect("equal")
plt.ylim((-1.2*max(max_obj_height, max(heights)), 1.2*max(max_obj_height, max(heights))))
# plt.show()

# Plot the maximal clear apertures
ax = fig.axes[0]
ax.plot(lens_sequence.z_nnint[1:-1], lens_sequence.y_nnint[1:-1], '-o', color="orange", markersize=5)
ax.plot(lens_sequence.z_nnint[1:-1], -lens_sequence.y_nnint[1:-1], '-o', color="orange", markersize=5)

# SECTION 5: Calculate aberrations
# Seidel coefficient for third-order monochromatic ray aberrations
# Copy from Remillard's code
CRI = np.zeros(num_surfs)
MRI = np.zeros(num_surfs)
# chief ray for the ray bundle *at maximum object* height 
y_chief = y[:,num_rays//2,0]
u_chief = u[:,num_rays//2,0]
y_chief_test, u_chief_test, zsag_chief_test, _ = trace_tangential_ray(y_cr[0,0], -u_cr[0,0], lens_sequence)
fig = plot_ray(zdist, y_cr[:,0], fig, np.zeros_like(y_chief), color="b")
fig = plot_ray(zdist, y_chief_test[:], fig, z_sag=zsag_chief_test, color="g", dashtype="--")
# marginal ray for the *on-axis* ray bundle
y_marg = y[:,0,num_fields-1]
u_marg = u[:,0,num_fields-1]

y_marg_test, u_marg_test, zsag_mag_test, _ = trace_tangential_ray(y_cr[0,0], -u_cr[0,0]-marginal_ray_angle, lens_sequence)
fig = plot_ray(zdist, y_marg_test[:], fig, z_sag=zsag_mag_test, color="y", dashtype="--")

fig = plot_ray(zdist, y_marg[:], fig, np.zeros_like(y_marg), color="g")

L = np.zeros(num_surfs)
S1 = np.zeros(num_surfs) # spherical
S2 = np.zeros(num_surfs) # coma
S3 = np.zeros(num_surfs) # astigmatism
S4 = np.zeros(num_surfs) # field curvature
S5 = np.zeros(num_surfs) # distortion
PetzSum = 0.0
for i in range(1,num_surfs):
    MRI[i] = n[i]*(y_marg[i]/R[i] + np.tan(u_marg[i])) # marginal ray "invariant" for the *on-axis* ray bundle (an invariant at a refracting surface but not under propagation)
    CRI[i] = n[i]*(y_chief[i]/R[i] + np.tan(u_chief[i])) # chief ray for the ray bundle *at maximum object* height 
    L[i] = n[i-1]*(y_marg[i]*np.tan(u_chief[i-1]) - y_chief[i]*np.tan(u_marg[i-1])) # Lagrange invariant for the above two rays
    S1[i] = -MRI[i]*MRI[i]*y_marg[i]*(np.tan(u_marg[i])/n[i] - np.tan(u_marg[i-1])/n[i-1])
    S2[i] = S1[i]*CRI[i]/MRI[i]
    S3[i] = S2[i]*CRI[i]/MRI[i]
    S4[i] = -L[i]*L[i]*((1/n[i]) - (1/n[i-1]))/R[i]
    S5[i] = -(S3[i] + S4[i])*CRI[i]/MRI[i]
    PetzSum += (1/R[i])*((1/n[i]) - (1/n[i-1]))

S1sum = np.sum(S1); S2sum = np.sum(S2); S3sum = np.sum(S3); S4sum = np.sum(S4); S5sum = np.sum(S5)
PetzvalRadius = 1.0 / PetzSum

print("MRI=", MRI)
print("S1=", S1)

# # Test
# y_chief_test, u_chief_test, _, _ = trace_tangential_ray(1.414,  -0.0554, lens_sequence)
# y_marg_test, u_marg_test, _, _ = trace_tangential_ray(0.0, 0.0784, lens_sequence)
# S1, S2, S3, S4, S5, S1sum, S2sum, S3sum, S4sum, S5sum, PetzvalRadius = Seidel3rd_aberrations(y_chief_test, u_chief_test, y_marg_test, u_marg_test, lens_sequence)
# #

# SECTION 6: Output 
with open("lens_report.txt", "w") as fh:
    header = str("#"+str("%12s"*2)+str("%22s"*3)+str("%26s"*2)+"\n")%("Surface","Stop Flag", "Radius  [mm]", "Thickness [mm]", "n_d", "Abbe value V_d", "Clear Semidiameter [mm]")
    header+= str("# "+str("="*(12*2+22*3+26*2)))
    print(header, file=fh)
    for s in range(num_surfs):
        print("%12d %12d %21.4e %21.6f %21.6f %25.6f %25.5f"%(s, stop_flag[s], R[s], t[s], n[s], Vd[s], heights[s]), file=fh)
    print(" ", file=fh)
    print(f"Chief ray launch angles:", file=fh)
    for f in range(num_fields):
        print(f"\t\t FIELD {f} u0 = {-u_cr[0,f]} radians  -> y0 = {y_cr[0,f]} mm", file=fh)
    print(f"Effective Focal Length EFL = {EFL} mm", file=fh)        
    print(f"Back Focal Length BFL = {BFL} mm", file=fh)
    print(f"Total Track Length TTL = {TTL} mm", file=fh)      
    print(f"Image Space NA = {ImgNA}", file=fh)
    print(f"Object Space NA = {ObjNA}", file=fh)
    print(f"Stop Radius = {stop_radius}", file=fh)
    print(f"magnification = {magnification}", file=fh)    
    print(f"Entrance pupil diameter ENPD = {EPD} mm", file=fh)
    print(f"Entrance pupil position ENPP = {EPL} mm (measured from front vertex)", file=fh)    
    print(f"Exit pupil diameter XPD = {XPD} mm", file=fh)
    print(f"Exit pupil position XPP = {XPL} mm (measured from back vertex)", file=fh)
    print(f"Back Image Distance BID = {BID} mm", file=fh)
    print(f"Field of view FOV = {FOV}", file=fh)    
    print(f"Marginal Ray Angle = {marginal_ray_angle} rad = {marginal_ray_angle*360/(2*np.pi)} degrees", file=fh)    
    print(f"Violation of Abbe sine condition: eps = {ObjNA - ImgNA / np.abs(magnification)}",file=fh)
    print(" ", file=fh)
    print("Aberrations:", file=fh)
    print(" ", file=fh)
    print("Petzval radius         :", PetzvalRadius, file=fh)
    print("Optical invariant      :", L[1], L[num_surfs//2], L[num_surfs-1], file=fh)
    print(" ", file=fh)
    print("Seidel coefficients for third-order monochromatic ray aberrations (in units of mm)", file=fh)
    print(str("%8s" + " %16s "*5)%("Surf", "SPHA S1", "COMA S2", "ASTI S3", "FCUR S4", "DIST S5"), file=fh)
    for i in range(1, num_surfs):
        print(str("%8d" + " %16.8f "*5)%(i, S1[i], S2[i], S3[i], S4[i], S5[i]), file=fh)
    print(str("%8s" + " %16.8f "*5)%("TOT ", S1sum, S2sum, S3sum, S4sum, S5sum), file=fh)
    print(" ", file=fh)
    print("Seidel coefficients (in units of waves)", file=fh)

print("DONE")