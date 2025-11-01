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

from trace_ray import LensSequence, trace_tangential_ray

mpl.rcParams["lines.linewidth"] = 1
    
def plot_ray(dists, ys, fig=None, z_sag=None, color="red", linewidth=1):

    params = {"surfcolor" : "red"}

    if fig is None:
        fig = plt.figure("fig1", figsize=(6,6), layout="tight")
        ax = fig.subplots(1,1)
        # plot optical axis
        ax.axhline(y=0, color="k", linestyle="--")
        # draw paraxial lens surfaces
        l=-dists[0] # object distance is negative, lens system starts at z=0
        ax.axvline(x=l, color=params["surfcolor"], linewidth=0.5)
        ax.text(l-0.4, 0.0, "OBJECT", rotation=90, va="center")
        for i in range(0, len(dists)):        
            l += dists[i]
            ax.axvline(x=l, color=params["surfcolor"])
        ax.text(l-0.2, 0.0, "IMAGE", rotation=90, va="center")
    else:
        ax = fig.axes[0]
    l=-dists[0]
    zz = [l]           
    yy = [ys[0]]     
    for i in range(0, len(dists)):        
        l += dists[i]
        zz.append(l)
        yy.append(ys[i+1])

    zz = np.array(zz)
    yy = np.array(yy)
    if z_sag is None:
        z_sag = np.zeros_like(zz)
    ax.plot(zz+z_sag, yy, "-o", color=color, linewidth=linewidth)

    if dists[0] > 600:
        ax.set_xlim(-10, np.sum(dists[1:]))
    
    return fig


def plot_surfaces(dists, Rs, heights, ns, fig=None):
    """Plot spherical lens surfaces up to their clear apertures, which is given by heights[:]."""

    # idenfity singlets, doublets and triplets so that the clear apertures of their
    # surfaces can be combined into a lens
    n_air = 1.00
    nmat = np.invert((np.isclose(ns, n_air, atol=1e-2)))  # Which segements are materials other than air ?
    ymax = heights.copy()   
    y_ = 0.0; multiplet_elements = 0
    for i in range(1,len(heights)):        
        if nmat[i]:
            if i <= len(heights)-2:
                y_ = max(y_, heights[i], heights[i+1])
            else:
                y_ = max(y_, heights[i])
            multiplet_elements += 1
        else:
            ymax[i-multiplet_elements:i+1] = y_
            y_ = 0.0; multiplet_elements = 0

    if fig is None:
        fig = plt.figure("fig1", figsize=(6,6), layout="tight")
        ax = fig.subplots(1,1)
        # plot optical axis
        ax.axhline(y=0, color="k", linestyle="--")

    fig.axes[0].axis([-dists[0], np.sum(dists[1:]), -max(ymax)-2.0, max(ymax)+2.0])
    vertex = 0
    lens_edges = []
    for i in range(1, len(dists)):
        if not np.isinf(Rs[i]):
            zmax = np.abs(Rs[i])*(1.0-np.sqrt(1.0-(ymax[i]/Rs[i])**2))
            zz = np.linspace(0, zmax, 2000)
            for pm in [+1,-1]:
                yy = pm*np.sqrt(Rs[i]**2-(np.abs(Rs[i])-zz)**2)
                fig.axes[0].plot(vertex+np.sign(Rs[i])*zz, yy, color="k", linewidth=2)
                if pm == +1:
                    lens_edges.append([vertex+np.sign(Rs[i])*zz[-1], yy[-1]])
            if not nmat[i]:
                # print("lens_edges=", lens_edges)
                for pm in [+1,-1]:
                    fig.axes[0].plot([e[0] for e in lens_edges],
                                     [pm*e[1] for e in lens_edges], 
                                        color="k", linewidth=2)
                lens_edges = []
        vertex += dists[i]    
    
    return fig


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
R = np.zeros(num_surfs, dtype=np.float64)
t = np.zeros(num_surfs, dtype=np.float64)
n = np.zeros(num_surfs, dtype=np.float64)
Vd = np.zeros(num_surfs, dtype=np.float64)
phi = np.zeros(num_surfs, dtype=np.float64)

t[0] = lens_layout[0,3]
AS_surf = 0
for s in range(0, num_surfs):
    stop_flag[s] = lens_layout[s,1]
    R[s] = lens_layout[s, 2] 
    t[s] = lens_layout[s, 3]
    n[s] = lens_layout[s, 4]
    Vd[s] = lens_layout[s, 5]
    if s  > 0:
        phi[s] = (n[s] - n[s-1]) / R[s]  # surface power 

# zdist[s] is the total distance of surface s from the first vertex of the lens system.
# zdist[0] < 0 is the object distance.
zdist = np.zeros(num_surfs+1)
zdist[0] = -t[0]
zdist[1] = 0
zdist[2:] = np.cumsum(t[1:])


# special case: first surface is the aperture stop
if stop_flag[1] == 1:
    AS_surf = 1
    pass # MISSING

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

y_cr = np.zeros((AS_surf+1, num_fields))
u_cr = np.zeros((AS_surf+1, num_fields))

z_sag_cr = np.zeros((AS_surf+1, num_fields))

for f in range(num_fields):
    print("field=", f)
    # By definition, at the aperture stop, the chief ray intersects the optical axis.
    y_cr[AS_surf,f] = 0.0

    u_max = 0.4 #np.pi/2.0 - 0.01  # 0.4 # radians 
    u_min = -0.4 #-np.pi/2.0 + 0.01 # -0.4 #    
    CHIEF_RAY_FOUND = False
    print("determining chief ray launch angle")
    while(not CHIEF_RAY_FOUND): # and y_cr[0,f] < obj_height[f]):
        # update launch angle using bisection search
        # It is assumed that increasing the chief ray launch angle will 
        # monotonically increase its height in object space.
        u_middle = 0.5*(u_max + u_min)
        print("u_middle=", u_middle)
        u_cr[AS_surf-1,f] = u_middle   
        y_cr[AS_surf-1,f] = y_cr[AS_surf,f] + np.tan(u_cr[AS_surf-1,f])*t[AS_surf-1]
        for i in range(AS_surf-1, 0, -1):      
            if np.isinf(R[i]) or not SAG:
                u_cr[i-1,f] = np.arctan((n[i]/n[i-1])*np.tan(u_cr[i,f]) - phi[i]*y_cr[i,f]/n[i-1])
                y_cr[i-1,f] = y_cr[i,f] + np.tan(u_cr[i-1,f])*t[i-1]

            # take surface sag into account
            else:
                y0 = y_cr[i,f]
                u0 = u_cr[i,f]
                # intersection with spherical surface
                # When ray tracing from right to left:
                #   (A) Indices of refraction (before, after) are exchanged.
                #   (B) The radius of curvature is inverted and all quantities are computed as if for that problem.
                #   (C) The surface sag thus computed needs to be inverted. 
                R_ = (-1)*R[i] # modification (B)
                sgnR = np.sign(R_)
                tanu0 = np.tan(u0)
                Delta = R_**2 - 2*y0*tanu0*R_ - y0**2
                assert Delta > 0, "Delta < 0, %f: No intersection point found with surface nr %d"%(Delta, i)
                zp = (R_ - y0*tanu0 - sgnR*np.sqrt(Delta))/(1 + tanu0**2)                
                yp = y0 + tanu0*zp 
                theta = np.arctan(sgnR*yp/(R_-zp))
                u_prime = sgnR*(np.arcsin(n[i]/n[i-1]*np.sin(theta + sgnR*u0)) - theta)  # modification (A)         

                z_sag_cr[i,f] = (-1)*zp # modification (C)
                y_cr[i,f] = yp # reset to value at intersection point
                u_cr[i-1,f] = u_prime
                y_cr[i-1,f] = yp + np.tan(u_cr[i-1,f])*(t[i-1] - zp)

        # criterion whether chief ray has been found
        CHIEF_RAY_FOUND = np.isclose(y_cr[0,f], obj_height[f], atol=1e-6)
        # update bracketing interval for binary search
        if y_cr[0,f] < obj_height[f]:
            print(f"y_cr[0,{f}] < obj_height[{f}]")
            print(f"u_min = {u_min} and u_max = {u_max}")
            u_min = u_middle 
        else:
            print(f"y_cr[0,{f}] > obj_height[{f}]")
            print(f"u_min = {u_min} and u_max = {u_max}")
            u_max = u_middle

# entrance pupil location (measured from the vertex of the first surface)
EPL = obj_height[0]/np.tan(u_cr[0,0]) - t[0]
# for f in range(num_fields-1):
#     print("EPL=", obj_height[f]/np.tan(u_cr[1,f]) - t[0])

marginal_ray_angle = np.arctan((EPD/2.0)/(EPL+t[0]))
ObjNA = n[0]*np.sin(marginal_ray_angle)
FOV = np.arctan((obj_height[0]-y_cr[1,0])/t[0])

y_tmp = np.zeros((len(t)+1, num_fields))
y_tmp[0:len(y_cr[:,0]),:] = y_cr[:,:]
z_sag_tmp = np.zeros((len(t)+1, num_fields))
z_sag_tmp[0:len(z_sag_cr[:,0]),:] = z_sag_cr[:,:]

# Plot the chief rays
for f in range(num_fields):
    if f==0:
        fig = plot_ray(t, y_tmp[:,f], z_sag=z_sag_tmp[:,f], color="orange", linewidth=4)
    else:
        fig = plot_ray(t, y_tmp[:,f], fig, z_sag=z_sag_tmp[:,f], color="orange", linewidth=4)

# plt.show()

# SECTION 3: Trace "fields" of height [obj_hgt, obj_hgt / sqrt(2), 0]
# with a cone of rays around each chief ray launch angle. For half the opening angle of the 
# cone of rays we choose the marginal ray angle.
num_rays = 55 # number of rays in a ray bundle for a given field
assert num_rays % 2 == 1
y = np.zeros((num_surfs+1, num_rays, num_fields))
u = np.zeros((num_surfs, num_rays, num_fields))

z_sag = np.zeros((num_surfs+1, num_rays, num_fields))
y_intersection = np.zeros((num_surfs, num_rays, num_fields))

t1 = time()
# loop over fields
for f in range(num_fields):
    y[0,:,f] = obj_height[f]
    dtheta = 2*marginal_ray_angle/num_rays
    # k=0 and k=num_rays-1 are the marginal rays, k=num_rays//2 is the chief ray of the ray bundle.
    u[0,:,f] = np.array([-u_cr[0,f] + (k-num_rays//2)*dtheta for k in range(num_rays)])

    # loop over rays in a ray bundle
    for r in range(num_rays):
        y[1,r,f] = y[0,r,f] + np.tan(u[0,r,f])*t[0]
        for i in range(1, num_surfs):
            if np.isinf(R[i]) or not SAG:
                u[i,r,f] = np.arctan((n[i-1]/n[i])*np.tan(u[i-1,r,f]) - phi[i]*y[i,r,f]/n[i])
                y[i+1,r,f] = y[i,r,f] + np.tan(u[i,r,f])*t[i]   

            # take surface sag into account
            else:                
                y0 = y[i,r,f]
                u0 = u[i-1,r,f]
                # intersection with spherical surface
                sgnR = np.sign(R[i]) # The formula depends on the sign of the radius of curvature.
                tanu0 = np.tan(u0)
                Delta = R[i]**2 - 2*y0*tanu0*R[i] - y0**2
                assert Delta > 0, "Delta < 0, %f"%(Delta)  
                zp = (R[i] - y0*tanu0 - sgnR*np.sqrt(Delta))/(1 + tanu0**2)
                yp = y0 + tanu0*zp
                theta = np.arctan(sgnR*yp/(R[i]-zp))
                u_prime = sgnR*(np.arcsin(n[i-1]/n[i]*np.sin(theta + sgnR*u0)) - theta)

                z_sag[i,r,f] = zp
                # y_intersection[i,r,f] = yp
                y[i,r,f] = yp # reset to value at intersection point
                u[i,r,f] = u_prime 
                y[i+1,r,f] = yp + np.tan(u_prime)*(t[i] - zp)

t2 = time()
print("elapsed =", t2 - t1)

lens_sequence = LensSequence(
    num_surfs, 
    AS_surf,
    SAG,
    "mm",
    R[:],
    t[:],
    n[:],
    Vd[:],
    phi[:],
)

y_test, u_test, z_sag_test, y_vertexplane_test = trace_tangential_ray(y[0,:,:], u[0,:,:], lens_sequence, surf_start=0)
y_test2, u_test2, z_sag_test2, y_vertexplane_test2 = trace_tangential_ray(y_vertexplane_test[5,:,:], u[4,:,:], lens_sequence, surf_start=5)
print("y[1,0,0]=", y[1,0,0])
print("u[1,0,0]=", u[1,0,0])
# exit(1)
print("u_test=", u_test[0:,0,1])
print("u = ", u[0:,0,1])
print("y_test=", y_test[0:,0,1])
print("y = ", y[0:,0,1])
# assert np.isclose(y_test, y).all()
# exit(1)

# plt.show()
# generate_ray_fan_plot(y, AS_surf, 1.0, num_surfs)
# exit(1)

# The height of the  marginal ray of the on-axis field at the aperture stop gives the stop radius.
stop_radius = np.abs(y[AS_surf,num_rays-1,0])

# ===============================================================
# Locate the position of the *exit pupil* and its diameter.
# ===============================================================
# Trace two rays from the edge of the aperture stop towards the image side. 
# Their first intersection point as viewed from the image side gives the edge 
# of the exit pupil since it is the image of the aperture stop. The exit pupil
# can be a virtual image.
# (y_pupil, u_pupil) is the ray that goes through the edge of the aperture stop and through 
# the edge of the exit pupil.
y_pupil = np.zeros((num_surfs+1,2))
u_pupil = np.zeros((num_surfs,2))

# Ray 1 is just a copy of the on-axis marginal ray.
y_pupil[AS_surf:,0] = y[AS_surf:,num_rays-1,0]
u_pupil[AS_surf:,0] = u[AS_surf:,num_rays-1,0]
# Ray 2 goes through the center of the next lens element. In the paraxial approximation, 
# any other ray would be just as good. 
y_pupil[AS_surf,1] = stop_radius
if np.isclose(t[AS_surf], 0):
    dist_left_of_AS = t[AS_surf+1]
else:
    dist_left_of_AS = t[AS_surf]
u_pupil[AS_surf,1] = np.arctan(-stop_radius/dist_left_of_AS)
# First check whether the exit pupil is virtual, i.e. ray 1 and ray 2 
# intersect to the left of the first lens element behind the aperture stop.
z_intersection = - stop_radius/(np.tan(u_pupil[AS_surf+1,0]) - np.tan(u_pupil[AS_surf,1]))
if z_intersection < 0:
    print(f"exit pupil is virtual, z={z_intersection}")
    print(f"location of the aperture stop {zdist[AS_surf]}")
    XPL = z_intersection + zdist[AS_surf]
    XP_radius = np.tan(u_pupil[AS_surf,1]) * z_intersection
    print(f"exit pupil location XPL={XPL}")
    print(f"exit pupil semidiameter XP_radius={XP_radius}")
else:
    print(f"exit pupil is a real image of the aperture stop")
    y_tmp, u_tmp, z_sag_tmp, _ = trace_tangential_ray(y_pupil, u_pupil, lens_sequence, surf_start=AS_surf)
    # Ray 2 needs to be traced paraxially (!) so that its value behind the last lens element is known.
    # for s in range(AS_surf+1, num_surfs+1, 1):
    #     pass
        

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
y_inf[1] = max_obj_height # trivial transfer
for s in range(1, num_surfs):
    # Replacing u -> tan(u) in the paraxial ray tracing equations leads to results 
    # for EFL and BFL matching perfectly with Zemax Optics Studio.
    u_inf[s] = np.arctan((n[s-1]/n[s])*np.tan(u_inf[s-1]) - phi[s]*y_inf[s]/n[s])
    y_inf[s+1] = y_inf[s] + np.tan(u_inf[s])*t[s]

BFL = - y_inf[num_surfs-2] / np.tan(u_inf[num_surfs-1])
EFL = BFL - (y_inf[0] - y_inf[num_surfs-2])/np.tan(u_inf[num_surfs-1])

# Calculate the BID from the intersection of the marginal rays of the on-axis ray bundle.
BID = (y[num_surfs-2,num_rays-1,0] - y[num_surfs-2,0,0])/(np.tan(u[num_surfs-1,0,0]) - np.tan(u[num_surfs-1,num_rays-1,0]))
TTL = np.sum(t[1:num_surfs])

# Calculate the image space numerical aperture from the angle between the marginal ray and the optical axis in image space.
ImgNA = n[num_surfs-1]*np.abs(np.sin(u[num_surfs-1, num_rays-1, num_fields-1]))
# Calculate magnification 
magnification = obj_height[0] / y[num_surfs, num_rays//2, 0] # As image height we take the height of the chief ray.

# SECTION 4: Plot
colors = ["blue", "green", "red"] if num_fields == 3 else mpl.color_sequences["tab10"][0:num_fields]
for f in range(num_fields):
    for r in range(num_rays):
        fig = plot_ray(t, y_test[:,r,f], fig, z_sag_test[:,r,f], color=colors[f])
        fig = plot_ray(t, y_test2[:,r,f], fig, z_sag_test2[:,r,f], color=colors[f])
        # fig = plot_ray(t, y[:,r,f], fig, z_sag[:,r,f], color=colors[f])
        fig = plot_ray(t, y_tmp[:,0], fig, z_sag_tmp[:,0], color="k")
        fig = plot_ray(t, y_tmp[:,1], fig, z_sag_tmp[:,1], color="k")

# horizontal incoming ray
fig = plot_ray(t, y_inf[:], fig, color="m", linewidth=1)
plot_surfaces(t, R, heights, n, fig)
plt.ylim((-1.2*max(max_obj_height, max(heights)), 1.2*max(max_obj_height, max(heights))))
plt.show()

# SECTION 5: Calculate aberrations
# Seidel coefficient for third-order monochromatic ray aberrations
# Copy from Remillard's code
CRI = np.zeros(num_surfs)
MRI = np.zeros(num_surfs)
L = np.zeros(num_surfs)
S1 = np.zeros(num_surfs)
S2 = np.zeros(num_surfs)
S3 = np.zeros(num_surfs)
S4 = np.zeros(num_surfs)
S5 = np.zeros(num_surfs)
PetzSum = 0.0
for i in range(1,num_surfs):
    MRI[i] = n[i-1]*(y[i,0,num_fields-1]/R[i] + np.tan(u[i-1,0,num_fields-1])) # marginal ray for the *on-axis* ray bundle
    CRI[i] = n[i-1]*(y[i,num_rays//2,0]/R[i] + np.tan(u[i-1,num_rays//2,0])) # chief ray for the ray bundle *at maximum object* height 
    L[i] = n[i-1]*(y[i,0,num_fields-1]*np.tan(u[i-1,num_rays//2,0]) - y[i,num_rays//2,0]*np.tan(u[i-1,0,num_fields-1])) # Lagrange invariant for the above two rays
    S1[i] = -MRI[i]*MRI[i]*y[i,0,num_fields-1]*(np.tan(u[i,0,num_fields-1])/n[i] - np.tan(u[i-1,0,num_fields-1])/n[i-1])
    S2[i] = S1[i]*CRI[i]/MRI[i]
    S3[i] = S2[i]*CRI[i]/MRI[i]
    S4[i] = -L[i]*L[i]*((1/n[i]) - (1/n[i-1]))/R[i]
    S5[i] = (S3[i] + S4[i])*CRI[i]/MRI[i]
    PetzSum += (1/R[i])*((1/n[i]) - (1/n[i-1]))

S1sum = np.sum(S1); S2sum = np.sum(S2); S3sum = np.sum(S3); S4sum = np.sum(S4); S5sum = np.sum(S5)
PetzvalRadius = 1.0 / PetzSum

# SECTION 6: Output 
with open("lens_summary.txt", "w") as fh:
    header = str("#"+str("%12s"*2)+str("%22s"*3)+str("%26s"*2)+"\n")%("Surface","Stop Flag", "Radius  [mm]", "Thickness [mm]", "n_d", "Abbe value V_d", "Clear Semidiameter [mm]")
    header+= str("# "+str("="*(12*2+22*3+26*2)))
    print(header, file=fh)
    for s in range(num_surfs):
        print("%12d %12d %21.4e %21.6f %21.6f %25.6f %25.5f"%(s, stop_flag[s], R[s], t[s], n[s], Vd[s], heights[s]), file=fh)
    print(" ", file=fh)
    print(f"Chief ray launch angles:", file=fh)
    for f in range(num_fields):
        print(f"\t\t FIELD {f} u0 = {-u_cr[0,f]*360/(2*np.pi)} degrees  -> y0 = {y_cr[0,f]} mm", file=fh)
    print(f"Entrance pupil position ENPP = {EPL} mm", file=fh)
    print(f"Entrance pupil diameter ENPD = {EPD} mm", file=fh)
    print(f"Marginal Ray Angle = {marginal_ray_angle} rad = {marginal_ray_angle*360/(2*np.pi)} degrees", file=fh)
    print(f"Object Space NA = {ObjNA}", file=fh)
    print(f"Image Space NA = {ImgNA}", file=fh)
    print(f"magnification = {magnification}", file=fh)
    print(f"Violation of Abbe sine condition: eps = {ObjNA - ImgNA / np.abs(magnification)}",file=fh)
    print(f"Field of view FOV = {FOV}", file=fh)
    print(f"Stop Radius = {stop_radius}", file=fh)
    print(f"Back Focal Length BFL = {BFL} mm", file=fh)
    print(f"Effective Focal Length EFL = {EFL} mm", file=fh)
    print(f"Back Image Distance BID = {BID} mm", file=fh)
    print(f"Total Track Length TTL = {TTL} mm", file=fh)
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