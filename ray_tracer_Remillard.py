"""
Ray tracer in paraxial approximation for finite conjugate system.

Based on Stephen Remillard's ray tracing code:
https://www.youtube.com/watch?v=5962dgvPZCk
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["lines.linewidth"] = 1
    
def plot_ray(dists, ys, fig=None, color="red", linewidth=1):

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

    ax.plot(zz, yy, "-", color=color, linewidth=linewidth)

    if dists[0] > 600:
        ax.set_xlim(-10, np.sum(dists[1:]))
    
    return fig


# SECTION 1:
# User input: lens prescription file, field of view, F/# and wavelength.
# Load txt file, determine the surface powers and locate the surface which 
# is the aperture stop. Make sure there is only one aperture stop. 

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

t[0] = lens_layout[0,3]
AS_surf = 0
for s in range(0, num_surfs):
    stop_flag[s] = lens_layout[s,1]
    R[s] = lens_layout[s, 2] 
    t[s] = lens_layout[s, 3]
    n[s] = lens_layout[s, 4]
    V_d[s] = lens_layout[s, 5]
    if s  > 0:
        phi[s] = (n[s] - n[s-1]) / R[s]  # surface power 

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
# on axis, at 70% object height and at full object height.
obj_height = [max_obj_height, max_obj_height / np.sqrt(2.0), 0.0]
num_fields = len(obj_height)

y_cr = np.zeros((AS_surf+1, num_fields))
u_cr = np.zeros((AS_surf+1, num_fields))

for f in range(num_fields):
    print("field=", f)
    y_cr[AS_surf,f] = 0.0
    du = 1e-6
    u_cr[AS_surf-1,f] = -0.003 - du

    CHIEF_RAY_FOUND = False
    while(not CHIEF_RAY_FOUND):
        u_cr[AS_surf-1,f] += du    
        y_cr[AS_surf-1,f] = y_cr[AS_surf,f] + np.tan(u_cr[AS_surf-1,f])*t[AS_surf-1]
        for s in range(AS_surf-1, 0, -1):        
            u_cr[s-1,f] = (n[s] / n[s-1])*u_cr[s,f] - phi[s]*y_cr[s,f]/n[s-1]
            y_cr[s-1,f] = y_cr[s,f] + np.tan(u_cr[s-1,f])*t[s-1]

        CHIEF_RAY_FOUND =  np.isclose(y_cr[0,f], obj_height[f], atol=1e-4)    

        # print("u_cr=", u_cr[AS_surf], "y_cr[0]=", y_cr[0])

print(f"Chief ray launch angles:")
for f in range(num_fields):
    print(f"\t\t FIELD {f} u0 = {-u_cr[1,f]}")
EPL = obj_height[0]/np.tan(u_cr[0,0]) - t[0]
# for f in range(num_fields-1):
#     print("EPL=", obj_height[f]/np.tan(u_cr[1,f]) - t[0])
print(f"Entrance pupil location EPL = {EPL}")
print(f"Entrance pupil diameter EPD = {EPD}")
marginal_ray_angle = np.arctan((EPD/2.0)/(EPL+t[0]))
print(f"Marginal Ray Angle = {marginal_ray_angle} rad = {marginal_ray_angle*360/(2*np.pi)} degrees")
ObjNA = n[0]*np.sin(marginal_ray_angle)
print(f"Object-side NA = {ObjNA}")
FOV = np.arctan((obj_height[0]-y_cr[1,0])/t[0])
print(f"Field of view FOV = {FOV}")

y_tmp = np.zeros((len(t)+1, num_fields))
y_tmp[0:len(y_cr[:,0]), :] = y_cr[:,:]

# Plot the chief rays
for f in range(num_fields):
    if f==0:
        fig = plot_ray(t, y_tmp[:,f], color="orange", linewidth=2)
    else:
        fig = plot_ray(t, y_tmp[:,f], fig, color="orange", linewidth=2)

# plt.show()

# SECTION 3: Trace "fields" of height [obj_hgt, obj_hgt / sqrt(2), 0]
# with a cone of rays around each chief angle. For half the opening angle of the 
# cone of rays we choose the marginal ray angle.
nr = 5 # number of rays in a ray bundle for a given field
y = np.zeros((num_surfs+1, nr, num_fields))
u = np.zeros((num_surfs, nr, num_fields))

for f in range(num_fields):
    y[0,:,f] = obj_height[f]
    dtheta = 2*marginal_ray_angle/nr
    u[0,:,f] = np.array([-u_cr[0,f] + (k-nr//2)*dtheta for k in range(nr)])

    for r in range(nr):
        y[1,r,f] = y[0,r,f] + np.tan(u[0,r,f])*t[0]
        for i in range(1, num_surfs):
            # theta = y[i]/R[i]
            # print("i=", i, "theta=", theta, "u[i-1]=", u[i-1])
            # u[i] = np.asin(n[i-1]/n[i] * np.sin(u[i-1] + theta)) - theta
            u[i,r,f] = ((n[i-1]/n[i])*u[i-1,r,f] - phi[i]*y[i,r,f]/n[i])
            y[i+1,r,f] = y[i,r,f] + np.tan(u[i,r,f])*t[i]    

# The height of the  marginal ray of the on-axis field at the aperture stop gives the stop radius.
stop_radius = y[AS_surf,nr-1,0]
print(f"Stop Radius = {stop_radius}")

# The heights of the outermost rays at each surface determine its clear aperture radius.
heights = np.zeros(num_surfs)
heights[0] = max_obj_height
for s in range(1, num_surfs):
    for f in range(num_fields):
        for r in [0,nr-1]: # consider only outermost rays
            hs = np.abs(y[s,r,f])
            if (hs > heights[s]): 
                heights[s] = hs
    print(f"heights[{s}] = {heights[s]}")

fh = open("lens_summary.txt", "w")
header = "# Surface \t Stop Flag \t Radius  [mm] \t Thickness [mm] \t n_d \t Abbe value V_d \t Clear Aperture [mm] \n"
header+= "# ================================================================================================="
print(header, file=fh)
for s in range(num_surfs):
    str = "%d \t %d \t %8.4e \t %12.6f \t %12.6f \t %12.6f \t %9.5f" % (s, stop_flag[s], R[s], t[s], n[s], V_d[s], heights[s] )
    print(str, file=fh)
fh.close()

# Calculate back focal length BFL, effective focal length EFL, back image distance BID, and total track length TTL.

# SECTION 4: Plot
colors = ["blue", "green", "red"]
for f in range(num_fields):
    for r in range(nr):
        if f==0 and r == 0:
            fig = plot_ray(t, y[:,r,f], fig, color=colors[f])
        else:
            fig = plot_ray(t, y[:,r,f], fig, color=colors[f])

plt.show()


# SECTION 5: Calculate aberrations
