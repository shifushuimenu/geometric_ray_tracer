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

R = np.zeros(num_surfs)
t = np.zeros(num_surfs)
n = np.zeros(num_surfs)
V_d = np.zeros(num_surfs)
phi = np.zeros(num_surfs)

t[0] = lens_layout[0,3]
AS_surf = 0
for i in range(0, num_surfs):
    R[i] = lens_layout[i, 2] 
    t[i] = lens_layout[i, 3]
    n[i] = lens_layout[i, 4]
    V_d[i] = lens_layout[i, 5]
    if i  > 0:
        phi[i] = (n[i] - n[i-1]) / R[i]  # surface power 

# special case: first surface is the aperture stop
if lens_layout[1,1] == 1:
    AS_surf = 1
    pass

# Find the aperture stop and verify that there is only one aperture stop.
found_AS = False
for i in range(1, num_surfs):    
    if lens_layout[i,1] == 1:
        if not found_AS:
            AS_surf = i
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
    print(f"\t\t FIELD {f} u0={-u_cr[1,f]}")
EPL = obj_height[0]/np.tan(u_cr[0,0]) - t[0]
for f in range(num_fields-1):
    print("EPL=", obj_height[f]/np.tan(u_cr[1,f]) - t[0])
print(f"Entrance pupil location EPL={EPL}")
print(f"Entrance pupil diameter EPD={EPD}")
ObjNA = n[0]*np.sin(np.atan(EPD/(2*EPL)))
print(f"Object-side NA={ObjNA}")

y_tmp = np.zeros((len(t)+1, num_fields))
y_tmp[0:len(y_cr[:,0]), :] = y_cr[:,:]

# Plot the chief rays
for f in range(num_fields):
    if f==0:
        fig = plot_ray(t, y_tmp[:,f], color="orange", linewidth=2)
    else:
        fig = plot_ray(t, y_tmp[:,f], fig, color="orange", linewidth=2)

# plt.show()


# Having obtained the chief ray launch angles, propagate a cone of 
# rays around each chief angle.



# SECTION 3: Trace "fields" of height [obj_hgt, obj_hgt / sqrt(2), 0]
# with a cone of rays around each chief angle.
nr = 5
y = np.zeros((num_surfs+1, nr, num_fields))
u = np.zeros((num_surfs, nr, num_fields))

for f in range(num_fields):
    y[0,:,f] = obj_height[f]
    dt = (0.2/360.0)*2*np.pi
    u[0,:,f] = np.array([-u_cr[0,f] - (nr//2)*dt + j*dt for j in range(nr)])

    for r in range(nr):
        y[1,r,f] = y[0,r,f] + np.tan(u[0,r,f])*t[0]
        for i in range(1, num_surfs):
            # theta = y[i]/R[i]
            # print("i=", i, "theta=", theta, "u[i-1]=", u[i-1])
            # u[i] = np.asin(n[i-1]/n[i] * np.sin(u[i-1] + theta)) - theta
            u[i,r,f] = ((n[i-1]/n[i])*u[i-1,r,f] - phi[i]*y[i,r,f]/n[i])
            y[i+1,r,f] = y[i,r,f] + np.tan(u[i,r,f])*t[i]    

colors = ["blue", "green", "red"]

for f in range(num_fields):
    for r in range(nr):
        if f==0 and r == 0:
            fig = plot_ray(t, y[:,r,f], fig, color=colors[f])
        else:
            fig = plot_ray(t, y[:,r,f], fig, color=colors[f])

plt.show()

# SECTION 4: Calculate aberrations

# SECTION 5: Plot
