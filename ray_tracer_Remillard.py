"""
Ray tracer in paraxial approximation for finite conjugate system.

Based on Stephen Remillard's ray tracing code:
https://www.youtube.com/watch?v=5962dgvPZCk
"""
import numpy as np
import sys
import matplotlib.pyplot as plt

def plot_ray_bundle(dists, ys):

    nr = ys.shape[1]
    #assert len(dists) == ys.shape[0]
    params = {"surfcolor" : "red"}

    fig = plt.figure("fig1", figsize=(6,6), layout="tight")
    ax = fig.subplots(1,1)
    # # object distance is negative, the lens system starts at z=0 
    # if (dists[0] < 1000):
    #     ax.axvline(x=-dists[0], color=params["surfcolor"])
    #     ax.text(-dists[0]-0.2, 0.0, "OBJECT", rotation=90, va="center")
    l=0
    ax.axvline(x=l, color=params["surfcolor"])
    ax.text(-0.2, 0.0, "OBJECT", rotation=90, va="center")
    for i in range(0, len(dists)):        
        l += dists[i]
        ax.axvline(x=l, color=params["surfcolor"])    

    for r in range(nr):
        l=0
        xx = [l]           
        yy = [ys[0,r]]     
        for i in range(0, len(dists)):        
            l += dists[i]
            xx.append(l)
            yy.append(ys[i+1,r])        

        ax.plot(xx, yy, "o-")

    if dists[0] > 200:
        ax.set_xlim((np.sum(dists[0:2])-200, np.sum(dists)+10))
    
    # plot optical axis
    ax.plot(xx, 0.0*np.ones_like(xx), "--")

    plt.show()
    

def plot_ray(dists, ys, fig=None):

    params = {"surfcolor" : "red"}

    if fig is None:
        fig = plt.figure("fig1", figsize=(6,6), layout="tight")
        ax = fig.subplots(1,1)
    else:
        ax = fig.axes[0]
    # # object distance is negative, the lens system starts at z=0 
    # if (dists[0] < 1000):
    #     ax.axvline(x=-dists[0], color=params["surfcolor"])
    #     ax.text(-dists[0]-0.2, 0.0, "OBJECT", rotation=90, va="center")
    l=0
    ax.axvline(x=l, color=params["surfcolor"])
    ax.text(-0.2, 0.0, "OBJECT", rotation=90, va="center")
    for i in range(0, len(dists)):        
        l += dists[i]
        ax.axvline(x=l, color=params["surfcolor"])    

    l=0
    xx = [l]           
    yy = [ys[0]]     
    for i in range(0, min(len(ys), len(dists))-1):        
        l += dists[i]
        xx.append(l)
        yy.append(ys[i+1])        

    ax.plot(xx, yy, "o-")

    if dists[0] > 200:
        ax.set_xlim((np.sum(dists[0:2])-200, np.sum(dists)+10))
    
    # plot optical axis
    ax.plot(xx, 0.0*np.ones_like(xx), "--")

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

# SECTION 2: Calculate the chief ray piercing height on the first surface.
# The chief ray goes from the tip of the object through the center of the aperture stop.

# The specified aperture stop is assumed to be the true aperture stop, that is, all lens elements
# are taken to have inifinite radial extent such that none acts as the actual aperture stop.

# The chief ray is calculated for three object heights ("field positions" in optics jargon):
# on axis, at 70% object height and at full object height.
obj_height = [max_obj_height, max_obj_height / np.sqrt(2.0), 0.0]
num_fields = len(obj_height)

# special case: first surface is the aperture stop

y_cr = np.zeros((AS_surf+1, num_fields))
u_cr = np.zeros((AS_surf+1, num_fields))

for f in range(num_fields):
    print("field=", f)
    y_cr[AS_surf,f] = 0.0
    du = 1e-7
    u_cr[AS_surf,f] = -0.003

    CHIEF_RAY_FOUND = False
    while(not CHIEF_RAY_FOUND):
        u_cr[AS_surf,f] += du    
        y_cr[AS_surf-1,f] = y_cr[AS_surf,f] + np.tan(u_cr[AS_surf,f])*t[AS_surf-1]
        for s in range(AS_surf-1, 0, -1):        
            u_cr[s,f] = (n[s+1] / n[s])*u_cr[s+1,f] - phi[s]*y_cr[s,f]/n[s]
            y_cr[s-1,f] = y_cr[s,f] + np.tan(u_cr[s,f])*t[s-1]

        CHIEF_RAY_FOUND =  np.isclose(y_cr[0,f], obj_height[f], atol=1e-4)    

        # print("u_cr=", u_cr[AS_surf], "y_cr[0]=", y_cr[0])

print(f"Chief ray launch angles:")
for f in range(num_fields):
    print(f"\t\t FIELD {f} u0={-u_cr[1,f]}")
EPL = obj_height[0]/np.tan(u_cr[1,0])
print(f"Entrance pupil location EPL={EPL}")
print(f"Entrance pupil diameter EPD={EPD}")
ObjNA = n[0]*np.sin(np.atan(EPD/(2*EPL)))
print(f"Object-side numerical aperture ObjNA={ObjNA}")

# Having obtained the cheif ray launch angles, propagate a cone of 
# rays around each chief angle.

for f in range(num_fields):
    if f==0:
        fig = plot_ray(t, y_cr[:,f])
    else:
        fig = plot_ray(t, y_cr[:,f], fig)
plt.show()

# SECTION 3: Trace "fields" of height [obj_hgt, obj_hgt / sqrt(2), 0].
nr = 5
y = np.zeros((num_surfs+1, nr))
u = np.zeros((num_surfs, nr))
y[0, :] = 1.0
dt = (1.0/360.0)*2*np.pi
u[0, :] = [-(nr//2)*dt + j*dt for j in range(nr)]

for r in range(nr):
    y[1,r] = y[0,r] + np.tan(u[0,r])*t[i]
    for i in range(1, num_surfs):
        # theta = y[i]/R[i]
        # print("i=", i, "theta=", theta, "u[i-1]=", u[i-1])
        # u[i] = np.asin(n[i-1]/n[i] * np.sin(u[i-1] + theta)) - theta
        u[i,r] = ((n[i-1]/n[i])*u[i-1,r] - phi[i]*y[i,r]/n[i])
        y[i+1,r] = y[i,r] + np.tan(u[i,r])*t[i]    

plot_ray_bundle(t, y)
# SECTION 4: Calculate aberrations

# SECTION 5: Plot
