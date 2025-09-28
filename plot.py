import numpy as np
import matplotlib.pyplot as plt 
from time import time


def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


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


@timer_func
def intersection_with_surface(ray_bundle, zS):
    '''
    ray_bundle : a tuple (P_intersect[0:3,0:num_rays], ravecs[0:3,0:num_rays])
    zS : float, the position of the surface orthogonal to the optical axis
    '''
    P_intersect, rayvecs = ray_bundle
    assert P_intersect.shape == rayvecs.shape
    zz = ((zS - P_intersect[2,:])/rayvecs[2,:])
    P = P_intersect.copy()
    P[2,:] = zS
    P[0:2,:] = P[0:2,:] + zz*rayvecs[0:2,:]

    return P


@timer_func
def intersection_with_surface_v2(ray_bundle, zS):
    '''
    ray_bundle : a tuple (P_intersect[0:3,0:num_rays], ravecs[0:3,0:num_rays])
    zS : float, the position of the surface orthogonal to the optical axis
    '''
    P_intersect, rayvecs = ray_bundle
    num_rays = rayvecs.shape[1]
    intersection_points = np.zeros((3, num_rays))
    for r in range(num_rays):
        zz = ((zS - P_intersect[2,r])/rayvecs[2,r])
        x_is = P_intersect[0,r] + zz*rayvecs[0,r]
        y_is = P_intersect[1,r] + zz*rayvecs[1,r]
        z_is = zS
        intersection_points[0:3,r] = np.array([x_is, y_is, z_is])

    return intersection_points

