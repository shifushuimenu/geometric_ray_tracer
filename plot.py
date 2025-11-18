import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from utils import timer_func

from typing import Iterable

__all__ = ["plot_ray", "plot_surfaces"]

def plot_ray(dists: Iterable, ys: Iterable, z_sag: Iterable = None, 
             fig: Figure = None, color="red", linewidth=1) -> Figure:

    params = {"surfcolor" : "red", "ASstopcolor" : "green"}

    if fig is None:
        fig = plt.figure("fig1", figsize=(6,6), layout="tight")
        ax = fig.subplots(1,1)
        # plot optical axis
        ax.axhline(y=0, color="k", linestyle="--")
        # draw paraxial lens surfaces, i.e. the plane through the vertex orthogonal to the optical axis
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
    
    fig.axes[0].set_aspect("equal")
    return fig


def plot_surfaces(dists: Iterable, Rs: Iterable, clear_aperture: Iterable, ns: Iterable, 
                  fig: Figure = None) -> Figure:
    """
    Plot spherical lens surfaces up to their clear apertures.

    Mark refractive materials with colors, differentiating between lens elements with 
    positive and negative refractive index. The hue of the color indicates the absolute 
    magnitude of the refractive index. 
    """

    # idenfity singlets, doublets and triplets so that the clear apertures of their
    # surfaces can be combined into a lens
    n_air = 1.0 # 1.000302
    nmat = np.invert((np.isclose(ns, n_air, atol=1e-6)))  # Which segements are materials other than air ?
    ymax = clear_aperture.copy()   
    y_ = 0.0; multiplet_elements = 0
    for i in range(1,len(Rs)):        
        if nmat[i]:
            if i <= len(Rs)-2:
                y_ = max(y_, clear_aperture[i], clear_aperture[i+1])
            else:
                y_ = max(y_, clear_aperture[i])
            multiplet_elements += 1
        else:
            ymax[i-multiplet_elements:i+1] = y_
            # reset
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
                # plot the curved lens surface above and below the optical axis
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
    
    fig.axes[0].set_aspect("equal")
    return fig