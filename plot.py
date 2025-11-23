import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from utils import timer_func

from typing import Iterable, Tuple
from config import Config # Config is a global variable

__all__ = ["plot_paraxial_surfaces", "plot_ray", "plot_spherical_surfaces"]

plot_params = {"figsize" : (12,4), "surfcolor" : "red"}



def _init_figure():
    fig = plt.figure("fig1", figsize=plot_params["figsize"], layout="tight")
    ax = fig.subplots(1,1)
    # plot optical axis
    ax.axhline(y=0, color="k", linestyle="--")
    ax.set_xlabel(f"{Config.lens_unit}")
    ax.set_ylabel(f"{Config.lens_unit}")

    return fig


def plot_paraxial_surfaces_v2(dists: Iterable, fig: Figure = None) -> Figure:

    if fig is None:
        fig = _init_figure()

    ax = fig.axes[0]        
    
    # draw paraxial lens surfaces, i.e. the plane through the vertex orthogonal to the optical axis
    l=-dists[0] # object distance is negative, lens system starts at z=0
    ax.axvline(x=l, color=plot_params["surfcolor"], linewidth=0.5)
    ax.text(l-0.4, 0.0, "OBJECT", rotation=90, va="center")
    for i in range(0, len(dists)):        
        l += dists[i]
        ax.axvline(x=l, color=plot_params["surfcolor"])
    ax.text(l-0.2, 0.0, "IMAGE", rotation=90, va="center")

    return fig 


def plot_paraxial_surfaces(vertex: Iterable, fig: Figure = None) -> Figure:

    if fig is None:
        fig = _init_figure()

    ax = fig.axes[0]        
    
    # draw paraxial lens surfaces, i.e. the plane through the vertex orthogonal to the optical axis
    # object distance is negative, lens system starts at z=0
    ax.axvline(x=vertex[0], color=plot_params["surfcolor"], linewidth=0.5)
    ax.text(vertex[0]-0.4, 0.0, "OBJECT", rotation=90, va="center")
    for i in range(1, len(vertex)):        
        ax.axvline(x=vertex[i], color=plot_params["surfcolor"])
    ax.text(vertex[-1]-0.2, 0.0, "IMAGE", rotation=90, va="center")

    return fig 


def plot_ray(dists: Iterable, ys: Iterable, fig: Figure = None, z_sag: Iterable = None, 
             color="red", linewidth=1, dashtype="-") -> Figure:        
    if fig is None:
        fig = plot_paraxial_surfaces(dists, fig)

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
    print("zz.shape=", zz.shape)
    print("z_sag.shape=", z_sag.shape)

    ax.plot(zz+z_sag, yy, dashtype+"o", color=color, linewidth=linewidth)

    if dists[0] > 600:
        ax.set_xlim(-10, np.sum(dists[1:]))
    
    return fig


def plot_ray_v2(vertex: Iterable, ys: Iterable, fig: Figure = None, z_sag: Iterable = None, 
             color="red", linewidth=1, dashtype="-") -> Figure:
    
    if fig is None:
        fig = plot_paraxial_surfaces(vertex, fig)

    print("vertex.shape=", vertex.shape)
    print("z_sag.shape=", z_sag.shape)

    ax = fig.axes[0]
    if z_sag is None:
        z_sag = np.zeros_like(vertex)
    ax.plot(vertex+z_sag, ys, dashtype+"o", color=color, linewidth=linewidth)

    if np.abs(vertex[0]) > 600:
        ax.set_xlim(-10, vertex[-1])
    
    return fig



def plot_spherical_surfaces(dists: Iterable, Rs: Iterable, clear_aperture: Iterable, ns: Iterable, 
                  fig: Figure = None) -> Figure:
    """
    Plot spherical lens surfaces up to their clear apertures.

    Mark refractive materials with colors, differentiating between lens elements with 
    positive and negative refractive index. The hue of the color indicates the absolute 
    magnitude of the refractive index. 
    """
    if fig is None:
        fig = _init_figure()

    # idenfity singlets, doublets and triplets so that the clear apertures of their
    # surfaces can be combined into a lens
    n_air = 1.0 # 1.000302
    nmat = np.invert((np.isclose(ns, n_air, atol=1e-6)))  # Which segements are materials other than air ?
    print("nmat=", nmat)
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

    fig.axes[0].axis([-dists[0], np.sum(dists[1:]), -max(ymax)-2.0, max(ymax)+2.0])
    vertex = 0
    lens_edges = []
    for i in range(1, len(dists)):
        if np.isinf(Rs[i]):
            fig.axes[0].axvline(x=vertex, ymin=-ymax[i], ymax=ymax[i], color="k")
        else:
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
    
    fig.axes[0].set_aspect("auto")
    return fig


def plot_spherical_surfaces_v2(vertex: Iterable, Rs: Iterable, clear_aperture: Iterable, ns: Iterable, 
                  fig: Figure = None) -> Figure:
    """
    Plot spherical lens surfaces up to their clear apertures.

    Mark refractive materials with colors, differentiating between lens elements with 
    positive and negative refractive index. The hue of the color indicates the absolute 
    magnitude of the refractive index. 
    
    Returns
    -------
    fig: Figure object
    surface_segments[1:num_surfs-1]: Line2D objects representing the drawn lens surfaces
    edge_segments: Line2D objects representing the horizontal lens edges
    """

    def _plot_spherical_segment_i(i: int, front: bool, fig: Figure) -> Tuple[Figure, Tuple]:
        zmax = np.abs(Rs[i])*(1.0-np.sqrt(1.0-(clear_aperture[i]/Rs[i])**2))
        zz = np.linspace(0, zmax, 2000)
        # plot the curved lens surface above and below the optical axis
        yy = +np.sqrt(Rs[i]**2-(np.abs(Rs[i])-zz)**2)
        curved_up = fig.axes[0].plot(vertex[i]+np.sign(Rs[i])*zz, yy, color="k", linewidth=2)    
        yy = -np.sqrt(Rs[i]**2-(np.abs(Rs[i])-zz)**2)
        curved_dn = fig.axes[0].plot(vertex[i]+np.sign(Rs[i])*zz, yy, color="k", linewidth=2)
        line2d_segments = (curved_up, curved_dn)    

        zpos = vertex[i]+np.sign(Rs[i])*zmax
        if front: # front surface of a lens elements
            if clear_aperture[i] < clear_aperture[i+1]:
                # plot vertical lines
                vline_up = fig.axes[0].plot([zpos, zpos], [clear_aperture[i], clear_aperture[i+1]], color="k", linewidth=2)
                vline_dn = fig.axes[0].plot([zpos, zpos], [-clear_aperture[i+1], -clear_aperture[i]], color="k", linewidth=2)
                line2d_segments += (vline_up, vline_dn,)
        else: # back surface of a lens element
            if clear_aperture[i] < clear_aperture[i-1]:
                print("back surface i=", i, "clear_aperture[i]=", clear_aperture[i], "clear_aperture[i-1]=", clear_aperture[i-1])
                vline_up = fig.axes[0].plot([zpos, zpos], [clear_aperture[i], clear_aperture[i-1]], color="k", linewidth=2)
                vline_dn = fig.axes[0].plot([zpos, zpos], [-clear_aperture[i-1], -clear_aperture[i]], color="k", linewidth=2)
                line2d_segments += (vline_up, vline_dn,)
        return fig, line2d_segments
    
    def _draw_horizontal_edge(i: int, fig: Figure) -> Figure:
        zmax1 = np.abs(Rs[i])*(1.0-np.sqrt(1.0-(clear_aperture[i]/Rs[i])**2))
        zmax2 = np.abs(Rs[i+1])*(1.0-np.sqrt(1.0-(clear_aperture[i+1]/Rs[i+1])**2))

        max_CA = max(clear_aperture[i], clear_aperture[i+1])
        zpos1 = vertex[i]+np.sign(Rs[i])*zmax1
        zpos2 = vertex[i+1]+np.sign(Rs[i+1])*zmax2
        # top 
        edge_top = fig.axes[0].plot([zpos1, zpos2], [max_CA, max_CA], color="k", linewidth=2)
        # bottom
        edge_bottom = fig.axes[0].plot([zpos1, zpos2], [-max_CA, -max_CA], color="k", linewidth=2)

        return fig, (edge_top, edge_bottom)


    if fig is None:
        fig = plot_paraxial_surfaces(vertex) #_init_figure()

    nmat = np.invert((np.isclose(ns, Config.n_air, atol=1e-6)))  # Which segements are materials other than air ?
    
    for i in range(1,len(vertex)-1):
        if nmat[i]:
            # Draw front surface up to its clear aperture and a vertical line from a clear aperture that is smaller 
            # than the maximal clear aperture of the lens element to the lens edge.
            fig, line2d_segments_front = _plot_spherical_segment_i(i, True, fig)

            # Draw back surface. In a cemented lens adjacent surface are plotted twice. Draw a vertical line from a 
            # clear aperture that is smaller than the maximal clear aperture of the lens element to the lens edge.
            fig, line2d_segments_back = _plot_spherical_segment_i(i+1, False, fig)

            # Draw horizontal lens edges.
            fig, edge_segments = _draw_horizontal_edge(i, fig)

    # Sort out duplicate surface segments
    # for seg in line2d_segments

    fig.axes[0].set_aspect("equal")
    return fig