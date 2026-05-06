import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from typing import Iterable, Tuple, List

from trace_ray import MeridionalRayData
from config import Config
from paraxial import ParaxialQuantities
from lens import LensSequence

__all__ = ["plot_paraxial_surfaces", "plot_ray", "plot_spherical_surfaces", "plot_ray_bundles", "display_pupils_and_stops"]

plot_params = {"figsize" : (12,4), "surfcolor" : "red", "lens_unit" : "mm"}

def init_figure():
    fig = plt.figure("fig1", figsize=plot_params["figsize"], layout="tight")
    ax = fig.subplots(1,1)
    # plot optical axis
    ax.axhline(y=0, color="k", linestyle="--")
    ax.set_xlabel(f"{plot_params["lens_unit"]}")
    ax.set_ylabel(f"{plot_params["lens_unit"]}")

    return fig


def plot_paraxial_surfaces(vertex: Iterable, fig: Figure = None) -> Figure:
    """
    Draw paraxial lens surfaces, i.e. the planes through the vertices orthogonal to the optical axis.
    The object distance is negative, the lens system starts at z=0.
    """
    if fig is None:
        fig = init_figure()

    ax = fig.axes[0]        
    
    ax.axvline(x=vertex[0], color=plot_params["surfcolor"], linewidth=0.5)
    ax.text(vertex[0]-0.4, 0.0, "OBJECT", rotation=90, va="center")
    for i in range(1, len(vertex)):        
        ax.axvline(x=vertex[i], color=plot_params["surfcolor"])
    ax.text(vertex[-1]-0.2, 0.0, "IMAGE", rotation=90, va="center")

    return fig 


def plot_ray(vertex: Iterable, ys: Iterable, fig: Figure = None, z_sag: Iterable = None, 
             color="red", linewidth=1, dashtype="-") -> Figure:
    
    if fig is None:
        fig = plot_paraxial_surfaces(vertex, fig)
    ax = fig.axes[0]

    if z_sag is None:
        z_sag = np.zeros_like(vertex)

    ax.plot(vertex+z_sag, ys, dashtype, color=color, linewidth=linewidth)

    if np.abs(vertex[0]) > 600:
        ax.set_xlim(-10, vertex[-1])
    
    return fig


def plot_spherical_surfaces(vertex: Iterable, Rs: Iterable, ns: Iterable, clear_apertures: Iterable, AS_surf: float,
                            config: Config, fig: Figure = None) -> Tuple[Figure, List, List]:
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
    def _plot_spherical_segments_i(i: int, front: bool, fig: Figure) -> Tuple[Figure, Tuple]:
        zmax = np.abs(Rs[i])*(1.0-np.sqrt(1.0-(clear_apertures[i]/Rs[i])**2))
        zz = np.linspace(0, zmax, 2000)
        # plot the curved lens surface above and below the optical axis
        yy = +np.sqrt(Rs[i]**2-(np.abs(Rs[i])-zz)**2)
        curved_up = fig.axes[0].plot(vertex[i]+np.sign(Rs[i])*zz, yy, color="k", linewidth=2)    
        yy = -np.sqrt(Rs[i]**2-(np.abs(Rs[i])-zz)**2)
        curved_dn = fig.axes[0].plot(vertex[i]+np.sign(Rs[i])*zz, yy, color="k", linewidth=2)
        line2d_segments = [curved_up, curved_dn]

        zpos = vertex[i]+np.sign(Rs[i])*zmax
        if front: # front surface of a lens elements
            if clear_apertures[i] < clear_apertures[i+1]:
                # plot vertical lines
                vline_up = fig.axes[0].plot([zpos, zpos], [clear_apertures[i], clear_apertures[i+1]], color="k", linewidth=2)
                vline_dn = fig.axes[0].plot([zpos, zpos], [-clear_apertures[i+1], -clear_apertures[i]], color="k", linewidth=2)
                line2d_segments += [vline_up, vline_dn]
        else: # back surface of a lens element
            if clear_apertures[i] < clear_apertures[i-1]:
                vline_up = fig.axes[0].plot([zpos, zpos], [clear_apertures[i], clear_apertures[i-1]], color="k", linewidth=2)
                vline_dn = fig.axes[0].plot([zpos, zpos], [-clear_apertures[i-1], -clear_apertures[i]], color="k", linewidth=2)
                line2d_segments += [vline_up, vline_dn]
        return fig, line2d_segments
    
    def _draw_horizontal_edge(i: int, fig: Figure) -> Figure:
        zmax1 = np.abs(Rs[i])*(1.0-np.sqrt(1.0-(clear_apertures[i]/Rs[i])**2))
        zmax2 = np.abs(Rs[i+1])*(1.0-np.sqrt(1.0-(clear_apertures[i+1]/Rs[i+1])**2))

        max_CA = max(clear_apertures[i], clear_apertures[i+1])
        zpos1 = vertex[i]+np.sign(Rs[i])*zmax1
        zpos2 = vertex[i+1]+np.sign(Rs[i+1])*zmax2
        # top 
        edge_top = fig.axes[0].plot([zpos1, zpos2], [max_CA, max_CA], color="k", linewidth=2)
        # bottom
        edge_bottom = fig.axes[0].plot([zpos1, zpos2], [-max_CA, -max_CA], color="k", linewidth=2)

        return fig, [edge_top, edge_bottom]

    num_surfs = len(vertex)

    if fig is None:
        fig = plot_paraxial_surfaces(vertex) #_init_figure()

    nmat = np.invert((np.isclose(ns[:-1], config.n_air, atol=1e-6)))  # Which segements are materials other than air ?
    
    surface_segments = []
    edge_segments = []
    surface_done = np.array([False for _ in range(num_surfs)])

    # plot object plane
    line2d_object = fig.axes[0].plot([vertex[0], vertex[0]], [-clear_apertures[0], clear_apertures[0]],
                                     color="k", linewidth=2)
    # plot image plane 
    line2d_image = fig.axes[0].plot([vertex[num_surfs-1],vertex[num_surfs-1]], 
                                    [-clear_apertures[num_surfs-1], clear_apertures[num_surfs-1]], color="k", linewidth=2)

    for i in range(1,num_surfs-1):
        if nmat[i]:            
            # Draw front surface up to its clear aperture and a vertical line from a clear aperture that is smaller 
            # than the maximal clear aperture of the lens element to the lens edge.
            fig, line2d_segments_front = _plot_spherical_segments_i(i, True, fig)

            # Draw back surface. In a cemented lens adjacent surface are plotted twice. Draw a vertical line from a 
            # clear aperture that is smaller than the maximal clear aperture of the lens element to the lens edge.
            fig, line2d_segments_back = _plot_spherical_segments_i(i+1, False, fig)

            # Draw horizontal lens edges.
            fig, edges = _draw_horizontal_edge(i, fig)
            edge_segments.append(edges)

            # Avoid duplicate surface segments for cemented lens elements
            if nmat[i] and nmat[i+1]:
                surface_segments.extend([line2d_segments_front])
                surface_done[i] = True
            else:
                surface_segments.extend([line2d_segments_front, line2d_segments_back])
                surface_done[i:i+2] = True


    # Insert empty entries so that surface indexing is consistent.
    for i in range(0, num_surfs):
        if not surface_done[i]:
            surface_segments.insert(i, [])

    # # object and image plane
    surface_segments[0] = [line2d_object]
    surface_segments[num_surfs-1] = [line2d_image]

    # aperture stop
    aw = 0.25 # lens units # 0.1*np.max(clear_apertures) # size of symbols drawn at the edges of the aperture stop and pupils
    fig, line2d_segments_AS = plot_aperture_edges(vertex[AS_surf], clear_apertures[AS_surf],  aw, "", fig)
    surface_segments[AS_surf].extend(line2d_segments_AS)

    fig.axes[0].set_aspect("auto")
    return fig, surface_segments, edge_segments


def plot_ray_bundles(ray_data: MeridionalRayData, fig: Figure) -> Figure:
    num_fields = ray_data.num_fields
    num_rays = ray_data.num_rays
    colors = ["blue", "green", "red"] if num_fields == 3 else mpl.color_sequences["tab10"][0:num_fields]
    for f in range(num_fields):
        # plot the chief ray for each ray bundle 
        fig = plot_ray(ray_data.vertex, ray_data.y[:,ray_data.CHIEF_RAY_INDEX,f], fig, ray_data.z_sag[:,ray_data.CHIEF_RAY_INDEX,f], 
                       color="orange", linewidth=2)
        for r in range(num_rays):
            fig = plot_ray(ray_data.vertex, ray_data.y[:,r,f], fig, ray_data.z_sag[:,r,f], color=colors[f])
    return fig      


def plot_aperture_edges(aperture_position: float, aperture_radius: float, marker_w: float, label: str, fig: Figure) -> Tuple[Figure, List]:
    aspectr = fig.axes[0].get_aspect()
    ar = 1.0 if aspectr in ["auto"] else aspectr    
    fig.axes[0].text(aperture_position, aperture_radius+1.2*marker_w*ar, label)
    line_segments = []
    for pm in [+1,-1]:
        line_segments_h = fig.axes[0].plot([aperture_position-marker_w, aperture_position+marker_w], [pm*aperture_radius, pm*aperture_radius], color="black", linewidth=2)
        line_segments_v = fig.axes[0].plot([aperture_position, aperture_position], [pm*aperture_radius, pm*(aperture_radius+marker_w/ar)], color="black", linewidth=2)
        line_segments.extend([line_segments_h, line_segments_v])
    return fig, line_segments


def display_pupils_and_stops(lens_sequence: LensSequence, paraxial_quantities: ParaxialQuantities, fig: Figure) -> Figure:
    y_chief = paraxial_quantities.y_chief
    y_marg = paraxial_quantities.y_marg
    u_marg = paraxial_quantities.u_marg
    EPP = paraxial_quantities.EPP
    XPP = paraxial_quantities.XPP
    EPD = paraxial_quantities.EPD
    XPD = paraxial_quantities.XPD
    aw = 0.1 * np.max(lens_sequence.clear_diameter) # size of symbols drawn at the edges of the aperture stop and pupils
    for pm in [+1, -1]:
        # Plot chief ray and marginal ray.
        fig = plot_ray(lens_sequence.vertex, pm*y_chief, fig, z_sag=np.zeros_like(y_chief), color="blue", linewidth=2)
        fig = plot_ray(lens_sequence.vertex, pm*y_marg, fig, z_sag=np.zeros_like(y_chief), color="red", linewidth=2)
        # Extrapolate the chief ray on the object side and image side till it hits the optical axis.
        fig = plot_ray([lens_sequence.vertex[1], EPP], [pm*y_chief[1], 0], fig, z_sag=np.zeros(2), color="blue", linewidth=2, dashtype="--")
        # Note that the exit pupil position is measured relative to the image plane.
        fig = plot_ray([lens_sequence.vertex[-2], lens_sequence.vertex[-1]+XPP], [pm*y_chief[-2], 0], fig, z_sag=np.zeros(2), color="blue", linewidth=2, dashtype="--")

        # Extrapolate the marginal ray on the object side and image side till it hits the edge 
        # of the entrance and exit pupil.
        sgn_EP_ = np.sign(pm*(y_marg[1] + u_marg[1])*(EPP - lens_sequence.vertex[1]))
        fig = plot_ray([lens_sequence.vertex[1], EPP], [pm*y_marg[1], sgn_EP_*EPD/2.0], fig, 
                       z_sag=np.zeros(2), color="red", linewidth=2, dashtype="--")
        sgn_XP_ = np.sign(pm*(y_marg[-2] + u_marg[-2]*(XPP + (lens_sequence.vertex[-1] - lens_sequence.vertex[-2]))))
        fig = plot_ray([lens_sequence.vertex[-2], lens_sequence.vertex[-1]+XPP], [pm*y_marg[-2], sgn_XP_*XPD/2.0], fig, 
                       z_sag=np.zeros(2), color="red", linewidth=2, dashtype="--")

    # Draw edges of the aperture stop, the entrance pupil and the exit pupil.
    fig, _ = plot_aperture_edges(EPP, EPD/2.0, aw, "EP", fig)
    fig, _ = plot_aperture_edges(lens_sequence.vertex[-1]+XPP, XPD/2.0, aw, "XP", fig)
    # fig = _plot_aperture_edges(lens_sequence.vertex[lens_sequence.AS_surf], paraxial_quantities.stop_radius, aw, "AS", fig)

    return fig


# def plot_paraxial_surfaces_v2(dists: Iterable, fig: Figure = None) -> Figure:

#     if fig is None:
#         fig = _init_figure()

#     ax = fig.axes[0]        
    
#     # draw paraxial lens surfaces, i.e. the plane through the vertex orthogonal to the optical axis
#     l=-dists[0] # object distance is negative, lens system starts at z=0
#     ax.axvline(x=l, color=plot_params["surfcolor"], linewidth=0.5)
#     ax.text(l-0.4, 0.0, "OBJECT", rotation=90, va="center")
#     for i in range(0, len(dists)):        
#         l += dists[i]
#         ax.axvline(x=l, color=plot_params["surfcolor"])
#     ax.text(l-0.2, 0.0, "IMAGE", rotation=90, va="center")

#     return fig 


# def plot_ray_v2(dists: Iterable, ys: Iterable, fig: Figure = None, z_sag: Iterable = None, 
#              color="red", linewidth=1, dashtype="-") -> Figure:        
#     if fig is None:
#         fig = plot_paraxial_surfaces(dists, fig)

#     ax = fig.axes[0]
#     l=-dists[0]
#     zz = [l]           
#     yy = [ys[0]]     
#     for i in range(0, len(dists)):        
#         l += dists[i]
#         zz.append(l)
#         yy.append(ys[i+1])

#     zz = np.array(zz)
#     yy = np.array(yy)
#     if z_sag is None:
#         z_sag = np.zeros_like(zz)
#     print("zz.shape=", zz.shape)
#     print("z_sag.shape=", z_sag.shape)

#     ax.plot(zz+z_sag, yy, dashtype+"o", color=color, linewidth=linewidth)

#     if dists[0] > 600:
#         ax.set_xlim(-10, np.sum(dists[1:]))
    
#     return fig


# def plot_spherical_surfaces_v2(dists: Iterable, Rs: Iterable, clear_aperture: Iterable, ns: Iterable, 
#                   fig: Figure = None) -> Figure:
#     """
#     Plot spherical lens surfaces up to their clear apertures.

#     Mark refractive materials with colors, differentiating between lens elements with 
#     positive and negative refractive index. The hue of the color indicates the absolute 
#     magnitude of the refractive index. 
#     """
#     if fig is None:
#         fig = _init_figure()

#     # idenfity singlets, doublets and triplets so that the clear apertures of their
#     # surfaces can be combined into a lens
#     n_air = 1.0 # 1.000302
#     nmat = np.invert((np.isclose(ns, n_air, atol=1e-6)))  # Which segements are materials other than air ?
#     ymax = clear_aperture.copy()   
#     y_ = 0.0; multiplet_elements = 0
#     for i in range(1,len(Rs)):        
#         if nmat[i]:
#             if i <= len(Rs)-2:
#                 y_ = max(y_, clear_aperture[i], clear_aperture[i+1])
#             else:
#                 y_ = max(y_, clear_aperture[i])
#             multiplet_elements += 1
#         else:
#             ymax[i-multiplet_elements:i+1] = y_
#             # reset
#             y_ = 0.0; multiplet_elements = 0

#     fig.axes[0].axis([-dists[0], np.sum(dists[1:]), -max(ymax)-2.0, max(ymax)+2.0])
#     vertex = 0
#     lens_edges = []
#     for i in range(1, len(dists)):
#         if np.isinf(Rs[i]):
#             fig.axes[0].axvline(x=vertex, ymin=-ymax[i], ymax=ymax[i], color="k")
#         else:
#             zmax = np.abs(Rs[i])*(1.0-np.sqrt(1.0-(ymax[i]/Rs[i])**2))
#             zz = np.linspace(0, zmax, 2000)
#             for pm in [+1,-1]:
#                 # plot the curved lens surface above and below the optical axis
#                 yy = pm*np.sqrt(Rs[i]**2-(np.abs(Rs[i])-zz)**2)
#                 fig.axes[0].plot(vertex+np.sign(Rs[i])*zz, yy, color="k", linewidth=2)
#                 if pm == +1:
#                     lens_edges.append([vertex+np.sign(Rs[i])*zz[-1], yy[-1]])
#             if not nmat[i]:
#                 # print("lens_edges=", lens_edges)
#                 for pm in [+1,-1]:
#                     fig.axes[0].plot([e[0] for e in lens_edges],
#                                      [pm*e[1] for e in lens_edges], 
#                                         color="k", linewidth=2)
#                 lens_edges = []
#         vertex += dists[i]
    
#     fig.axes[0].set_aspect("auto")
#     return fig