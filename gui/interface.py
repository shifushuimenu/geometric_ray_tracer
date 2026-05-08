"""
Interface between the lens sequence and ray tracing data on the one hand 
and matplotlib figures that will be registered with a canvas of the GUI
on the other hand.

  (ray tracer, lens sequence) <-->     interface        <--> GUI
                                   (matplotlib figures) 
"""
import numpy as np
from matplotlib.figure import Figure
import matplotlib as mpl

from raytracer.paraxial import ParaxialQuantities
from raytracer.lens import LensSequence
from raytracer.config import Config
from raytracer.aberrations import Aberrations3rd
from raytracer.trace_ray import MeridionalRayData, NonmeridionalRayData
from raytracer.plot import (plot_spherical_surfaces, 
                  plot_ray_bundles,
                  display_pupils_and_stops)

# Put all plot parameters here: colors, linewidths
params = {
    "spherical_surface_color" : "black",
    "highlight_surface_color" : "magenta",
}


class DisplayInterfaceRaytrace(object):
    def __init__(self, lens_sequence: LensSequence, config: Config, ray_data: MeridionalRayData,
                 paraxial_quantities: ParaxialQuantities = None) -> None:        
        self.LS = lens_sequence
        self.config = config        
        self.RD = ray_data
        self.PQ = paraxial_quantities 

        self.objects_on_screen = dict()
        self.highlighted_surface_i = None

    def init_figure(self, fig: Figure=None) -> Figure:
        if fig is None:
            fig = Figure()
            fig.set_size_inches(12,6)
            fig.set_layout_engine("tight")
            ax = fig.subplots(1,1)
        fig = self.plot_labels_and_optical_axis(fig)
        return fig
    
    def plot_labels_and_optical_axis(self, fig: Figure) -> Figure:
        ax = fig.get_axes()[0]
        # plot optical axis
        ax.axhline(y=0, color="k", linestyle="--")
        ax.set_xlabel("mm")
        ax.set_ylabel("mm")
        return fig         

    def plot_spherical_surfaces(self, fig: Figure) -> Figure:
        fig, surface_segments, edge_segments = plot_spherical_surfaces(self.LS.vertex, self.LS.R, self.LS.n, 
                                                                       self.RD.clear_apertures, self.LS.AS_surf, self.config, fig)
        self.objects_on_screen["surfaces"] = surface_segments
        self.objects_on_screen["edges"] = edge_segments

        return fig

    def hide_spherical_surfaces(self, fig: Figure) -> Figure:
        return fig 
    
    def highlight_surface(self, i):
        # IMPROVE: - change color to entire linestyle
        #          - In a new config, surfaces are not highlighted when pressing Enter on the corresponding line.
        """Highlight the surface with index i with a given color""" 
        print(f"DisplayInterfaceRaytracer called highlight_surface(), id={id(self)}")
        if "surfaces" in self.objects_on_screen:
            if self.highlighted_surface_i is not None:
                #  reset 
                for l in self.objects_on_screen["surfaces"][self.highlighted_surface_i]:
                    l[0].set_color(params["spherical_surface_color"])

            for l in self.objects_on_screen["surfaces"][i]: # loop over surface segments
                l[0].set_color(params["highlight_surface_color"])
                print("l[0].get_color()=", l[0].get_color())
            self.highlighted_surface_i = i

    def color_lens_elements(self, fig: Figure) -> Figure:
        return fig 

    def uncolor_lens_elements(self, fig: Figure) -> Figure:
        return fig

    def plot_paraxial_surfaces(self, fig: Figure) -> Figure:
        return fig 

    def hide_paraxial_surfaces(self, fig: Figure) -> Figure:
        return fig

    def plot_pupils_and_stops(self, fig: Figure) -> Figure:
        return display_pupils_and_stops(self.LS, self.PQ, fig) 

    def hide_pupils_and_stops(self, fig: Figure) -> Figure:
        return fig

    def plot_ray_bundles(self, ray_data: MeridionalRayData, fig: Figure) -> Figure:
        fig = plot_ray_bundles(ray_data, fig)
        return fig
    

class DisplayInterfaceRayspot(object):
    def __init__(self, lens_sequence: LensSequence, config: Config, ray_data: NonmeridionalRayData):
        self.lens_sequence = lens_sequence
        self.config = config
        self.RD = ray_data

    def init_figure(self) -> Figure:
        fig = Figure()
        fig.set_size_inches(9,3)
        fig.set_layout_engine("tight")
        fig.subplots(1,self.config.num_fields, sharex=True, sharey=True)
        return fig
    
    def plot_ray_spots(self, ray_data: NonmeridionalRayData, surf: int=-1, fig: Figure=None) -> Figure:
        if fig is None:
            fig = self.init_figure()
        axs = fig.axes
        P_intersect = ray_data.P_intersect
        colors = ["blue", "green", "red"] if self.config.num_fields == 3 else mpl.color_sequences["tab10"][0:self.config.num_fields]
        # Plot intersection points in the image plane.
        # Plot off-axis ray bundles relative to the chief ray.
        for f in range(self.config.num_fields):
            axs[f].set_aspect("equal", adjustable="box")
            axs[f].set_frame_on(True)
            axs[f].grid(True)
            axs[f].set_xlabel("mm")
            axs[f].set_ylabel("mm")
            axs[f].set_title(f"OBJ: {P_intersect[1,0,ray_data.CHIEF_RAY_INDEX,f]:.3f} mm \n"
                             f"IMA: {P_intersect[1,surf,ray_data.CHIEF_RAY_INDEX,f]:.3f} mm")
            y_CR = P_intersect[1,surf,ray_data.CHIEF_RAY_INDEX,f]
            axs[f].plot(P_intersect[0,surf,:,f], P_intersect[1,surf,:,f] - y_CR, 'o', color=colors[f])
        fig.tight_layout()
        return fig

class DisplayInterfaceRayfan(object):
    def __init__(self, lens_sequence: LensSequence, config: Config):
        self.lens_sequence = lens_sequence
        self.config = config

    def init_figure(self) -> Figure:
        fig = Figure()
        fig.set_size_inches(9,4)
        fig.subplots(nrows=1, ncols=self.config.num_fields, squeeze=True)
        fig.tight_layout(pad=4.0)
        return fig
    
    def plot_rayfan(self, tangential_ray_data: MeridionalRayData, P_intersect: np.ndarray, surf: int, fig: Figure=None) -> Figure:
        if fig is None:
            fig = self.init_figure()
        axs = fig.axes

        for f in range(self.config.num_fields)[::-1]:            
            # tangential ray fan
            CHIEF_INDEX = tangential_ray_data.CHIEF_RAY_INDEX
            y = tangential_ray_data.y[surf,:,f] - tangential_ray_data.y[surf,CHIEF_INDEX,f]
            # We take the coordinate in the aperture stop as the pupil coordinate since the aperture stop and entrance pupil are conjugate planes.
            Py = tangential_ray_data.y[self.lens_sequence.AS_surf,:,f] # chief ray height at aperture stop is zero by definition
            # normalize
            Py /= np.max(np.abs(Py))

            # sagittal ray fan
            x = P_intersect[0,surf,:,f] - 0.0 # should be symmetric about x=0
            Px = P_intersect[0,self.lens_sequence.AS_surf,:,f]
            Px /= np.max(np.abs(Px))

            axs[f].set_ylabel(r"$y / x$"+f"\t [{self.config.lens_unit}]", loc="center")
            axs[f].set_xlabel(r"$P_y / P_x$", loc="center")
            axs[f].set_title(f"OBJ: {self.config.obj_heights[f]:.3f} {self.config.lens_unit}")
            if f == 0:
                line1, = axs[f].plot(Py, y, '-', label="tangential (y, P_y)")
                line2, = axs[f].plot(Px, x, '--', label="sagittal (x, P_x)")
            else:
                axs[f].plot(Py, y, '-', label="tangential")
                axs[f].plot(Px, x, '--', label="sagittal")
            axs[f].grid(visible=True)
            ylim = np.round(np.max(np.abs(axs[f].get_ylim())), decimals=3) # min scale is E-3 lens units
            axs[f].set_ylim(bottom=-ylim, top=ylim)
            axs[f].set_box_aspect(1.0)
        
            axs[0].legend(bbox_to_anchor=(0.4,0.08), loc="center left",  bbox_transform=fig.transFigure)

        return fig


class DisplayInterfaceSeidelDiagram(object):
    def __init__(self, lens_sequence: LensSequence, config: Config, aberrations: Aberrations3rd) -> Figure:
        self.lens_sequence = lens_sequence 
        self.config = config
        self.aberrations = aberrations

    def init_figure(self) -> Figure:
        fig = Figure()
        fig.set_size_inches(12,8)
        fig.set_layout_engine("tight")
        ax = fig.subplots(1,1)
        return fig

    def plot_Seidel_diagram(self, AS_surf, config: Config, fig: Figure=None) -> Figure:
        num_aberrations = 5
        num_surfs = self.lens_sequence.num_surfs
        barWidth = 1.0/(num_aberrations+1)

        # IMPROVE: Skip object surface which has not aberrations.
        spherical = self.aberrations.S1
        coma = self.aberrations.S2
        astigmatism = self.aberrations.S3
        field_curvature = self.aberrations.S4
        distortion = self.aberrations.S5

        br1 = range(0, num_surfs) 
        br2 = [x + barWidth for x in br1] 
        br3 = [x + barWidth for x in br2] 
        br4 = [x + barWidth for x in br3] 
        br5 = [x + barWidth for x in br4] 
        br_sep = [x + barWidth for x in br5] 

        if fig is None:
            fig = self.init_figure()

        fig.axes[0].bar(br1, spherical, color ='red', width = barWidth, 
                edgecolor ='grey', label ='spherial') 
        fig.axes[0].bar(br2, coma, color ='green', width = barWidth, 
                edgecolor ='grey', label ='coma') 
        fig.axes[0].bar(br3, astigmatism, color ='magenta', width = barWidth, 
                edgecolor ='grey', label ='astigmatism') 
        fig.axes[0].bar(br4, field_curvature, color ='cyan', width = barWidth, 
                edgecolor ='grey', label ='field curvature') 
        fig.axes[0].bar(br5, distortion, color ='yellow', width = barWidth, 
                edgecolor ='grey', label ='distortion') 
        ymin, ymax = fig.axes[0].get_ylim()
        fig.axes[0].vlines(br_sep, ymin, ymax, linewidth=2, color="black")
        fig.axes[0].tick_params(top=True, labeltop=True, bottom=True, labelbottom=True)

        fig.axes[0].set_ylabel('Seidel aberrations [%s]'%(config.lens_unit), fontweight ='bold', fontsize = 15) 
        fig.axes[0].set_xticks([r + 2*barWidth for r in range(1,num_surfs)], 
            ["STO" if r == AS_surf else str(r) for r in range(1,num_surfs-1)]+["SUM"])

        fig.axes[0].set_title("Surfaces", fontsize=15)
        fig.axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5, fontsize=15)

        return fig
