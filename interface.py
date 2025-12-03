from paraxial import ParaxialQuantities
from lens import LensSequence
from config import Config
from matplotlib.figure import Figure
from trace_ray import MeridionalRayData, NonmeridionalRayData
from plot import plot_spherical_surfaces, plot_ray_bundles

import matplotlib as mpl

# Put all plot parameters here: colors, linewidths
params = {
    "spherical_surface_color" : "black",
    "highlight_surface_color" : "magenta",
}


class DisplayInterface:
    def __init__(self, lens_sequence: LensSequence, config: Config, ray_data: MeridionalRayData,
                 paraxial_quantities: ParaxialQuantities = None):        
        self.LS = lens_sequence
        self.config = config        
        self.RD = ray_data
        self.PQ = paraxial_quantities 

        self.objects_on_screen = dict()
        self.highlighted_surface_i = None

    def init_figure(self) -> Figure:
        fig = Figure()
        fig.set_size_inches(12,6)
        fig.set_layout_engine("tight")
        ax = fig.subplots(1,1)
        # plot optical axis
        ax.axhline(y=0, color="k", linestyle="--")
        ax.set_xlabel("mm")
        ax.set_ylabel("mm")
        return fig 

    def plot_spherical_surfaces(self, fig: Figure) -> Figure:
        fig, surface_segments, edge_segments = plot_spherical_surfaces(self.LS.vertex, self.LS.R, self.LS.n, 
                                                                       self.RD.clear_apertures, self.config, fig)
        self.objects_on_screen["surfaces"] = surface_segments
        self.objects_on_screen["edges"] = edge_segments

        return fig

    def hide_spherical_surfaces(self, fig: Figure) -> Figure:
        return fig 
    
    def highlight_surface(self, i):
        # IMPROVE: change color to entire linestyle
        """Highlight the surface with index i with a given color"""            
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
        return fig 

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

