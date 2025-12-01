from paraxial import ParaxialQuantities
from lens import LensSequence
from config import Config
from matplotlib.figure import Figure
from trace_ray import MeridionalRayData
from plot import plot_spherical_surfaces, plot_ray_bundles, init_figure

# Put all plot parameters here: colors, linewidths
params = {
    "spherical_surface_color" : "black",
    "highlight_surface_color" : "magenta",
}


class DisplayInterface:
    def __init__(self, lens_sequence: LensSequence, ray_data: MeridionalRayData, config: Config,
                 paraxial_quantities: ParaxialQuantities = None):
        self.PQ = paraxial_quantities 
        self.LS = lens_sequence
        self.RD = ray_data
        self.config = config

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
    
