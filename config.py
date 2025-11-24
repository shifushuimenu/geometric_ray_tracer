"""Global configurations"""
import numpy as np

class Config():
    def __init__(self, max_obj_height: float, entrance_pupil_diameter: float):
        # TODO: These should be choices from a set of options (use argparse)
        self.EPD = entrance_pupil_diameter # entrance pupil diameter
        self.max_obj_height = max_obj_height
        self.n_air = 1.0 # 1.000302 # which refractive index to use
        self.lens_unit = "mm"        
        self.wavelength = 0.588 # in micrometers, sodium Fraunhofer d-line
        self.num_rays = 7
        self.obj_heights = np.array([self.max_obj_height, self.max_obj_height/np.sqrt(2.0), 0.0])
        self.ON_AXIS_FIELD_INDEX = -1  # last index
        self.MAX_OBJ_HEIGHT_INDEX = 0
        self.num_fields = len(self.obj_heights)


