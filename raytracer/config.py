"""Global configurations"""
import numpy as np

class Config():
    def __init__(self, max_obj_height: float, entrance_pupil_diameter: float, config_mode="EPD_and_height"):
        # TODO: These should be choices from a set of options (use argparse)
        if config_mode == "EPD_and_height":
            self._EPD = entrance_pupil_diameter # entrance pupil diameter
            self._max_obj_height = max_obj_height
            self._lens_unit = "mm"        

            self.obj_heights = np.array([self.max_obj_height, self.max_obj_height/np.sqrt(2.0), 0.0])
            self.MAX_OBJ_HEIGHT_INDEX = 0
            self.num_fields = len(self.obj_heights)
        else:
            raise NotImplementedError
    
        self.n_air = 1.0 # 1.000302 # which refractive index to use   
        self.wavelength = 0.588 # in micrometers, sodium Fraunhofer d-line
        self.num_rays = 7        
        self.ON_AXIS_FIELD_INDEX = -1  # last index
        
    @property
    def EPD(self):
        return self._EPD
    @EPD.setter
    def EPD(self, val):
        if val < 0:
            raise ValueError("Entrance pupil diameter must be positive number.")
        self._EPD = val

    @property
    def max_obj_height(self):
        return self._max_obj_height
    @max_obj_height.setter
    def max_obj_height(self, val):
        if val < 0:
            raise ValueError("Object height should be a positive number.")
        self._max_obj_height = val
        # update the object heights 
        self.obj_heights = np.array([self._max_obj_height, self._max_obj_height/np.sqrt(2.0), 0.0])

    @property
    def lens_unit(self):
        return self._lens_unit
    @lens_unit.setter
    def lens_unit(self, unit):
        if unit not in ["mm", "inch"]:
            raise ValueError("Invalid lens units")
        self._lens_unit = unit