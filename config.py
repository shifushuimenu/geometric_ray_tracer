"""Global configurations"""
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Config():
    # TODO: These should be choices from a set of options (use argparse)
    EPD: float # entrance pupil diameter
    n_air: float = 1.0 # 1.000302 # which refractive index to use
    lens_unit: str = "mm"
    obj_height: float = field(default_factory = np.ndarray) # arraz of field positions
    wavelength: float = 0.588 # in micrometers, sodium Fraunhofer d-line

    def __post_init__(self):
        self.max_obj_height = np.max(self.obj_height)
        self.num_fields = len(self.obj_height)
