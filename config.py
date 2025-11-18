"""Global configurations"""
from dataclasses import dataclass, field

@dataclass
class Config():
    # TODO: These should be choices from a set of option
    n_air: float = 1.000302 # which refractive index to use
    lens_unit: str = "mm"