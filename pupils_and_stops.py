import numpy as np
from typing import Tuple
from trace_ray import trace_tangential_ray
from utils import RayIntersectionNotFoundError, ChiefRayNotFoundError

def find_chief_rays(lens_sequence, obj_height, u_max_start = np.pi/3, eps=1e-6):
    """
    For all field positions given in `obj_height[0:num_fields]`, find the chief ray launch 
    angles at the aperture stop. The launch angles are modified using a binary search 
    and the ray is traced backwards from the center of the aperture stop until it hits the object position within `eps` tolerance
    (in lens units).

    IMPORTANT Note: It is assumed that increasing the chief ray launch angle in positive direction will monotonically 
    increase the ray height in the object plane. This is not always the case, especially for non-physical surfaces,
    i.e. lens elements outside their clear apertures where front and back surface are inverted. 
    """
    obj_height = np.asarray(obj_height) #
    assert (obj_height >= 0).all()      #
    num_fields = len(obj_height)
    y_cr = np.zeros((lens_sequence.num_surfs+1, num_fields))
    u_cr = np.zeros((lens_sequence.num_surfs+1, num_fields))
    z_sag_cr = np.zeros((lens_sequence.num_surfs+1, num_fields))

    # Same maximum chief ray launch angle for all field positions.
    u_min_start = -u_max_start 

    for f in range(num_fields):
        print("field=", f)  
        if obj_height[f] == 0.0:
            # on-axis field point: y_cr and u_cr are zero
            continue
        
        u_min = u_min_start
        u_max = u_max_start

        y_cr[lens_sequence.AS_surf,f] = 0.0

        # Choose the initial cone of angles 
        INTERSECTION_FOUND = False
        while(not INTERSECTION_FOUND):
            try:
                y, u, z_sag, _ = trace_tangential_ray(0.0, u_max, lens_sequence, lens_sequence.AS_surf, forward=False)
            except RayIntersectionNotFoundError:
                # make the opening angle of the cone within which the chief ray launch angle is searched smaller
                u_max -= 0.1*u_max
                u_min -= 0.1*u_min
                print(f"RayIntersectionNotFoundError u_min={u_min}, u_max={u_max}")
                continue
            INTERSECTION_FOUND = True
            print("INTERSECTION FOUND")

        if y[0] < obj_height[f]: 
            print("y[0]=", y[0])
            raise ChiefRayNotFoundError(f"Object height {obj_height[f]} is too large, it cannot be reached "
                             f"from the aperture stop with any chief ray launch angle. Consider adjusting u_max.")

        CHIEF_RAY_FOUND = False
        print("determining chief ray launch angle")        
        while(not CHIEF_RAY_FOUND):
            # Update launch angle using bisection search.
            # It is assumed that increasing the chief ray launch angle will 
            # monotonically increase its height in object space.
            u_middle = 0.5*(u_max + u_min)                  
            # By definition, at the aperture stop, the chief ray intersects the optical axis.
            y, u, z_sag, _ = trace_tangential_ray(0.0, u_middle, lens_sequence, lens_sequence.AS_surf, forward=False)
            # criterion whether chief ray has been found
            CHIEF_RAY_FOUND = np.isclose(y[0], obj_height[f], atol=eps)
            # update bracketing interval for binary search            
            if y[0] < obj_height[f]:
                print(f"u_min = {u_min} and u_max = {u_max}")
                print(f"u_middle = {u_middle}")                          
                print(f"y_cr[0,{f}] < obj_height[{f}]")
                u_min = u_middle 
            else:
                print(f"u_min = {u_min} and u_max = {u_max}")
                print(f"u_middle = {u_middle}")
                print(f"y_cr[0,{f}] > obj_height[{f}]")
                u_max = u_middle
        y_cr[0:lens_sequence.AS_surf+1, f] = y[0:lens_sequence.AS_surf+1]
        u_cr[0:lens_sequence.AS_surf+1, f] = u[0:lens_sequence.AS_surf+1]
        z_sag_cr[0:lens_sequence.AS_surf+1, f] = z_sag[0:lens_sequence.AS_surf+1]

    return y_cr, u_cr, z_sag_cr

def intersection_line_segments(y1: float, u1: float, y2: float, u2: float, z0: float) -> Tuple[float, float]:
    z_int = z0 - (y1 -y2)/(np.tan(u1) - np.tan(u2))
    y_int = y1 + np.tan(u1)*(z_int-z0)
    return z_int, y_int