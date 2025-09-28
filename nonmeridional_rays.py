"""Ray tracing of rays that do not lie in the tangential plane spanned by the object height and the optical axis (= non-meridional rays)"""
from math import sqrt, cos, sin
import numpy as np

def raytrace_nonmeridional_rays(zS, R, n, P_intersect, rayvecs):
    assert R.shape == n.shape
    num_surfs = n.shape[0]
    assert zS.shape == (num_surfs+1,)
    assert P_intersect.shape == rayvecs.shape
    num_rays = P_intersect.shape[2]
    num_fields = P_intersect.shape[3]    
    
    for f in range(num_fields):
        for r in range(num_rays):
            for i in range(1, num_surfs+1):
                # Calculate the intersection point with surface i
                xO, yO, zO = P_intersect[0:3,i-1,r,f]    
                xr, yr, zr = rayvecs[0:3,i-1,r,f]

                if i < num_surfs and np.abs(R[i]) != np.inf:
                    # spherical surface 
                    b = 2*xO*xr + 2*yO*yr + 2*(zO - (zS[i]+R[i]))*zr
                    c = xO**2 + yO**2 + (zO-zS[i])**2 - 2*(zO - zS[i])*R[i]
                    print("b=", b, "c=", c)
                    if R[i] > 0:
                        alpha = 0.5*(-b - np.sqrt(b**2 - 4*c))
                    elif R[i] < 0:
                        alpha = 0.5*(-b + np.sqrt(b**2 - 4*c))
                else:
                    # flat surface 
                    alpha = (zS[i] - zO)/zr

                P_intersect[0:3,i,r,f] = P_intersect[0:3,i-1,r,f] + alpha*rayvecs[0:3,i-1,r,f]

                # Calculate the new normalized ray vector using Snell's law
                if i < num_surfs and np.abs(R[i]) != np.inf:
                    N_pt = P_intersect[0:3,i,r,f] - np.array([0,0,zS[i] + R[i]])
                    surface_norm_pt = N_pt / np.linalg.norm(N_pt)
                    rn = np.dot(rayvecs[0:3,i-1,r,f], surface_norm_pt[0:3])
                    n_ratio = n[i-1]/n[i]
                    rayvecs[0:3,i,r,f] = -( np.sign(R[i])*sqrt(1-(n_ratio)**2 * (1 - rn**2)) + n_ratio*rn )*surface_norm_pt + n_ratio*rayvecs[0:3,i-1,r,f]
                    assert np.isclose(np.linalg.norm(rayvecs[0:3,i,r,f]), 1.0, atol=1e-8), f"ray vector not normalized, norm={np.linalg.norm(rayvecs[0:3,i,r,f])}"
                else:
                    rayvecs[0:3,i,r,f] = rayvecs[0:3,i-1,r,f]

    return P_intersect, rayvecs
