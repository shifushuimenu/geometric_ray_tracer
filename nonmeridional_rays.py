"""Ray tracing of rays that do not lie in the tangential plane spanned by the object height and the optical axis (= non-meridional rays)"""
from math import sqrt, cos, sin
import numpy as np
from utils import timer_func

@timer_func
def raytrace_nonmeridional_rays_v2(zS, R, n, P_intersect, rayvecs):
    # slow version which does not use broadcasting 
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


@timer_func
def raytrace_nonmeridional_rays(zS, R, n, P_intersect, rayvecs):
    assert R.shape == n.shape
    num_surfs = n.shape[0]
    assert zS.shape == (num_surfs+1,)
    assert P_intersect.shape == rayvecs.shape
    # num_rays = P_intersect.shape[2]
    # num_fields = P_intersect.shape[3] 
    # in the future: num_wavelengths = p_intersect.shape[4]
    
    dims = len(P_intersect.shape)-1

    # P_intersect = np.transpose(P_intersect, axes=(0,2,3,1))
    # rayvecs = np.transpose(rayvecs, axes=(0,2,3,1))

    for i in range(1, num_surfs+1):
        # Calculate the intersection point with surface i
        xO, yO, zO = P_intersect[0:3,i-1,...]    
        xr, yr, zr = rayvecs[0:3,i-1,...]

        if i < num_surfs and np.abs(R[i]) != np.inf:
            # spherical surface 
            b = 2*xO*xr + 2*yO*yr + 2*(zO - (zS[i]+R[i]))*zr
            c = xO**2 + yO**2 + (zO-zS[i])**2 - 2*(zO - zS[i])*R[i]
            if R[i] > 0:
                alpha = 0.5*(-b - np.sqrt(b**2 - 4*c))
            elif R[i] < 0:
                alpha = 0.5*(-b + np.sqrt(b**2 - 4*c))
        else:
            # flat surface 
            alpha = (zS[i] - zO)/zr # zr is never zero by construction

        P_intersect[0:3,i,...] = P_intersect[0:3,i-1,...] + alpha*rayvecs[0:3,i-1,...]

        # Calculate the new normalized ray vector using Snell's law
        if i < num_surfs and np.abs(R[i]) != np.inf:
            N_pt = P_intersect[0:3,i,...] - np.array([0,0,zS[i] + R[i]]).reshape((3,)+(1,)*(dims-1))
            surface_norm_pt = N_pt / np.linalg.norm(N_pt, axis=0)
            einsum_str = "ijklmno"[0:dims]+","+"ijklmno"[0:dims]+"->"+"ijklmno"[1:dims]
            rn = np.einsum(einsum_str, rayvecs[0:3,i-1,...], surface_norm_pt[0:3,...]) # dot product over first axis
            n_ratio = n[i-1]/n[i]
            pref = -( np.sign(R[i])*np.sqrt(1-(n_ratio)**2 * (1 - rn**2)) + n_ratio*rn )
            rayvecs[0:3,i,...] = pref*surface_norm_pt + n_ratio*rayvecs[0:3,i-1,...]
            assert np.isclose(np.linalg.norm(rayvecs[0:3,i,...], axis=0), 1.0, atol=1e-8).all(), f"ray vector not normalized"
        else:
            rayvecs[0:3,i,...] = rayvecs[0:3,i-1,...]

    # undo reordering of axes
    # P_intersect = np.transpose(P_intersect, axes=(0,3,1,2))
    # rayvecs = np.transpose(rayvecs, axes=(0,3,1,2))

    return P_intersect, rayvecs

@timer_func
def calculate_OPD(n, P_intersect):
    """Calculate optical path difference (OPD) relative to the chief ray of a ray bundle."""

    n = np.asarray(n)
    P_intersect = np.asarray(P_intersect)
    _, tmp, num_rays, num_fields = P_intersect.shape
    num_surfs = tmp - 1
    # Image surface has optical path zero.
    P_diff = np.zeros((3, num_surfs, num_rays, num_fields))    
    # optical path segments OPS between surfaces
    P_diff[:,0:,...] = np.diff(P_intersect[:,:,...],axis=1)
    OPS = np.linalg.norm(P_diff,axis=0) * n[:,np.newaxis,np.newaxis]
    # Cumulative optical path up to a given surface.    
    OP = np.cumsum(OPS, axis=0)    
    # Optical path difference relative to the chief ray.
    OPD = np.zeros_like(OP)
    # for f in range(num_fields):
    #     for r in range(num_rays):
    #         OPD[:,r,f] = OP[:,r,f] - OP[:,num_rays//2,f]
    OPD[:,:,...] = OP[:,:,...] - OP[:,num_rays//2,:][:,np.newaxis,...]            

    return OPD