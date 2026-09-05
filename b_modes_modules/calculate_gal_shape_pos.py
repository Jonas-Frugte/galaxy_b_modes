import h5py
import numpy as np
import healpy as hp
from b_modes_modules.cls_spec import Tracer
from b_modes_modules.lens_spec import LensSpec

def new_coords(theta_old, phi_old, radius_old, delta_angle):
    theta_new = theta_old + delta_angle[:, 0]
    # this is not stable for gals near pole, probably good to eventually change to a rotation matrix method
    phi_new   = phi_old + delta_angle[:, 1] / np.sin(theta_old)   # orthonormal -> coordinate

    # fold theta into [0, pi]; crossing a pole flips phi by pi
    theta_new = theta_new % (2 * np.pi)
    crossed_pole = theta_new > np.pi
    theta_new[crossed_pole] = 2 * np.pi - theta_new[crossed_pole]
    phi_new[crossed_pole] += np.pi
    phi_new = phi_new % (2 * np.pi)

    return hp.ang2vec(theta_new, phi_new) * radius_old[:, np.newaxis]

# TODO: check references (seitz @ schneider 1997 and Bartelmann & Schneider 2001) for correctness of this
def new_shape(shapes_old, psi_ij):
    '''
    e = 2g(1+iw)/(1+g^2+w^2), where g = gamma/(1-kappa), w = omega/(1-kappa)
    at leading order we get e = 2g
    '''

    e1 = shapes_old[:, 0]                          # (ngal,)
    e2 = shapes_old[:, 1]
    ngal = len(e1)

    # reconstruct Q up to overall scale (T := 1), batched -> (ngal, 2, 2)
    Q = np.empty((ngal, 2, 2))
    Q[:, 0, 0] = 0.5 * (1.0 + e1)
    Q[:, 0, 1] = 0.5 * e2
    Q[:, 1, 0] = 0.5 * e2
    Q[:, 1, 1] = 0.5 * (1.0 - e1)

    A = np.eye(2)[np.newaxis, :, :] - psi_ij             # broadcasts (2, 2) -> (1, 2, 2) -> (ngal, 2, 2)
    A_inv = np.linalg.inv(A)                       # batched inverse, (ngal, 2, 2)

    # Q_lensed = A_inv @ Q @ A_inv^T, per galaxy
    Q_lensed = A_inv @ Q @ np.transpose(A_inv, (0, 2, 1))   # (ngal, 2, 2)

    T = Q_lensed[:, 0, 0] + Q_lensed[:, 1, 1]      # (ngal,)
    out = np.empty((ngal, 2))
    out[:, 0] = (Q_lensed[:, 0, 0] - Q_lensed[:, 1, 1]) / T
    out[:, 1] = 2.0 * Q_lensed[:, 0, 1] / T
    return out

def get_gal_shape_pos(psi_ij, delta_angle, tracer: Tracer, lens_spec: LensSpec):
    with h5py.File(tracer.filepaths.SHELLS_RESOLVED, "r") as f:
        pos_old = f["halo_coords"][:]
        if tracer.IA == "real":
            shapes_old = f[lens_spec.shape_type][:]
        elif tracer.IA == "off":
            shapes_old = np.zeros((len(pos_old), 2))   # perfectly circular: pure lensing signal, no IA

    radius_old = np.linalg.norm(pos_old, axis=1)
    theta_old, phi_old = hp.vec2ang(pos_old)

    shapes_new = new_shape(shapes_old, psi_ij)
    pos_new = new_coords(theta_old, phi_old, radius_old, delta_angle)

    return shapes_new, pos_new
