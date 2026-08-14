from lensing_pert_alt import synth
import healpy as hp
import numpy as np
from datetime import datetime

nside = 4096
lmax = 2 * nside

map = np.random.rand(hp.nside2npix(nside))
alms = hp.map2alm(map, lmax=lmax)

npoints_arr = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
for npoints in npoints_arr:
    # is not uniformly sampled on the sphere bc of theta but thats fine here
    loc = np.random.rand(npoints, 2)
    loc[:, 0] *= np.pi
    loc[:, 1] *= 2*np.pi

    time_i = datetime.now()
    synth(alms, spin=2, lmax=lmax, loc=loc, nthreads=4)
    time_f = datetime.now()
    print(f"for {npoints} points: {(time_f-time_i).total_seconds()} seconds runtime")
