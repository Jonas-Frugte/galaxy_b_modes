from matter_power_spectrum_interpolater import Pk
import matplotlib.pyplot as plt
import numpy as np

ks = np.loadtxt("data/L1_m9/power_spectra/power_matter_0122.txt")[:, 1]
Pks = [Pk(k, 0)[0] for k in ks]
plt.loglog(ks, Pks)

from rotation_lps import camb_nonlinear_pk
ks, Pks = camb_nonlinear_pk(z=0.0, ks=ks)
plt.loglog(ks, Pks, linestyle=":")

plt.show()