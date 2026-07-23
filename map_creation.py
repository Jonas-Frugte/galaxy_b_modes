import numpy as np
import healpy as hp
import h5py
import matplotlib.pyplot as plt
import pymaster as nmt
from born_lps_analytic import lps

def create_cl(filenames, nside, order=1):
    # these will be list of lists, each sublist corresponding to data of a single shell
    coords = []
    e1 = []
    e2 = []

    if order==1:
        coords_ds_name = "halo_coords_lensed"
        shapes_ds_name = "projected_tensors_lensed"
    elif order==2:
        coords_ds_name = "halo_coords_lensed_2o"
        shapes_ds_name = "projected_tensors_lensed_2o"
    for filename in filenames:
        with h5py.File(filename, "r") as data:
            data_coords = data[coords_ds_name][:]
            coords.append(data_coords)
            if len(data_coords) > 0:
                e1.append(data[shapes_ds_name][:, 0])
                e2.append(data[shapes_ds_name][:, 1])

    # join into one long list and automatically convert to np array as well
    coords = np.concatenate(coords, axis=0)
    e1 = np.concatenate(e1)
    e2 = np.concatenate(e2)

    npix = hp.nside2npix(nside)

    # tells you which pixel each coordinate vector lies in, for a list of coordinates gives a list of pixels
    pix = hp.vec2pix(nside, coords[:, 0], coords[:, 1], coords[:, 2])

    sum_e1 = np.bincount(pix, weights=e1, minlength=npix)
    sum_e2 = np.bincount(pix, weights=e2, minlength=npix)
    count = np.bincount(pix, minlength=npix)

    e1_map = np.zeros(npix)
    e2_map = np.zeros(npix)
    hit = count > 0 # array of bools
    e1_map[hit] = sum_e1[hit] / count[hit]
    e2_map[hit] = sum_e2[hit] / count[hit]

    lmax = 3 * nside - 1 # TODO: check this later
    # takes two maps, first map is Q, second is U, then calculates spin-2 raised a_lm for spin-2 field
    # f = Q + i * U, then converts to a_lm of E and B modes and hands back two lists containing them
    alm_E, alm_B = hp.map2alm_spin([e1_map, e2_map], spin=2, lmax=lmax)

    cl_EE = hp.alm2cl(alm_E)
    cl_BB = hp.alm2cl(alm_B)
    cl_EB = hp.alm2cl(alm_E, alm_B)

    mask = (count > 0).astype(float) # TODO: this is just a on/off mask, probably want to use count as a weight itself
    field2 = nmt.NmtField(mask, [e1_map, e2_map], purify_e=False, purify_b=False, beam=hp.pixwin(nside, pol=True)[1]) # TODO: read about hp.pixwin
    b = nmt.NmtBin.from_nside_linear(nside, nlb=8)
    # calculate coupling matrix, depends only on mask and binning
    w = nmt.NmtWorkspace.from_fields(field2, field2, b)
    # calculate coupled c_ell then decouple
    cl = w.decouple_cell(nmt.workspaces.compute_coupled_cell(field2, field2)) # can add bias and noise here

    shape_noise = np.array([(np.std(e1)**2 + np.std(e2)**2) * 0.5 / (len(coords) / (4 * np.pi)) for ell in b.get_effective_ells()])

    return b.get_effective_ells(), cl, shape_noise

if __name__ == "__main__":
    nside = 2**8

    ells, cl, nl = create_cl(["data/mock_catalogue_s0.00_1e6gals_z1.0.hdf5",], nside)
    plt.loglog(ells, cl[0]-nl, label = "EE - noise")
    plt.loglog(ells, nl, label = "noise floor")
    plt.loglog(ells, cl[3], label = "BB")
    plt.loglog(ells, np.abs(cl[1]), label = "|EB|", linestyle = ":")

    from rotation_lps import get_kappa_power
    ls, cl = get_kappa_power(1.0)
    plt.loglog(ls, cl, label="CAMB ee")
    # ells_2o, cl_2o, nl_2o = create_cl(["data/mock_catalogue_s0.00_1e7gals.hdf5",], nside, order=2)
    # plt.loglog(ells_2o, cl_2o[0]-nl_2o, label = "EE - noise, 2o", linestyle = '--')
    # plt.loglog(ells_2o, nl_2o, label = "noise floor, 2o", linestyle = '--')
    # plt.loglog(ells_2o, cl_2o[3], label = "BB, 2o", linestyle = '--')
    # plt.loglog(ells_2o, np.abs(cl_2o[1]), label = "|EB|, 2o", linestyle = ":")

    # plt.loglog(ells, np.abs(cl[1]), label="|EB|")
    # plt.loglog(ells(), np.abs(cl[1]), label = "|EB|")
    # plt.loglog(ells(), cl[3], label = "BB")

    # ells_analytical = np.logspace(np.log10(2), np.log10(ells[-1]), 5)
    # lps_vals = [lps(l) for l in ells_analytical]

    # plt.loglog(ells_analytical, lps_vals, label = "analytical")

    plt.legend()
    plt.title(f"nside={nside}")
    plt.show()