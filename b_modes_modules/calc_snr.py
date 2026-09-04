from dataclasses import replace
import itertools
import numpy as np

from b_modes_modules.cls_spec import ClSpec, Tracer

ARCMIN2_PER_STERADIAN = (180 / np.pi)**2 * 3600

def shape_noise_euclid(spec: ClSpec) -> float:
    n_per_sr = spec.n_gal_per_arcmin2 * ARCMIN2_PER_STERADIAN / spec.num_z_bins
    return spec.sigma_e**2 / n_per_sr

def bin_pairs(spec: ClSpec):
    return list(itertools.combinations_with_replacement(range(spec.num_z_bins), 2))

def spec_for_pair(spec: ClSpec, i: int, j: int) -> ClSpec:
    '''Same run (same tag / out_dir), different bin pair.'''
    return replace(spec,
                   tracer_1=replace(spec.tracer_1, bin_num=i),
                   tracer_2=replace(spec.tracer_2, bin_num=j))

def calc_snr(spec: ClSpec, lmax: float) -> float:
    pairs = bin_pairs(spec)
    npairs = len(pairs)
    noise = shape_noise_euclid(spec)

    ls = np.loadtxt(spec_for_pair(spec, 0, 0).out_dir / spec_for_pair(spec, 0, 0).filename)[-1]
    keep = ls <= lmax
    ls = ls[keep]

    cls = np.zeros((len(ls), npairs))
    for p, (i, j) in enumerate(pairs):
        s = spec_for_pair(spec, i, j)
        cls[:, p] = np.loadtxt(s.out_dir / s.filename)[3][keep]  # BB, already noise-subtracted

    pair_index = {pr: idx for idx, pr in enumerate(pairs)}
    def C(a, b):
        return cls[:, pair_index[(min(a, b), max(a, b))]]
    def N(a, b):
        return noise if a == b else 0.0

    denom = (2 * ls + 1) * spec.f_sky * spec.nlb
    cls_cov = np.zeros((len(ls), npairs, npairs))
    for p1, (a, b) in enumerate(pairs):
        for p2, (c, d) in enumerate(pairs):
            cls_cov[:, p1, p2] = ((C(a, c) + N(a, c)) * (C(b, d) + N(b, d)) +
                                  (C(a, d) + N(a, d)) * (C(b, c) + N(b, c))) / denom

    x = np.linalg.solve(cls_cov, cls[:, :, np.newaxis])[:, :, 0]
    return np.sqrt(np.sum(cls * x))

if __name__ == "__main__":
    spec = ClSpec(tracer_1=Tracer(bin_num=0), tracer_2=Tracer(bin_num=0))
    spec.products.SNR.mkdir(parents=True, exist_ok=True)

    for lmax in [200, 500, 1000, 10_000]:
        snr = calc_snr(spec, lmax)
        out_path = spec.products.SNR / f"{spec.tag}_lmax{lmax}.txt"
        np.savetxt(out_path, [snr])
        print(f"{spec.tag}, lmax={lmax}: SNR={snr:.3f}, wrote {out_path}")
