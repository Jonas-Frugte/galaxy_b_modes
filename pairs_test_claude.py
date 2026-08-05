import numpy as np
import itertools
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- parameters
n_side   = 40                      # need >= 40 to get enough near-aligned pairs
n_halos  = n_side**3
rmax     = 2.0                     # chi_max in box units
pos_obs  = np.array([0.5, 0.5, 0.5])
sep_cut  = np.radians(10.0)        # only keep near-aligned pairs
replication_steps = 4

rng = np.random.default_rng(0)
coords_3d = rng.uniform(0, 1, size=(n_halos, 3))

print("STEP 1  halo positions")
print("   coords_3d.shape =", coords_3d.shape)
print("   range           = [%.4f, %.4f]" % (coords_3d.min(), coords_3d.max()))
print()

# no jitter: coords are already continuous random, jitter would only add
# an unseeded perturbation and push halos outside [0,1]

def delta_ang(v1s, v2s):
    return np.arccos(np.clip(
        np.einsum('ij,ij->i', v1s, v2s) /
        (np.linalg.norm(v1s, axis=1) * np.linalg.norm(v2s, axis=1)), -1, 1))

# ---------------------------------------------------------------- routes
possible_steps = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
])

routes_not_unique = np.concatenate([
    np.sum(np.array(list(itertools.combinations_with_replacement(possible_steps, i))), axis=1)
    for i in range(1, replication_steps + 1)])
routes = np.unique(routes_not_unique, axis=0).astype(float)

print("STEP 2  lattice routes")
print("   routes_not_unique.shape =", routes_not_unique.shape)
print("   routes.shape            =", routes.shape)
print("   origin present?         =", bool((np.abs(routes).sum(axis=1) == 0).any()))
need = rmax + np.sqrt(3) / 2
allv = np.array(list(itertools.product(range(-6, 7), repeat=3)), dtype=float)
needed = allv[np.linalg.norm(allv, axis=1) <= need]
missing = [v for v in needed if not (np.abs(routes - v).sum(axis=1) < 1e-9).any()]
print("   need |d| <= %.3f, MISSING routes = %d" % (need, len(missing)))
if missing:
    print("   -> raise replication_steps, the L1 ball is too small")
print()

# ---------------------------------------------------------------- images
imgs = coords_3d[None, :, :] + routes[:, None, :] - pos_obs
rs   = np.linalg.norm(imgs, axis=2)
inside = rs < rmax
u    = imgs / rs[:, :, None]

print("STEP 3  images")
print("   imgs.shape      =", imgs.shape, " (n_routes, n_halos, 3)")
print("   inside.shape    =", inside.shape)
print("   images per halo = %.2f  (analytic %.2f)" % (inside.sum()/n_halos, 4/3*np.pi*rmax**3))
print("   |u| == 1 ?      =", np.allclose(np.linalg.norm(u, axis=2), 1.0))
print()

print("STEP 4  pairing check")
i0, j0 = 0, 5
print("   routes[%d]-routes[%d] =" % (j0, i0), routes[j0] - routes[i0])
print("   same for every halo? =", np.allclose(imgs[j0] - imgs[i0], routes[j0] - routes[i0]))
print("   -> True means index k in two boxes is the same halo")
print()

# ---------------------------------------------------------------- special dirs
special_dirs = np.array([v for v in itertools.product([-1, 0, 1], repeat=3)
                         if v != (0, 0, 0)], dtype=float)
sd = special_dirs / np.linalg.norm(special_dirs, axis=1)[:, None]

print("STEP 5  special directions")
print("   sd.shape =", sd.shape)
print("   near-aligned pairs need |d| < rmax = %.1f" % rmax)
print("   -> these 26 dirs are complete only for rmax <= 2")
print()

# ---------------------------------------------------------------- pair loop
TH, PSI, DD, R1, R2 = [], [], [], [], []
n_inside = n_kept = 0

for i, j in itertools.combinations(range(len(routes)), 2):
    both_inside = inside[i] & inside[j]
    if not both_inside.any():
        continue
    n_inside += both_inside.sum()

    v1 = imgs[i][both_inside]
    v2 = imgs[j][both_inside]
    angs = delta_ang(v1, v2)

    keep = angs < sep_cut
    if not keep.any():
        continue
    n_kept += keep.sum()

    a = u[i][both_inside][keep]
    b = u[j][both_inside][keep]
    mid = a + b
    mid /= np.linalg.norm(mid, axis=1)[:, None]
    psi = np.arccos(np.clip(np.abs(mid @ sd.T).max(axis=1), -1, 1))

    r_i = rs[i][both_inside][keep]
    r_j = rs[j][both_inside][keep]

    TH.append(angs[keep])
    PSI.append(psi)
    DD.append(np.full(keep.sum(), np.linalg.norm(routes[j] - routes[i])))
    R1.append(np.minimum(r_i, r_j))
    R2.append(np.maximum(r_i, r_j))

TH  = np.concatenate(TH)
PSI = np.concatenate(PSI)
DD  = np.concatenate(DD)
R1  = np.concatenate(R1)
R2  = np.concatenate(R2)

print("STEP 6  pair loop")
print("   pairs with both images inside rmax =", n_inside)
print("   pairs with sep < %.0f deg           = %d" % (np.degrees(sep_cut), n_kept))
print("   TH.shape =", TH.shape, " PSI.shape =", PSI.shape)
if n_kept < 20000:
    print("   -> thin statistics, raise n_side")
print()

# ---------------------------------------------------------------- geometry check
pred = PSI * 2 * DD / (R1 + R2)
print("STEP 7  check  theta = psi_mid * 2|d| / (r1 + r2)")
print("   median |pred/actual - 1| = %.4f" % np.median(np.abs(pred / TH - 1)))
print("   90th pct                 = %.4f" % np.percentile(np.abs(pred / TH - 1), 90))
print("   -> should be ~1%%; if it is ~27%% you used psi of the near image, not the midpoint")
print()
print("   |d| contributing to near-aligned pairs:")
for v in np.unique(np.round(DD, 3)):
    f = 100 * (np.round(DD, 3) == v).mean()
    if f > 0.1:
        print("      |d| = %.3f : %5.1f%%" % (v, f))
print()

# ---------------------------------------------------------------- result
ell = np.pi / TH
psi_deg = np.degrees(PSI)

print("STEP 8  mask radius -> surviving multipoles")
print("   alpha[deg]  pairs_removed[%]  sky_removed[%]  max_ell  p99.9_ell  predicted")
for alpha in [0.25, 0.5, 1.0, 2.0, 5.0]:
    surv = psi_deg > alpha
    sky  = 100 * len(sd) * (1 - np.cos(np.radians(alpha))) / 2
    pred_ell = 180 * (2 * rmax - 1) / (2 * alpha)
    print("   %6.2f      %10.3f      %10.4f   %7.0f  %9.0f  %9.0f" % (
        alpha, 100 * (1 - surv.mean()), sky,
        ell[surv].max(), np.percentile(ell[surv], 99.9), pred_ell))

plt.hist(np.clip(ell, 0, 4000), bins=100, density=True)
plt.xlabel(r'$\ell = \pi/\theta$')
plt.show()