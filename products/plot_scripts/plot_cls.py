from dataclasses import replace
import itertools
import numpy as np
import matplotlib.pyplot as plt

from b_modes_modules.cls_spec import ClSpec, Tracer, Mask

MODE_ROW = {"EE": 0, "EB": 1, "BE": 2, "BB": 3}
MODE_COLOR = {"EE": "C0", "BB": "C1", "EB": "C2", "BE": "C3"}
MASK_LS = {"none": "-", "special": "--", "random": ":"}
ALL_MODES = ("EE", "EB", "BE", "BB")


def spec_for_pair(spec: ClSpec, i: int, j: int) -> ClSpec:
    return replace(spec,
                   tracer_1=replace(spec.tracer_1, bin_num=i),
                   tracer_2=replace(spec.tracer_2, bin_num=j))


def load_pair(spec: ClSpec, i, j):
    s = spec_for_pair(spec, i, j)
    d = np.loadtxt(s.out_dir / s.filename)
    return d[-1], {m: d[MODE_ROW[m]] for m in ALL_MODES}


def load_all_bins(spec: ClSpec):
    '''Full sample, no tomography: sum over every bin pair / N^2.'''
    tot = None
    for i, j in itertools.combinations_with_replacement(range(spec.num_z_bins), 2):
        ells, cl = load_pair(spec, i, j)
        arr = (1 if i == j else 2) * np.array([cl[m] for m in ALL_MODES])
        tot = arr if tot is None else tot + arr
    return ells, dict(zip(ALL_MODES, tot / spec.num_z_bins**2))


def draw(ax, ells, cl, modes, mask_kind, label_mask):
    for m in modes:
        y = np.abs(cl[m]) if m in ("EB", "BE") else cl[m]
        label = f"{m} ({mask_kind})" if label_mask else m
        ax.loglog(ells, y, color=MODE_COLOR[m], ls=MASK_LS[mask_kind], label=label)


def decorate(ax, title):
    ax.set_title(title)
    ax.set_xlabel(r"$\ell$")
    ax.grid(alpha=0.3, which="both")
    ax.legend()


def finish(fig, axes, spec: ClSpec):
    axes[0].set_ylabel(r"$C_\ell$")
    fig.suptitle(spec.tag)
    fig.tight_layout()

    plots = spec.products.PLOTS
    plots.mkdir(parents=True, exist_ok=True)
    n = 0
    while (out := plots / f"cls_{n:03d}.png").exists():
        n += 1
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.show()


def plot_pairs(spec: ClSpec, pairs, masks=("none",), modes=("EE", "BB", "EB")):
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (i, j) in zip(axes, pairs):
        for mask_kind in masks:
            s = replace(spec, mask=replace(spec.mask, kind=mask_kind))
            draw(ax, *load_pair(s, i, j), modes, mask_kind, len(masks) > 1)
        decorate(ax, f"bin {i} x {j}")
    finish(fig, axes, spec)


def plot_all_bins(spec: ClSpec, masks=("none",), modes=("EE", "BB", "EB")):
    fig, axes = plt.subplots(1, 1, figsize=(6, 4.5))
    axes = np.atleast_1d(axes)
    for mask_kind in masks:
        s = replace(spec, mask=replace(spec.mask, kind=mask_kind))
        draw(axes[0], *load_all_bins(s), modes, mask_kind, len(masks) > 1)
    decorate(axes[0], "all bins combined")
    finish(fig, axes, spec)


if __name__ == "__main__":
    spec = ClSpec(tracer_1=Tracer(bin_num=0), tracer_2=Tracer(bin_num=0))
    plot_pairs(spec, pairs=[(2, 2), (2, 5)])
