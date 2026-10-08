r"""
3D categorical simulation with folded facies
--------------------------------------------

Direct Sampling is not limited to 2D images. A training image can be any
structured array, including a 3D categorical volume. This example uses the
GAIA-UNIL ``fold_categorical`` volume as a larger binary 3D training image and
visualizes a saved Direct Sampling result with cutaway voxel views and
orthogonal slice plots.

The bundled output is generated with a stricter offline-quality setup on a
64x64x48 grid: ``n_neighbors=100``, ``max_radius=20``,
``scan_fraction=0.5``, ``threshold=0.0``, seed 10, and four threads. The
simulation is intentionally expensive, so normal gallery execution displays
precomputed figures. Set ``GSTOOLS_GENERATE_EXAMPLE_OUTPUT=1`` to regenerate
them.

.. note::

    **Data source / license.** The bundled training image is stored in
    ``input/gaia_unil_fold_categorical_3d.npz`` and was prepared from
    ``MPS_book_data/Part2/fold_categorical.zip`` in the
    `GAIA-UNIL trainingimages repository
    <https://github.com/GAIA-UNIL/trainingimages>`_, distributed under
    **GPL-3.0**. Full redistribution notices are in ``input/LICENSE.txt``.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import gstools as gs

example_dir = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
)
data_path = example_dir / "input" / "gaia_unil_fold_categorical_3d.npz"
output_dir = example_dir / "output"
output_paths = {
    "3d": output_dir / "three_dimensional_fold_facies_3d.png",
    "slices": output_dir / "three_dimensional_fold_facies_slices.png",
    "proportions": output_dir / "three_dimensional_fold_facies_proportions.png",
    "correlation": output_dir / "three_dimensional_fold_facies_two_point_correlation.png",
}

FACIES_COLORS = ["#d8c69f", "#1d76a8"]
FACIES_LABELS = ["0 background facies", "1 folded facies"]
FACIES_CMAP = ListedColormap(FACIES_COLORS)
SIMULATION_SHAPE = (64, 64, 48)
N_NEIGHBORS = 100
MAX_RADIUS = 20
SCAN_FRACTION = 0.5
THRESHOLD = 0.0
SEED = 10
NUM_THREADS = 4
VALIDATION_MAX_LAG = 32


###############################################################################
# **Small display and plotting helpers.**
#
# These helpers keep the gallery body compact: each section below either saves
# one figure when regeneration is requested, or displays the corresponding
# bundled PNG.
def show_saved_figure(path, width=10.5):
    """Display a precomputed PNG in the gallery."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing precomputed figure: {path}. "
            "Set GSTOOLS_GENERATE_EXAMPLE_OUTPUT=1 to create it."
        )
    image = plt.imread(path)
    height = width * image.shape[0] / image.shape[1]
    fig, ax = plt.subplots(figsize=(width, height))
    ax.imshow(image)
    ax.axis("off")
    fig.tight_layout(pad=0)


def save_figure(fig, path):
    """Save one gallery figure and close the Matplotlib object."""
    output_dir.mkdir(exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    print(f"Saved {path}.")
    plt.close(fig)


def facies_handles():
    """Legend handles shared by the slice and bar-plot figures."""
    return [
        Patch(facecolor=color, edgecolor="0.35", label=label)
        for color, label in zip(FACIES_COLORS, FACIES_LABELS)
    ]


def decimate_volume(volume, step=4):
    """Return a smaller volume for responsive Matplotlib voxel plotting."""
    return volume[::step, ::step, ::step]


def cutaway_mask(active):
    """Hide one corner so internal facies geometry stays visible."""
    ix, iy, iz = np.indices(active.shape)
    return (
        (ix > 0.52 * active.shape[0])
        & (iy < 0.50 * active.shape[1])
        & (iz > 0.42 * active.shape[2])
    )


def plot_voxels(ax, volume, title):
    """Plot a cutaway view of facies 1."""
    active = (volume == 1) & ~cutaway_mask(volume.astype(bool))
    colors = np.empty(active.shape, dtype=object)
    colors[:] = "#00000000"
    colors[active] = FACIES_COLORS[1] + "cc"
    ax.voxels(
        active,
        facecolors=colors,
        edgecolor="#0b3f5755",
        linewidth=0.02,
    )
    ax.set_title(title, pad=8)
    ax.set_box_aspect(volume.shape)
    ax.set_axis_off()
    ax.set_proj_type("ortho")
    ax.view_init(elev=22, azim=-48)
    ax.set_xlim(0, volume.shape[0])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[2])


def center_slices(volume):
    """Return orthogonal centre slices from a 3D array."""
    cx, cy, cz = (size // 2 for size in volume.shape)
    return [
        (volume[cx, :, :].T, f"x={cx}"),
        (volume[:, cy, :].T, f"y={cy}"),
        (volume[:, :, cz].T, f"z={cz}"),
    ]


def plot_slices(row_axes, volume, prefix):
    """Plot the three centre slices of a categorical volume."""
    for ax, (slc, title) in zip(row_axes, center_slices(volume)):
        ax.imshow(
            slc,
            cmap=FACIES_CMAP,
            origin="lower",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"{prefix} {title}")
        ax.set_xticks([])
        ax.set_yticks([])


def facies_proportions(volume):
    """Return binary facies proportions."""
    counts = np.bincount(np.asarray(volume, dtype=int).ravel(), minlength=2).astype(float)
    return counts[:2] / counts[:2].sum()


def two_point_correlation(volume, facies=1, max_lag=VALIDATION_MAX_LAG, axis=0):
    """Indicator two-point correlation for one facies and one axis."""
    indicator = np.asarray(volume) == facies
    max_lag = min(max_lag, indicator.shape[axis] - 1)
    values = []
    for lag in range(max_lag + 1):
        if lag == 0:
            corr = indicator.mean()
        else:
            left = [slice(None)] * indicator.ndim
            right = [slice(None)] * indicator.ndim
            left[axis] = slice(None, -lag)
            right[axis] = slice(lag, None)
            corr = (indicator[tuple(left)] & indicator[tuple(right)]).mean()
        values.append(corr)
    values = np.asarray(values, dtype=float)
    if values[0] > 0:
        values /= values[0]
    return values


###############################################################################
# **Load the 3D folded facies training image.**
#
# The bundled array is a binary categorical volume prepared from the GAIA-UNIL
# SGEMS file. It is stored locally so this example never downloads data during
# normal execution or regeneration.
if not data_path.exists():
    raise FileNotFoundError(
        f"Missing bundled training image: {data_path}. "
        "Run this example from examples/13_mps or restore the input assets."
    )

with np.load(data_path) as data:
    ti_data = data["facies"].astype(int)

assert ti_data.shape == (180, 150, 120)
assert set(np.unique(ti_data)) == {0, 1}

###############################################################################
# **Configure the 3D data event.**
#
# This example uses strict DSBC-style settings for the saved high-quality
# output. The gallery path still stays fast because the expensive simulation is
# only recomputed when explicitly requested with an environment variable.
ti = gs.TrainingImage(
    ti_data,
    categorical=True,
    n_neighbors=N_NEIGHBORS,
    max_radius=MAX_RADIUS,
)
model = gs.MPSModel(
    ti,
    scan_fraction=SCAN_FRACTION,
    threshold=THRESHOLD,
    boundary="partial",
)
ds = gs.DirectSampling(model)

grid = [np.arange(size, dtype=float) for size in SIMULATION_SHAPE]

###############################################################################
# **Run Direct Sampling, or reuse the saved figures.**
#
# The setup above always runs so the example remains readable. The simulation
# itself runs only when the environment variable below is set, because the
# strict settings are intentionally expensive.
generate_output = os.environ.get("GSTOOLS_GENERATE_EXAMPLE_OUTPUT") == "1"
if generate_output:
    print(f"Simulating 3D folded facies field {SIMULATION_SHAPE}...")
    field = np.rint(ds(grid, seed=SEED, num_threads=NUM_THREADS)).astype(int)
    assert field.shape == SIMULATION_SHAPE
    assert set(np.unique(field)) == {0, 1}

###############################################################################
# **3D comparison.**
#
# The 3D view makes the fold geometry visible as a volume instead of only as
# flat slices.
def plot_3d_views(ti_data, field):
    """Compare cutaway voxel renderings of the TI and simulation."""
    fig = plt.figure(figsize=(11.0, 5.4), constrained_layout=True)
    ax_ti = fig.add_subplot(1, 2, 1, projection="3d")
    ax_sim = fig.add_subplot(1, 2, 2, projection="3d")
    plot_voxels(ax_ti, decimate_volume(ti_data, step=4), "Training image")
    plot_voxels(ax_sim, decimate_volume(field, step=2), "Simulation")
    fig.suptitle("Folded facies in 3D", fontsize=14)
    return fig


if generate_output:
    save_figure(plot_3d_views(ti_data, field), output_paths["3d"])
show_saved_figure(output_paths["3d"], width=10.5)

###############################################################################
# **Orthogonal slices.**
#
# The same centre slices are extracted from the training image and the simulated
# field. This is useful for checking whether the simulated fold continuity still
# looks like the training image.
def plot_slice_comparison(ti_data, field):
    """Compare centre slices through the TI and simulation."""
    fig = plt.figure(figsize=(11.5, 6.6), constrained_layout=True)
    layout = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.16])
    ti_axes = [fig.add_subplot(layout[0, i]) for i in range(3)]
    sim_axes = [fig.add_subplot(layout[1, i]) for i in range(3)]
    legend_ax = fig.add_subplot(layout[2, :])
    plot_slices(ti_axes, ti_data, "TI")
    plot_slices(sim_axes, field, "Simulation")
    legend_ax.axis("off")
    legend_ax.legend(
        handles=facies_handles(),
        loc="center",
        ncol=len(FACIES_LABELS),
        frameon=False,
        fontsize=10,
    )
    return fig


if generate_output:
    save_figure(plot_slice_comparison(ti_data, field), output_paths["slices"])
show_saved_figure(output_paths["slices"], width=10.5)

###############################################################################
# **Facies proportions.**
#
# A good categorical simulation should preserve the overall amount of each
# facies. Here each stacked bar is one complete volume.
def plot_proportion_comparison(ti_data, field):
    """Compare binary facies proportions as stacked TI/simulation bars."""
    proportions = np.vstack([facies_proportions(ti_data), facies_proportions(field)])
    x = np.arange(2)
    bottoms = np.zeros(2)
    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    for facies, color in enumerate(FACIES_COLORS):
        values = proportions[:, facies]
        ax.bar(
            x,
            values,
            bottom=bottoms,
            width=0.58,
            color=color,
            edgecolor="0.25",
            linewidth=0.35,
        )
        for xpos, bottom, value in zip(x, bottoms, values):
            if value >= 0.035:
                ax.text(
                    xpos,
                    bottom + value / 2,
                    f"{100 * value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="0.08",
                )
        bottoms += values
    ax.set_title("Facies proportions", fontsize=13)
    ax.set_ylabel("fraction of cells")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(["Training image", "Simulation"])
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.legend(
        handles=facies_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=len(FACIES_LABELS),
        frameon=False,
        fontsize=10,
    )
    return fig


if generate_output:
    save_figure(plot_proportion_comparison(ti_data, field), output_paths["proportions"])
show_saved_figure(output_paths["proportions"], width=8.2)

###############################################################################
# **Two-point correlation.**
#
# The two-point correlation checks whether the folded facies has a comparable
# spatial continuity in the training image and in the simulation. Solid lines
# are the simulation; dashed lines are the training image.
def plot_two_point_comparison(ti_data, field):
    """Compare folded-facies continuity along x, y, and z."""
    axis_names = ["x", "y", "z"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    for axis, axis_name in enumerate(axis_names):
        max_lag = min(VALIDATION_MAX_LAG, ti_data.shape[axis] - 1, field.shape[axis] - 1)
        lags = np.arange(max_lag + 1)
        color = ["#1d76a8", "#58a6c7", "#0b3f57"][axis]
        ax.plot(
            lags,
            two_point_correlation(ti_data, max_lag=max_lag, axis=axis),
            linestyle="--",
            color=color,
            linewidth=1.5,
            alpha=0.85,
        )
        ax.plot(
            lags,
            two_point_correlation(field, max_lag=max_lag, axis=axis),
            linestyle="-",
            color=color,
            linewidth=2.0,
            label=f"{axis_name}-axis",
        )
    ax.set_title("Two-point correlation for folded facies", fontsize=13)
    ax.set_xlabel("lag [cells]")
    ax.set_ylabel("S2(h) / S2(0)")
    ax.set_ylim(0, 1.05)
    ax.grid(color="0.88", linewidth=0.8)
    ax.legend(title="solid=simulation, dashed=TI", frameon=False, fontsize=10, ncol=3)
    ax.tick_params(labelsize=9)
    return fig


if generate_output:
    save_figure(plot_two_point_comparison(ti_data, field), output_paths["correlation"])
show_saved_figure(output_paths["correlation"], width=9.5)
