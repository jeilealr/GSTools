r"""
3D categorical simulation with voxel and slice plots
----------------------------------------------------

Direct Sampling is not limited to 2D images. A training image can be any
structured array, including a 3D categorical volume. This example keeps the
volume small for the documentation build and visualizes the result with a 3D
voxel view plus orthogonal slice plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import gstools as gs

###############################################################################
# Load the generated 3D training image. The channel facies was generated once
# from two smooth surfaces so the TI contains connected structures in all three
# directions.

example_dir = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
)
data_path = example_dir / "input" / "synthetic_3d_channel_ti.npz"

if not data_path.exists():
    raise FileNotFoundError(
        f"Missing bundled training image: {data_path}. "
        "Run this example from examples/13_mps or restore the input assets."
    )

with np.load(data_path) as data:
    ti_data = data["ti"].astype(int)

###############################################################################
# Use a modest simulation grid. ``max_radius`` limits neighbor selection to a
# local 3D search radius, which keeps the example fast while still using true 3D
# patterns. ``boundary="partial"`` allows data events near volume boundaries to
# use the available in-domain neighbors instead of requiring a complete stencil.

ti = gs.TrainingImage(
    ti_data,
    categorical=True,
    n_neighbors=10,
    max_radius=5,
)
model = gs.MPSModel(ti, scan_fraction=0.06, threshold=0.0, boundary="partial")
ds = gs.DirectSampling(model)

grid_size = 16
grid = [np.arange(grid_size, dtype=float) for _ in range(3)]
field = ds(grid, seed=11)

###############################################################################
# First plot the channel facies as 3D voxels. Only facies ``1`` is rendered, so
# the connected geometry is visible without drawing the full background volume.


def plot_voxels(ax, volume, title):
    """Plot the active facies of a small 3D categorical volume."""
    active = volume.astype(bool)
    colors = np.empty(active.shape, dtype=object)
    colors[:] = "#00000000"
    colors[active] = "#2f6f9fcc"
    ax.voxels(active, facecolors=colors, edgecolor="#1a3f5c", linewidth=0.08)
    ax.set_title(title)
    ax.set_box_aspect(volume.shape)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=24, azim=-55)


fig = plt.figure(figsize=(9, 4), constrained_layout=True)
ax_ti_3d = fig.add_subplot(1, 2, 1, projection="3d")
ax_sim_3d = fig.add_subplot(1, 2, 2, projection="3d")
plot_voxels(ax_ti_3d, ti_data, "Training image, facies 1")
plot_voxels(ax_sim_3d, field, "Simulation, facies 1")

###############################################################################
# Orthogonal centre slices complement the voxel view by showing the interior
# patterns through the middle of each volume.


def center_slices(volume):
    """Return orthogonal centre slices from a 3D array."""
    cx, cy, cz = (size // 2 for size in volume.shape)
    return [
        (volume[cx, :, :].T, "x slice"),
        (volume[:, cy, :].T, "y slice"),
        (volume[:, :, cz].T, "z slice"),
    ]


cmap = ListedColormap(["#d8c690", "#2f6f9f"])
fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)

for ax, (slc, title) in zip(axes[0], center_slices(ti_data)):
    ax.imshow(
        slc,
        cmap=cmap,
        origin="lower",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax.set_title(f"Training image {title}")
    ax.set_xticks([])
    ax.set_yticks([])

for ax, (slc, title) in zip(axes[1], center_slices(field)):
    ax.imshow(
        slc,
        cmap=cmap,
        origin="lower",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax.set_title(f"Simulation {title}")
    ax.set_xticks([])
    ax.set_yticks([])
