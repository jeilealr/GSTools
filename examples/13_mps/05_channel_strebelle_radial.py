r"""
Radial nonstationarity
----------------------

This example follows the geometric nonstationarity idea from Mariethoz et al.
(2010), Figure 7. The Strebelle channel TI is simulated with a radial rotation
map and a distance-based anisotropy map. These maps reorient and rescale local
data-event matching so the channel patterns tend to follow the radial geometry.
The paper uses a much larger simulation grid; this gallery example keeps the
grid smaller so it can run as part of the documentation build.

.. note::

    This example reuses the bundled Strebelle TI stored in
    ``input/strebelle_channel_ti.npz``. The data source and license are documented in
    :ref:`sphx_glr_examples_13_mps_03_channel_strebelle.py` and
    ``input/LICENSE.txt``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import gstools as gs

###############################################################################
# Load the bundled Strebelle training image.
example_dir = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
)
data_path = example_dir / "input" / "strebelle_channel_ti.npz"

if not data_path.exists():
    raise FileNotFoundError(
        f"Missing bundled training image: {data_path}. "
        "Run this example from examples/13_mps or restore the input directory."
    )

with np.load(data_path) as data:
    ti_arr = data["array1"].astype(int)
source = "Strebelle (2002) via bundled GeoDataSets asset"

ti = gs.TrainingImage(ti_arr, categorical=True, n_neighbors=24)
print(f"TI {ti.shape} ({source}), sand fraction = {ti_arr.mean():.3f}")

###############################################################################
# The paper used a 1000x1000 simulation grid. This gallery version keeps the
# grid smaller so the documentation build remains practical. Increase
# ``sg_size`` for a higher-resolution figure outside the gallery build.

sg_size = 200
xs = np.arange(sg_size, dtype=float)
ys = np.arange(sg_size, dtype=float)
gx, gy = np.meshgrid(xs, ys, indexing="ij")

center_x = sg_size / 2.0
center_y = sg_size / 2.0

# Rotation is the angle from the centre, creating a radial pattern. GSTools uses
# radians for rotation angles.
rotation = np.arctan2(gx - center_x, gy - center_y)

# The anisotropy map scales the search geometry based on distance from the
# centre. The paper's affinity ratio ranges from 1.0 in the centre to about 0.4
# near the corners.
radius = np.sqrt((gx - center_x) ** 2 + (gy - center_y) ** 2)
max_radius = np.sqrt(center_x**2 + center_y**2)
anis = 1.0 - 0.6 * (radius / max_radius)

###############################################################################
# Configure the MPS model. ``threshold=0.0`` uses the DSBC setting: scan the
# requested fraction of the search window and take the best candidate found.

ds = gs.DirectSampling(
    gs.MPSModel(ti, scan_fraction=0.08, threshold=0.0)
)
ds.set_nonstationary(rotation=rotation, anis=anis)

print(f"Simulating radial nonstationary field ({sg_size}x{sg_size})...")
field = ds([xs, ys], seed=53, num_threads=1)

###############################################################################
# Plot the maps, the TI crop, and one realization in an aligned 2x2 layout.
# The colorbars use their own narrow columns so the image panels keep the same
# size in both rows.

fig = plt.figure(figsize=(9.5, 8), constrained_layout=True)
grid_spec = fig.add_gridspec(
    2,
    4,
    width_ratios=[1.0, 0.04, 1.0, 0.04],
    hspace=0.22,
    wspace=0.08,
)
ax_rot = fig.add_subplot(grid_spec[0, 0])
cax_rot = fig.add_subplot(grid_spec[0, 1])
ax_aff = fig.add_subplot(grid_spec[0, 2])
cax_aff = fig.add_subplot(grid_spec[0, 3])
ax_ti = fig.add_subplot(grid_spec[1, 0])
ax_sim = fig.add_subplot(grid_spec[1, 2])

# a) Rotation map
im_rot = ax_rot.imshow(np.rad2deg(rotation), cmap="gray", origin="lower")
ax_rot.set_title("a) Rotation (degrees)")
fig.colorbar(im_rot, cax=cax_rot)

# b) Affinity map
im_aff = ax_aff.imshow(
    anis, cmap="gray", origin="lower", vmin=0.4, vmax=1.0
)
ax_aff.set_title("b) Affinity ratio")
fig.colorbar(im_aff, cax=cax_aff)

# c) Channel facies TI
cmap = ListedColormap(
    ["#000000", "#FFFFFF"]
)  # black background, white channels
ax_ti.imshow(
    ti_arr[:sg_size, :sg_size],
    cmap=cmap,
    origin="lower",
    interpolation="nearest",
    vmin=0,
    vmax=1,
)
ax_ti.set_title("c) Channel facies TI")

# d) Simulation
ax_sim.imshow(
    field,
    cmap=cmap,
    origin="lower",
    interpolation="nearest",
    vmin=0,
    vmax=1,
)
ax_sim.set_title("d) Radial DS realization")
