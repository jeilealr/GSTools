r"""
Radial nonstationarity with a spiral path
-----------------------------------------

This example follows the geometric nonstationarity idea from Mariethoz et al.
(2010), Figure 7. The Strebelle channel TI is simulated with a radial rotation
map and a distance-based anisotropy map, producing channels that radiate away
from the centre and become thinner toward the edges.

In addition to the nonstationary maps, we supply an explicit outward
**spiral simulation path**: nodes are visited in Archimedean spiral order from
the centre outward. This lets each new node use already simulated inner
neighbours, reinforcing coherent channel propagation toward the edges.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import gstools as gs

###############################################################################
# Load the bundled Strebelle training image.
if "__file__" in globals():
    data_path = Path(__file__).resolve().with_name("mps_strebelle.npz")
else:
    data_path = Path("mps_strebelle.npz")

if not data_path.exists():
    raise FileNotFoundError(
        f"Missing bundled training image: {data_path}. "
        "Run this example from examples/13_mps or restore mps_strebelle.npz."
    )

with np.load(data_path) as data:
    ti_arr = data["array1"].astype(float)
source = "Strebelle (2002) via bundled GeoDataSets asset"

ti = gs.TrainingImage(ti_arr, categorical=True, n_neighbors=30)
print(f"TI {ti.shape} ({source}), sand fraction = {ti_arr.mean():.3f}")

###############################################################################
# The paper used a larger simulation grid. This gallery version keeps the grid
# smaller so the documentation build remains practical.

sg_size = 250
xs = np.arange(sg_size, dtype=float)
ys = np.arange(sg_size, dtype=float)
gx, gy = np.meshgrid(xs, ys, indexing="ij")

center_x = sg_size / 2.0
center_y = sg_size / 2.0

# Rotation is the angle from the centre, creating a radial pattern. GSTools uses
# radians for rotation angles.
rotation = np.arctan2(gx - center_x, gy - center_y)

# The anisotropy map scales the search geometry based on distance from the
# centre. Values decrease toward the corners, making the channels thinner.
radius = np.sqrt((gx - center_x) ** 2 + (gy - center_y) ** 2)
max_radius = np.sqrt(center_x**2 + center_y**2)
anis = 1.0 - 0.8 * (radius / max_radius)

###############################################################################
# Build a spiral simulation path that visits nodes outward from the centre.

theta_grid = np.arctan2(gx - center_x, gy - center_y) % (2 * np.pi)
pitch = 3.0  # radial gap between successive spiral arms (pixels)
spiral_t = radius + theta_grid * (pitch / (2 * np.pi))
spiral_order = np.argsort(spiral_t.ravel(), kind="stable")
spiral_path = np.column_stack(
    np.unravel_index(spiral_order, (sg_size, sg_size))
)

###############################################################################
# Configure the MPS model. A positive threshold allows approximate matches,
# which is useful because exact matches under continuous rotation and scaling
# are rare.

ds = gs.DirectSampling(
    gs.MPSModel(ti, scan_fraction=0.25, threshold=0.05)
)
ds.set_nonstationary(rotation=rotation, anis=anis)

print(f"Simulating nonstationary field ({sg_size}x{sg_size}) with spiral path...")
field = ds([xs, ys], seed=53, num_threads=1, path=spiral_path)

###############################################################################
# Plot the maps, the TI crop, and one realization.

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# a) Rotation map
im_rot = axes[0, 0].imshow(np.rad2deg(rotation), cmap="gray", origin="lower")
axes[0, 0].set_title("a) Rotation (degrees)")
plt.colorbar(im_rot, ax=axes[0, 0], fraction=0.046, pad=0.04)

# b) Affinity map
im_aff = axes[0, 1].imshow(
    anis, cmap="gray", origin="lower", vmin=0.1, vmax=1.0
)
axes[0, 1].set_title("b) Affinity ratio")
plt.colorbar(im_aff, ax=axes[0, 1], fraction=0.046, pad=0.04)

# c) TI
cmap = ListedColormap(
    ["#000000", "#FFFFFF"]
)  # black background, white channels
axes[1, 0].imshow(ti_arr[:250, :250], cmap=cmap, origin="lower")
axes[1, 0].set_title("c) TI")

# d) Simulation
axes[1, 1].imshow(field, cmap=cmap, origin="lower")
axes[1, 1].set_title("d) One simulation")

fig.tight_layout()
