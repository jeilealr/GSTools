r"""
Continuous conditioning with a bundled texture
---------------------------------------------

The last example combines the pieces from the previous pages in a compact
continuous workflow. A smooth texture is derived from the bundled Strebelle
asset, random hard data are sampled from that texture, and the conditional
simulation is compared with the TI through both maps and histograms.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

import gstools as gs

###############################################################################
# Prepare a continuous training image from the bundled Strebelle asset.

# Load the bundled Strebelle training image and smooth it into a continuous
# texture so the example stays self-contained.
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
    ti_data = data["array1"].astype(float)

ti_data = uniform_filter(ti_data, size=9, mode="reflect")
ti_data = (ti_data - ti_data.min()) / (ti_data.max() - ti_data.min())

# Use a 200x200 crop to keep the gallery runtime manageable.
ti_data = ti_data[:200, :200]

###############################################################################
# Build a continuous TI using the ``"l2"`` distance.

ti = gs.TrainingImage(ti_data, categorical=False, distance="l2", n_neighbors=80)

###############################################################################
# Create hard conditioning values by sampling from the TI distribution.

np.random.seed(42)
n_cond = 100
grid_size = 200
cond_x = np.random.uniform(0, grid_size, n_cond)
cond_y = np.random.uniform(0, grid_size, n_cond)
rand_ti_x = np.random.randint(0, grid_size, n_cond)
rand_ti_y = np.random.randint(0, grid_size, n_cond)
cond_val = ti_data[rand_ti_x, rand_ti_y]

###############################################################################
# Set up the MPS model and conditioning data.

model = gs.MPSModel(ti, scan_fraction=0.5, threshold=0.01)

ds = gs.DirectSampling(model)
ds.set_condition([cond_x, cond_y], cond_val)

x = y = np.arange(grid_size, dtype=float)

###############################################################################
# Use a spiral simulation path so the field grows outward from the centre.

center = grid_size / 2.0
rows_g, cols_g = np.indices((grid_size, grid_size))
r_grid = np.sqrt((rows_g - center) ** 2 + (cols_g - center) ** 2)
theta_grid = np.arctan2(rows_g - center, cols_g - center) % (2 * np.pi)
pitch = 3.0
spiral_t = r_grid + theta_grid * (pitch / (2 * np.pi))
spiral_path = np.column_stack(
    np.unravel_index(np.argsort(spiral_t.ravel(), kind="stable"), (grid_size, grid_size))
)

print(f"Simulating continuous field ({grid_size}x{grid_size}) with 100 cond points...")
field = ds([x, y], seed=123, num_threads=1, path=spiral_path)

###############################################################################
# Plot the TI, the conditional simulation, and their marginal distributions.

fig = plt.figure(figsize=(12, 10))
ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(212)

# a) TI
im_a = ax1.imshow(ti_data.T, cmap="gray", origin="lower", vmin=0, vmax=1.0)
ax1.set_title("a) Training image")

# b) Simulation
im_b = ax2.imshow(field.T, cmap="gray", origin="lower", vmin=0, vmax=1.0)
# Overlay conditioning points (circles whose color indicates the value)
ax2.scatter(
    cond_x,
    cond_y,
    c=cond_val,
    cmap="gray",
    vmin=0,
    vmax=1.0,
    edgecolors="k",
    s=40,
    linewidths=0.5,
)
ax2.set_title("b) Simulation")
plt.colorbar(im_b, ax=ax2, fraction=0.046, pad=0.04)

# c) Histogram comparison
hist_ti, bins = np.histogram(ti_data.flatten(), bins=50, density=True)
hist_sim, _ = np.histogram(field.flatten(), bins=bins, density=True)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

ax3.plot(bin_centers, hist_sim, "k-", label="Simulation")
ax3.plot(bin_centers, hist_ti, "k--", label="Training image")
ax3.set_title("c) Comparison of histograms")
ax3.legend()

fig.tight_layout()
