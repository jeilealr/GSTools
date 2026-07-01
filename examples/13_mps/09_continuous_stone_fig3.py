r"""
Continuous conditioning with a bundled texture
----------------------------------------------

This compact workflow combines hard conditioning with a continuous texture. It
uses the ``stone`` continuous training image from GAIA-UNIL, samples random
hard data from that texture, and compares the conditional simulation with the
TI through both maps and histograms.

.. note::

    This example loads the bundled derived texture
    ``input/gaia_unil_stone_texture.npz``. It was prepared from
    ``stone.tiff`` in the GAIA-UNIL training-image collection, distributed
    under **GPL-3.0**. Full source and redistribution notices are documented in
    ``input/LICENSE.txt``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

###############################################################################
# Load the prepared continuous training image. The bundled file contains the
# 200x200 upstream image used by the gallery example.

example_dir = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
)
data_path = example_dir / "input" / "gaia_unil_stone_texture.npz"

if not data_path.exists():
    raise FileNotFoundError(
        f"Missing bundled training image: {data_path}. "
        "Run this example from examples/13_mps or restore the input assets."
    )

with np.load(data_path) as data:
    ti_full = data["texture"].astype(float)

grid_size = ti_full.shape[0]
ti_data = ti_full

###############################################################################
# Build a continuous TI using the ``"l2"`` distance.

ti = gs.TrainingImage(ti_data, categorical=False, distance="l2", n_neighbors=24)

###############################################################################
# Create hard conditioning values by sampling from the TI distribution.

rng = np.random.default_rng(42)
n_cond = 35
cond_x = rng.uniform(0, grid_size, n_cond)
cond_y = rng.uniform(0, grid_size, n_cond)
rand_ti_x = rng.integers(0, grid_size, n_cond)
rand_ti_y = rng.integers(0, grid_size, n_cond)
cond_val = ti_data[rand_ti_x, rand_ti_y]

###############################################################################
# Set up the MPS model and conditioning data.

model = gs.MPSModel(ti, scan_fraction=0.2, threshold=0.03)

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
    np.unravel_index(
        np.argsort(spiral_t.ravel(), kind="stable"), (grid_size, grid_size)
    )
)

print(
    f"Simulating continuous field ({grid_size}x{grid_size}) "
    f"with {n_cond} conditioning points..."
)
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
hist_ti, bins = np.histogram(ti_data.flatten(), bins=30, density=True)
hist_sim, _ = np.histogram(field.flatten(), bins=bins, density=True)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

ax3.plot(bin_centers, hist_sim, "k-", label="Simulation")
ax3.plot(bin_centers, hist_ti, "k--", label="Training image")
ax3.set_title("c) Comparison of histograms")
ax3.legend()

fig.tight_layout()
