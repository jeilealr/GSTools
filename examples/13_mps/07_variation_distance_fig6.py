r"""
Variation distance for continuous conditioning
----------------------------------------------

The continuous distance page compared metrics without conditioning. Here we
return to conditioning and focus on the ``"variation"`` distance: the TI has
local fluctuations around zero, while the conditioning data include a
large-scale trend. Comparing local variation rather than absolute level makes
that mismatch meaningful instead of less suitable for matching.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

###############################################################################
# Load the generated continuous TI. It was generated once as a multi-Gaussian
# field with anisotropic spatial correlation, using an exponential covariance
# model with ``len_scale=[35, 25]`` and ``seed=123``.

example_dir = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
)
data_path = example_dir / "input" / "synthetic_variation_ti.npz"

if not data_path.exists():
    raise FileNotFoundError(
        f"Missing bundled training image: {data_path}. "
        "Run this example from examples/13_mps or restore the input assets."
    )

with np.load(data_path) as data:
    ti_data = data["ti"]

grid_size = ti_data.shape[0]
x = y = np.arange(grid_size, dtype=float)

###############################################################################
# Use the variation distance for continuous data-event comparisons.

ti = gs.TrainingImage(
    ti_data,
    categorical=False,
    distance="variation",
    n_neighbors=12,
)

###############################################################################
# Create conditioning data with a spatial trend and a different absolute level
# than the TI.

rng = np.random.default_rng(42)
n_cond = 50
cond_x = rng.uniform(0, grid_size, n_cond)
cond_y = rng.uniform(0, grid_size, n_cond)
cond_val = 100 + 10 * (cond_x / grid_size) + rng.normal(0, 1.5, n_cond)

###############################################################################
# Simulate with the continuous variation distance.

model = gs.MPSModel(ti, scan_fraction=0.2, threshold=0.03)

ds = gs.DirectSampling(model)
ds.set_condition([cond_x, cond_y], cond_val)

print(
    f"Simulating variation-based field ({grid_size}x{grid_size}) "
    f"with {n_cond} conditioning points..."
)
field = ds([x, y], seed=42, num_threads=1)

###############################################################################
# Plot the TI, the conditioning data, and one conditional realization.

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# a) TI
im_a = axes[0].imshow(
    ti_data.T,
    cmap="gray",
    origin="lower",
    extent=[0, grid_size, 0, grid_size],
    vmin=-3,
    vmax=3,
)
axes[0].set_title("a) Training image")
plt.colorbar(im_a, ax=axes[0], fraction=0.046, pad=0.04)

# b) Conditioning data
im_b = axes[1].scatter(
    cond_x, cond_y, c=cond_val, cmap="gray", vmin=98, vmax=112, s=20
)
axes[1].set_xlim(0, grid_size)
axes[1].set_ylim(0, grid_size)
axes[1].set_aspect("equal")
axes[1].set_title("b) Conditioning data")
plt.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.04)

# c) Simulation
im_c = axes[2].imshow(
    field.T,
    cmap="gray",
    origin="lower",
    extent=[0, grid_size, 0, grid_size],
    vmin=98,
    vmax=112,
)
# Overlay conditioning points as open circles
axes[2].scatter(
    cond_x, cond_y, facecolors="none", edgecolors="k", s=40, linewidths=1
)
axes[2].set_title("c) One simulation")
plt.colorbar(im_c, ax=axes[2], fraction=0.046, pad=0.04)

fig.tight_layout()
