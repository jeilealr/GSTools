"""
Variation distance for continuous conditioning
----------------------------------------------

Example ``02`` compared continuous distance metrics without conditioning. This
example returns to conditioning and focuses on the ``"variation"`` distance:
the TI has local fluctuations around zero, while the conditioning data include
a large-scale trend. Comparing local variation rather than absolute level makes
that mismatch meaningful instead of fatal.
"""

import numpy as np
import matplotlib.pyplot as plt
import gstools as gs

###############################################################################
# Create a multi-Gaussian continuous training image with anisotropic spatial
# correlation.

grid_size = 250
x = y = np.arange(grid_size, dtype=float)
cov_model = gs.Exponential(dim=2, var=1.0, len_scale=[35, 25], angles=0)
srf = gs.SRF(cov_model, mean=0, seed=123)
ti_data = srf((x, y), mesh_type="structured")

###############################################################################
# Use the variation distance for continuous data-event comparisons.

ti = gs.TrainingImage(ti_data, categorical=False, distance="variation", n_neighbors=15)

###############################################################################
# Create conditioning data with a spatial trend and a different absolute level
# than the TI.

np.random.seed(42)
n_cond = 100
cond_x = np.random.uniform(0, grid_size, n_cond)
cond_y = np.random.uniform(0, grid_size, n_cond)
cond_val = 100 + 10 * (cond_x / grid_size) + np.random.normal(0, 1.5, n_cond)

###############################################################################
# Simulate with the continuous variation distance.

model = gs.MPSModel(ti, scan_fraction=0.5, threshold=0.01)

ds = gs.DirectSampling(model)
ds.set_condition([cond_x, cond_y], cond_val)

print(f"Simulating variation-based field ({grid_size}x{grid_size}) with 100 cond points...")
field = ds([x, y], seed=42, num_threads=1)

###############################################################################
# Plot the TI, the conditioning data, and one conditional realization.

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# a) TI
im_a = axes[0].imshow(ti_data.T, cmap="gray", origin="lower", extent=[0, 250, 0, 250], vmin=-3, vmax=3)
axes[0].set_title("a) Training image")
plt.colorbar(im_a, ax=axes[0], fraction=0.046, pad=0.04)

# b) Conditioning data
im_b = axes[1].scatter(cond_x, cond_y, c=cond_val, cmap="gray", vmin=98, vmax=112, s=20)
axes[1].set_xlim(0, 250)
axes[1].set_ylim(0, 250)
axes[1].set_aspect("equal")
axes[1].set_title("b) Conditioning data")
plt.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.04)

# c) Simulation
im_c = axes[2].imshow(field.T, cmap="gray", origin="lower", extent=[0, 250, 0, 250], vmin=98, vmax=112)
# Overlay conditioning points as open circles
axes[2].scatter(cond_x, cond_y, facecolors="none", edgecolors="k", s=40, linewidths=1)
axes[2].set_title("c) One simulation")
plt.colorbar(im_c, ax=axes[2], fraction=0.046, pad=0.04)

fig.tight_layout()
