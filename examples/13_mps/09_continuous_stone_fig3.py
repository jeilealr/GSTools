import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import weibull
from PIL import Image

import gstools as gs

# 1. Prepare Training Image
# Load the "stone" training image from the provided URL
TI_URL = "https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff"
CACHE = "stone.tiff"
if not os.path.exists(CACHE):
    urllib.request.urlretrieve(TI_URL, CACHE)

# Read the TIFF image
ti_img = Image.open(CACHE)
ti_data = np.array(ti_img).astype(float)
# The paper figure uses a 200x200 grid
ti_data = ti_data[:200, :200]

# Continuous variable using Distance 4 (Mariethoz Eq. 4, which is "l2" in GSTools)
ti = gs.TrainingImage(
    ti_data, categorical=False, distance="l4", n_neighbors=75
)

# 2. Setup Conditioning Data
np.random.seed(3)
n_cond = 10
grid_size = 200

# Paper: "Conditioning data are 100 values taken in the TI and located at random
# positions in the simulation."
cond_x = np.random.uniform(0, grid_size, n_cond)
cond_y = np.random.uniform(0, grid_size, n_cond)

# Randomly sample 100 values from the TI's marginal distribution
rand_ti_x = np.random.randint(0, grid_size, n_cond)
rand_ti_y = np.random.randint(0, grid_size, n_cond)
cond_val = ti_data[rand_ti_x, rand_ti_y]

# 3. Setup MPS Model
# From caption: n = 80, t = 0.01. We use scan_fraction=0.5
model = gs.MPSModel(ti, scan_fraction=0.4, threshold=0.01, cond_weight=2)

# 4. Run Simulation
ds = gs.DirectSampling(model)
ds.set_condition([cond_x, cond_y], cond_val)

x = y = np.arange(grid_size, dtype=float)

# Spiral outward simulation path. The engine silently drops any conditioned
# nodes present in an explicit path, so we can pass the full grid in spiral
# order without pre-computing which nodes are pre-filled.
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
    f"Simulating continuous field ({grid_size}x{grid_size}) with 100 cond points..."
)
field = ds([x, y], seed=1, num_threads=4, path="sequential")

# 5. Plotting
# Use a custom GridSpec to match the layout of Figure 3 (two images on top, histogram on bottom)
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
scat = ax2.scatter(
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
plt.savefig("mariethoz_fig3_reproduction.png")
print("Saved mariethoz_fig3_reproduction.png")
