r"""
Nonstationary geometry with the Strebelle channels
--------------------------------------------------

The previous example used the Strebelle training image with stationary search
geometry. Here the same TI is simulated with spatially varying rotation and
anisotropy maps, so the local data-event geometry changes across the simulation
grid while the training image itself stays fixed.
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
# Define a modest simulation grid and two smooth nonstationary maps.

sg_size = 80
xs = np.arange(sg_size, dtype=float)
ys = np.arange(sg_size, dtype=float)
gx, gy = np.meshgrid(xs, ys, indexing="ij")

# Rotation varies smoothly along the y-axis from 0 to 45 degrees.
rotation = (gy / sg_size) * (np.pi / 4.0)

# Anisotropy varies along the x-axis. Values above one stretch the local search
# geometry; values below one compress it.
anis = 0.8 + (gx / sg_size) * 0.4

###############################################################################
# Simulate with the nonstationary maps attached to the generator.

ds = gs.DirectSampling(
    gs.MPSModel(ti, scan_fraction=0.1, threshold=0.0)
)
ds.set_nonstationary(rotation=rotation, anis=anis)

print("Simulating nonstationary field...")
field = ds([xs, ys], seed=42)

###############################################################################
# Plot the original TI crop next to the nonstationary realization.

cmap = ListedColormap(["#c9a96e", "#2b6cb0"])  # shale / sand
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5.5))
ax0.imshow(ti_arr[:sg_size, :sg_size], cmap=cmap, origin="lower")
ax0.set_title("Training image (crop)")
ax1.imshow(field, cmap=cmap, origin="lower")
ax1.set_title("Nonstationary DS realization\n(varying rotation and anisotropy)")
fig.tight_layout()
