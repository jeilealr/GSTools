"""
Bivariate joint simulation from a channel TI
--------------------------------------------

Example ``06`` used a secondary variable as exhaustive conditioning data. Here
both variables are simulated jointly: a categorical facies variable from the
Strebelle image and a continuous secondary variable derived from a smoothed
version of the same image.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

import gstools as gs

###############################################################################
# Prepare the two training variables.

# Load the bundled Strebelle training image (Variable 1).
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
    ti_full = data["array1"].astype(float)
# Use the largest square crop that fits the bundled 256x256 asset while keeping
# the example runtime predictable.
ti_var1 = ti_full[:250, :250]

# Generate a continuous secondary variable by smoothing the facies image and
# adding small uncorrelated noise.
smoothed = uniform_filter(ti_var1, size=23, mode="reflect")

np.random.seed(42)
noise = np.random.uniform(0, 0.5, size=ti_var1.shape)
ti_var2 = smoothed + noise

###############################################################################
# Assemble the multivariate TI. The facies variable is categorical; the
# secondary variable is continuous and uses the ``"l2"`` distance.

ti = gs.TrainingImage([
    gs.Variable("facies",      ti_var1, categorical=True,  weight=0.5, distance="l1", n_neighbors=30),
    gs.Variable("resistivity", ti_var2, categorical=False, weight=0.5, distance="l2", n_neighbors=30),
])

###############################################################################
# Run an unconditional joint simulation of both variables.

model = gs.MPSModel(ti, scan_fraction=0.5, threshold=0.01)

ds = gs.DirectSampling(model)

grid_size = 250
x = y = np.arange(grid_size, dtype=float)

print(f"Simulating unconditional bivariate field ({grid_size}x{grid_size})...")
fields = ds([x, y], seed=123, num_threads=1)

sim_var1 = fields["facies"]
sim_var2 = fields["resistivity"]

###############################################################################
# Plot each TI variable next to its simulated counterpart.

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# a) TI var 1
axes[0, 0].imshow(ti_var1, cmap="gray", origin="lower")
axes[0, 0].set_title("a) Training image, variable 1")

# b) TI var 2
im_b = axes[0, 1].imshow(ti_var2, cmap="gray", origin="lower", vmin=0, vmax=1.5)
axes[0, 1].set_title("b) Training image, variable 2")
plt.colorbar(im_b, ax=axes[0, 1], fraction=0.046, pad=0.04)

# c) Sim var 1
axes[1, 0].imshow(sim_var1, cmap="gray", origin="lower")
axes[1, 0].set_title("c) Simulation, variable 1")

# d) Sim var 2
im_d = axes[1, 1].imshow(sim_var2, cmap="gray", origin="lower", vmin=0, vmax=1.5)
axes[1, 1].set_title("d) Simulation, variable 2")
plt.colorbar(im_d, ax=axes[1, 1], fraction=0.046, pad=0.04)

fig.tight_layout()
