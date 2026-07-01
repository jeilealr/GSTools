r"""
Bivariate joint simulation from a channel TI
--------------------------------------------

The context-variable page used a secondary variable as exhaustive conditioning
data. Here both variables are simulated jointly: a categorical facies variable
from the Strebelle image and a continuous secondary variable derived from a
smoothed version of the same image.

.. note::

    This example uses the bundled derived input
    ``input/strebelle_facies_resistivity_ti.npz``. It contains the 250x250 Strebelle
    subset used in the paper-style setup, plus a smoothed/noisy secondary
    variable. The source data and license are documented in
    :ref:`sphx_glr_examples_13_mps_03_channel_strebelle.py` and
    ``input/LICENSE.txt``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

###############################################################################
# Prepare the two training variables.

# Load the prepared bivariate TI. The gallery uses a 190x190 crop from this
# 250x250 input to keep a buffer below the runtime budget while preserving most
# of the paper-style setup.
example_dir = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
)
data_path = example_dir / "input" / "strebelle_facies_resistivity_ti.npz"

if not data_path.exists():
    raise FileNotFoundError(
        f"Missing bundled training image: {data_path}. "
        "Run this example from examples/13_mps or restore the input assets."
    )

with np.load(data_path) as data:
    ti_var1_full = data["facies"].astype(int)
    ti_var2_full = data["resistivity"].astype(float)

grid_size = 190
ti_var1 = ti_var1_full[:grid_size, :grid_size]
ti_var2 = ti_var2_full[:grid_size, :grid_size]

###############################################################################
# Assemble the multivariate TI. The facies variable is categorical; the
# secondary variable is continuous and uses the ``"l2"`` distance.

ti = gs.TrainingImage(
    [
        gs.Variable(
            "facies",
            ti_var1,
            categorical=True,
            weight=0.5,
            distance="l1",
            n_neighbors=18,
        ),
        gs.Variable(
            "resistivity",
            ti_var2,
            categorical=False,
            weight=0.5,
            distance="l2",
            n_neighbors=18,
        ),
    ]
)

###############################################################################
# Run an unconditional joint simulation of both variables.

model = gs.MPSModel(ti, scan_fraction=0.2, threshold=0.02)

ds = gs.DirectSampling(model)

x = y = np.arange(grid_size, dtype=float)

print(f"Simulating unconditional bivariate field ({grid_size}x{grid_size})...")
fields = ds([x, y], seed=123, num_threads=1)

sim_var1 = fields["facies"]
sim_var2 = fields["resistivity"]

###############################################################################
# Plot each TI variable next to its simulated counterpart.

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# a) Channel facies TI
axes[0, 0].imshow(ti_var1, cmap="gray", origin="lower")
axes[0, 0].set_title("a) Channel facies TI")

# b) Resistivity TI
im_b = axes[0, 1].imshow(ti_var2, cmap="gray", origin="lower", vmin=0, vmax=1.5)
axes[0, 1].set_title("b) Resistivity TI")
plt.colorbar(im_b, ax=axes[0, 1], fraction=0.046, pad=0.04)

# c) Simulated channel facies
axes[1, 0].imshow(sim_var1, cmap="gray", origin="lower")
axes[1, 0].set_title("c) Simulated channel facies")

# d) Simulated resistivity
im_d = axes[1, 1].imshow(sim_var2, cmap="gray", origin="lower", vmin=0, vmax=1.5)
axes[1, 1].set_title("d) Simulated resistivity")
plt.colorbar(im_d, ax=axes[1, 1], fraction=0.046, pad=0.04)

fig.tight_layout()
