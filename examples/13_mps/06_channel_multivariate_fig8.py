r"""
Multivariate channels with a context variable
---------------------------------------------

This page introduces multivariate MPS with a categorical channel variable and a
continuous context variable. The context field is provided everywhere on the
simulation grid, so Direct Sampling can use it as exhaustive secondary data
while simulating the channels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

###############################################################################
# Load the generated multivariate TI. It was generated once from a channel
# pattern whose orientation rotates with x and a context variable equal to the
# normalized x-coordinate. Keeping it as a local input avoids regenerating TI
# data during gallery builds.
example_dir = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
)
data_path = example_dir / "input" / "synthetic_channel_context_ti.npz"

if not data_path.exists():
    raise FileNotFoundError(
        f"Missing bundled training image: {data_path}. "
        "Run this example from examples/13_mps or restore the input assets."
    )

with np.load(data_path) as data:
    channels_ti = data["channels"]
    context_ti = data["context"]

###############################################################################
# Assemble a multivariate TI from named :any:`Variable` objects.

ti = gs.TrainingImage(
    [
        gs.Variable(
            "channels",
            channels_ti,
            categorical=True,
            weight=0.5,
            n_neighbors=18,
        ),
        gs.Variable(
            "context",
            context_ti,
            categorical=False,
            weight=0.5,
            distance="l1",
            n_neighbors=1,
        ),
    ]
)

###############################################################################
# Use a tight threshold because the context field is synthetic and exactly
# correlated with channel orientation.

model = gs.MPSModel(ti, scan_fraction=0.2, threshold=0.01)

###############################################################################
# On the simulation grid, provide the context variable at every node. The
# channel variable is left unknown by filling it with NaN.

sg_size = 120
xs_sg = np.arange(sg_size, dtype=float)
ys_sg = np.arange(sg_size, dtype=float)
gx_sg, gy_sg = np.meshgrid(xs_sg, ys_sg, indexing="ij")
context_sg = gx_sg / float(sg_size - 1)

cond_pos = [gx_sg.flatten(), gy_sg.flatten()]
cond_val = {
    "channels": np.full(sg_size * sg_size, np.nan),
    "context": context_sg.flatten(),
}

ds = gs.DirectSampling(model)
ds.set_condition(cond_pos, cond_val)

###############################################################################
# Run the simulation and plot the TI variables beside the simulated channels and
# exhaustive context field.

print(f"Simulating multivariate field ({sg_size}x{sg_size})...")
fields = ds([xs_sg, ys_sg], seed=42, num_threads=1)
channels_sim = fields["channels"]
context_sim = fields["context"]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(channels_ti, cmap="gray", origin="lower")
axes[0, 0].set_title("a) Channel facies TI")

axes[0, 1].imshow(context_ti, cmap="gray", origin="lower", vmin=0, vmax=1)
axes[0, 1].set_title("b) Context field TI")

axes[1, 0].imshow(channels_sim, cmap="gray", origin="lower")
axes[1, 0].set_title("c) Simulated channel facies")

axes[1, 1].imshow(context_sim, cmap="gray", origin="lower", vmin=0, vmax=1)
axes[1, 1].set_title("d) Exhaustive context data")

fig.tight_layout()
