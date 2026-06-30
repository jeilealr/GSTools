"""
Multivariate channels with a context variable
---------------------------------------------

This example introduces multivariate MPS after the geometric nonstationarity
examples. A synthetic TI contains a categorical channel variable and a
continuous context variable. The context variable is then provided everywhere on
the simulation grid, so Direct Sampling can use it as an exhaustive secondary
variable while simulating the channels.
"""

import numpy as np
import matplotlib.pyplot as plt
import gstools as gs

###############################################################################
# Create the synthetic multivariate TI.

ti_size = 250
gx_ti, gy_ti = np.meshgrid(np.arange(ti_size), np.arange(ti_size), indexing="ij")

# The channel orientation rotates as a function of x.
theta_ti = (gx_ti / float(ti_size - 1)) * (np.pi / 2.0)
channels_ti = ((np.sin(gx_ti * np.cos(theta_ti) * 0.5 + gy_ti * np.sin(theta_ti) * 0.5)) > 0).astype(float)

# The secondary variable records the normalized x-coordinate in the TI.
context_ti = gx_ti / float(ti_size - 1)

###############################################################################
# Assemble a multivariate TI from named :any:`Variable` objects.

ti = gs.TrainingImage([
    gs.Variable("channels", channels_ti, categorical=True,  weight=0.5, n_neighbors=30),
    gs.Variable("context",  context_ti,  categorical=False, weight=0.5, distance="l1", n_neighbors=1),
])

###############################################################################
# Use a tight threshold because the context field is synthetic and exactly
# correlated with channel orientation.

model = gs.MPSModel(ti, scan_fraction=0.5, threshold=0.01)

###############################################################################
# On the simulation grid, provide the context variable at every node. The
# channel variable is left unknown by filling it with NaN.

sg_size = 250
xs_sg = np.arange(sg_size, dtype=float)
ys_sg = np.arange(sg_size, dtype=float)
gx_sg, gy_sg = np.meshgrid(xs_sg, ys_sg, indexing="ij")
context_sg = gy_sg / float(sg_size - 1)

cond_pos = [gx_sg.flatten(), gy_sg.flatten()]
cond_val = {
    "channels": np.full(sg_size * sg_size, np.nan),
    "context": context_sg.flatten()
}

ds = gs.DirectSampling(model)
ds.set_condition(cond_pos, cond_val)

###############################################################################
# Run the simulation and plot the TI variables beside the simulated variables.

print(f"Simulating multivariate field ({sg_size}x{sg_size})...")
fields = ds([xs_sg, ys_sg], seed=42, num_threads=1)
channels_sim = fields["channels"]
context_sim = fields["context"]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(channels_ti, cmap="gray", origin="lower")
axes[0, 0].set_title("a) Training image, variable 1")

axes[0, 1].imshow(context_ti, cmap="gray", origin="lower", vmin=0, vmax=1)
axes[0, 1].set_title("b) Training image, variable 2")

axes[1, 0].imshow(channels_sim, cmap="gray", origin="lower")
axes[1, 0].set_title("c) Simulation, variable 1")

axes[1, 1].imshow(context_sim, cmap="gray", origin="lower", vmin=0, vmax=1)
axes[1, 1].set_title("d) Simulation, variable 2")

fig.tight_layout()
