r"""
A real training image: the Strebelle channels
----------------------------------------------

The previous examples used tiny synthetic training images. Here we use the
<<<<<<< HEAD
classic **Strebelle (2002) channelized fluvial training image**, a widely used
reference example for MPS, and condition the simulation on random hard data.

.. note::

    **Data source / license.** The training image is downloaded from
    `GeoDataSets <https://github.com/GeostatsGuy/GeoDataSets>`_ by Michael
    Pyrcz (GeostatsGuy), distributed under the **MIT license**. The underlying
    channel TI is due to Strebelle, S. (2002), *Conditional simulation of
    complex geological structures using multiple-point statistics*,
    Mathematical Geology, 34(1), 1-21.
"""

import urllib.request
from pathlib import Path
=======
classic **Strebelle (2002) channelized fluvial training image**, the de-facto
benchmark for MPS, and condition the simulation on random hard data.

.. note::

    **Data source / license.** The training image is downloaded from the
    `GeoDataSets <https://github.com/GeostatsGuy/GeoDataSets>`_ repository by
    Michael Pyrcz (GeostatsGuy), which is distributed under the **MIT license**
    (redistribution permitted with attribution). The underlying channel TI is
    due to Strebelle, S. (2002), *Conditional simulation of complex geological
    structures using multiple-point statistics*, Mathematical Geology, 34(1),
    1-21. If the download is unavailable, the example falls back to a synthetic
    training image so it still runs offline.
"""

import os
import urllib.request
>>>>>>> n0228a/mps-direct-sampling

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import gstools as gs

###############################################################################
<<<<<<< HEAD
# Download the Strebelle TI once and keep it next to this example. The cached
# ``.npz`` file is ignored by Git.
=======
# Load the Strebelle training image, with a synthetic fallback for offline use.
>>>>>>> n0228a/mps-direct-sampling

TI_URL = (
    "https://raw.githubusercontent.com/GeostatsGuy/"
    "GeoDataSets/master/MPS_Training_image_and_Realizations_500.npz"
)
<<<<<<< HEAD
if "__file__" in globals():
    CACHE = Path(__file__).resolve().with_name("mps_strebelle.npz")
else:
    cwd = Path.cwd()
    CACHE = Path("mps_strebelle.npz")
    for candidate in (
        cwd / "mps_strebelle.npz",
        cwd / "examples" / "13_mps" / "mps_strebelle.npz",
        cwd.parent / "examples" / "13_mps" / "mps_strebelle.npz",
    ):
        if candidate.exists():
            CACHE = candidate
            break
if not CACHE.exists():
    urllib.request.urlretrieve(TI_URL, CACHE)

with np.load(CACHE) as data:
    ti_arr = data["array1"].astype(int)

ti = gs.TrainingImage(ti_arr, categorical=True)

###############################################################################
# Take unique conditioning points from the training image patterns. The grid is
# intentionally modest so the example remains suitable for the documentation
# gallery while still using the real 256x256 TI.

sg_size = 56
n_cond = 60
rng = np.random.default_rng(0)
cond_idx = rng.choice(sg_size * sg_size, size=n_cond, replace=False)
cond_x_idx, cond_y_idx = np.unravel_index(cond_idx, (sg_size, sg_size))
cond_x = cond_x_idx.astype(float)
cond_y = cond_y_idx.astype(float)
cond_val = ti_arr[cond_x_idx, cond_y_idx]
=======
CACHE = "mps_strebelle.npz"
try:
    if not os.path.exists(CACHE):
        urllib.request.urlretrieve(TI_URL, CACHE)
    ti_arr = np.load(CACHE)["array1"].astype(float)
    source = "Strebelle (2002) via GeoDataSets (MIT)"
except Exception as err:  # pragma: no cover - network fallback
    print(f"download failed ({err}); using a synthetic channel TI instead")
    gx, gy = np.meshgrid(np.arange(150), np.arange(150), indexing="ij")
    ti_arr = ((np.sin(gx / 6.0) + np.sin((gx + gy) / 10.0)) > 0).astype(float)
    source = "synthetic fallback"

ti = gs.TrainingImage(ti_arr, categorical=True, n_neighbors=30)
print(f"TI {ti.shape} ({source}), sand fraction = {ti_arr.mean():.3f}")

###############################################################################
# Take 80 random conditioning points from the training image patterns.

sg_size = 80
rng = np.random.default_rng(0)
cond_x = rng.integers(0, sg_size, 80).astype(float)
cond_y = rng.integers(0, sg_size, 80).astype(float)
cond_val = ti_arr[cond_x.astype(int), cond_y.astype(int)]
>>>>>>> n0228a/mps-direct-sampling

###############################################################################
# Simulate with DSBC-style parameters (best-candidate + partial scan).

<<<<<<< HEAD
ds = gs.DirectSampling(ti, n_neighbors=20, scan_fraction=0.08, threshold=0.0)
ds.set_condition([cond_x, cond_y], cond_val)
grid = [np.arange(sg_size, dtype=float), np.arange(sg_size, dtype=float)]
field = ds(grid, seed=42)

assert np.all(field[cond_x_idx, cond_y_idx] == cond_val)
=======
ds = gs.DirectSampling(
    gs.MPSModel(ti, scan_fraction=0.2, threshold=0.0)
)
ds.set_condition([cond_x, cond_y], cond_val)
field = ds([np.arange(sg_size, dtype=float)] * 2, seed=42)

honored = int(
    (field[cond_x.astype(int), cond_y.astype(int)] == cond_val).sum()
)
print(f"conditioning honoured: {honored}/{cond_val.size}")
>>>>>>> n0228a/mps-direct-sampling

###############################################################################
# Plot the training image crop next to the conditional realization.

cmap = ListedColormap(["#c9a96e", "#2b6cb0"])  # shale / sand
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5.5))
<<<<<<< HEAD
ax0.imshow(
    ti_arr[:sg_size, :sg_size],
    cmap=cmap,
    origin="lower",
    vmin=0,
    vmax=1,
)
ax0.set_title("Training image (crop)")
ax1.imshow(field, cmap=cmap, origin="lower", vmin=0, vmax=1)
ax1.scatter(
    cond_y,
    cond_x,
    c=cond_val,
    cmap=cmap,
    edgecolors="k",
    linewidths=0.5,
    s=18,
    vmin=0,
    vmax=1,
)
=======
ax0.imshow(ti_arr[:sg_size, :sg_size], cmap=cmap, origin="lower")
ax0.set_title("Training image (crop)")
ax1.imshow(field, cmap=cmap, origin="lower")
ax1.scatter(cond_y, cond_x, c=cond_val, cmap=cmap, edgecolors="k", s=18)
>>>>>>> n0228a/mps-direct-sampling
ax1.set_title("Conditional DS realization")
fig.tight_layout()
