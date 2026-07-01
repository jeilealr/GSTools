Multiple Point Statistics
=========================

Two-point geostatistics describes spatial structure with pairs of points:
covariance models, variograms, kriging, and spatial random fields. That is
often enough, but it is not a natural language for connected or curvilinear
patterns such as channels, fractures, lenses, or object-like facies bodies.

**Multiple Point Statistics (MPS)** uses a **training image (TI)** instead: a
2D or 3D structured array containing patterns considered representative of the
field to simulate. The TI may be categorical, such as integer facies codes, or
continuous, such as permeability, porosity, elevation, or another property.

Use MPS when the geometry of the pattern matters more than a variogram can
express. If a covariance model already describes the field well, :any:`SRF`
is usually simpler and faster; for categorical fields generated from
continuous Gaussian fields, :any:`PGS` may also be a better first choice.

Direct Sampling in GSTools
--------------------------

GSTools implements the **Direct Sampling (DS)** algorithm
(`Mariethoz et al., 2010 <https://doi.org/10.1029/2008WR007621>`_) with the
**Direct Sampling Best Candidate (DSBC)** parametrization recommended by
`Juda et al. (2022) <https://doi.org/10.1016/j.acags.2022.100091>`_.
The API follows the usual GSTools model/generator pattern:

* :any:`TrainingImage` stores one TI, or a list of named :any:`Variable`
  objects for multivariate simulation.
* :any:`MPSModel` stores the Direct Sampling parameters.
* :any:`DirectSampling` generates a realization on a structured grid.

At each unsimulated grid node, DS gathers the already known neighboring values
around that node. This neighborhood is the *data event*. DS then searches the
TI for candidate neighborhoods with the same relative cell positions and
similar values; when a good candidate is found, the value at the candidate
center is copied into the simulation grid. In an unconditional simulation,
these known neighbors are values simulated earlier in the random path; hard
data simply add measured values that must be honored.

Three parameters control the quality and runtime tradeoff:

* ``n_neighbors`` (``n``): the maximum number of known neighbors in each data
  event. Larger values preserve richer patterns but cost more.
* ``scan_fraction`` (``f``): the fraction of the TI search examined for each
  simulated node, subject to the valid candidate positions for the current data
  event. Larger values search harder but run slower.
* ``threshold`` (``t``): an early-acceptance distance. Start with
  ``threshold=0.0`` (DSBC), which scans the requested fraction and takes the
  best candidate found; exact matches are still accepted immediately. Tune
  positive thresholds only when this is not sufficient.

The gallery is ordered as a learning path. Start with **A first Direct Sampling
simulation**, then add field measurements in **Conditioning to hard data**, and
switch from facies labels to continuous values in **Continuous variables and
distance metrics**.

The next pages move from small synthetic images to the bundled Strebelle
channel TI stored in ``input/strebelle_channel_ti.npz``. **A real training
image: the Strebelle channels** repeats the conditional workflow on a classic
MPS benchmark. **Nonstationary geometry with the Strebelle channels** and
**Radial nonstationarity** keep the same TI but change the local data-event
geometry with rotation and anisotropy maps.

The remaining 2D examples broaden the data model. **Multivariate channels with
a context variable** uses an exhaustive continuous context field to guide a
categorical channel simulation. **Variation distance for continuous
conditioning** then returns to continuous conditioning and shows why comparing
local variation can be useful when hard data have a different absolute level
than the TI. **Bivariate joint simulation from a channel TI** simulates
categorical and continuous variables jointly from a Strebelle-derived bivariate
TI. **Continuous conditioning with a bundled texture** uses the GAIA-UNIL
``stone`` training image and compares the simulated and training-image
histograms.

Finally, **3D categorical simulation with voxel and slice plots** changes the
dimensionality rather than the variable type. It keeps the volume small for the
documentation build while showing that the same Direct Sampling workflow
applies to structured 3D arrays.

Examples
--------
