# Technical information

This document provides a technical overview of the VTA (Volume of Tissue Activated) optimisation application, focusing on the underlying algorithms and data processing pipelines. The app is designed for Deep Brain Stimulation (DBS) research and clinical applications, particularly for optimising electrode contact and current selection in targets like the Subthalamic Nucleus (STN) for Parkinson's Disease (PD). It integrates with established tools such as Lead-DBS (for electrode localisation and anatomical atlases) and OSS-DBS (for electric field simulations). The primary goal is to automate contact selection, estimate optimal currents, and compute VTA overlaps with STN subregions (motor, associative, and limbic) to maximise therapeutic coverage while minimising spillover, with optional clinical evaluation input.

The app assumes pre-processed data from Lead-DBS, with electrode reconstructions (`reconstruction.mat`) and atlases (i.e. DISTAL atlas for STN subnuclei). All computations are performed in native patient space.

## Dependencies and Environment

- **Core Libraries**: NumPy, Pandas, SciPy (for interpolation, ranking, and spatial operations), Matplotlib/PyQtGraph (for plotting), PyQt5 (for GUI), NG-Solve/DiPy/Neuron (prerequisites of OSS-DBS as a subprocess, though not directly used in core algorithms).
- **External Integrations**:
  - Lead-DBS outputs: Provides electrode coordinates and atlases.
  - OSS-DBS (Python package): Used for finite element method (FEM) simulations of electric fields.
- **Other Modules**: Custom utilities for electrode dimensions (`electrodes.py`), clinical review nudges (`review_weights.py`), runtime caching (`runtime_storage.py`), and logging (`woopsies.py`).
- **Execution**: The app is executable with a Python installation, with provisional PyInstaller adjustments for standalone use. OSS-DBS is called via subprocess.

No internet access is required during runtime. Simulations are computationally intensive and multi-threaded for batch processing.

## Overall Architecture

The entry point is `VTA_start.py`, which initialises a PyQt5-based GUI with two tabs:

1. **Calculation Tab (Batch Processing)**: For multi-patient optimisation. Users select a Lead-DBS output folder ([...]/derivatives/leaddbs), patients, contact mode (monopolar/bipolar), and current settings. Clicking "Run" launches a QThread (`settings` class) that calls `contacts_mt.external_call`, triggering a pipeline of contact selection, simulation, and overlap computation. Results are saved as `overlaps.csv` in the base directory, with logs for errors (`woopsies.log`) and info (`info.log`).

2. **Single Patient Tab (Fine-Tuning)**: Loads `overlaps.csv` for a selected patient, interpolates data, and provides interactive sliders for current adjustment. Plots show STN coverage and spillover. Supports saving HTML reports with embedded plots. This tab's functionality is fully contained in `VTA_start.py`.

The UI is defined in `VTA_optimisation_Qt_extended.py` (generated from a PyQt .ui file), with dynamic visibility for elements like progress bars.

## Core Algorithms

### 1. Electrode Localisation and Geometry

- **Input**: Lead-DBS `reconstruction.mat` file (HDF5 format), containing electrode coordinates in native space (scrf).
- **Algorithm** (in `electrodes.py` and `contacts_mt.py`):
  - Extract coordinates for left/right hemispheres.
  - Determine electrode type (e.g., Medtronic 3389, Abbott Directed) and look up specifications (tip length, contact length, spacing, diameter).
  - Compute direction vector and rotation (for directional electrodes) using the reconstruction information and electrode specification.
- **Output**: Generates OSS-DBS JSON parameters, including contact currents, encapsulation, and brain region centering.

### 2. Atlas Loading and Mesh Calculations

- **Atlases**: DISTAL Minimal (Ewert 2017) atlas, with STN subregions (motor, associative, limbic). Loaded from `.mat` files via SciPy/h5py.
- **Algorithm** (in `contact_selection_module.py` and `current_estimation.py`):
  - **Atlas Loading**: Load coordinates for left/right hemispheres from atlas files, representing STN subregions as point clouds in native patient space.
  - **Discretisation for Current Estimation** (in `current_estimation.py`):
    - Load the E-field lattice from OSS-DBS output (E_field_Lattice.csv), containing coordinates (x, y, z) and magnitudes.
    - Transform E-field coordinates to atlas voxel space using the scaling matrix and offset from the loaded atlas (DISTAL Minimal, motor subregion).
    - Filter coordinates to those within the atlas bounds (valid voxel indices) and inside the motor STN subregion (where atlas values are positive).
    - Compute the median E-field magnitude for points inside the motor STN to estimate the current scaling factor (targeting 0.2 V/mm for axon activation).
    - Purpose: Enables efficient identification of E-field points within the motor STN to calculate the median field strength, which is used to scale the stimulation current to achieve the desired activation threshold.
  - **Mesh Calculations for Overlap** (in `contacts_mt.py`):
    - Use PyVista's `delaunay_3d` to define the convex hull of atlas subregion point clouds (motor, associative, limbic).
    - For electric field thresholding: Load OSS-DBS E-field VTK output (`E-field.vtu`), identify points where |E| > 0.2 V/mm (VTA approximation), and check containment within convex hulls using surface clipping.
    - Compute overlaps as the ratio of activated points (within hull and above threshold) to total subregion points.
- **Purpose**: Discretisation supports efficient median E-field estimation for current scaling, while convex hulls enable precise overlap calculations for VTA coverage and spillover metrics.

### 3. Contact Selection

- **Algorithm** (in `contact_selection_module.py`):
  - Compute distances from each contact to the motor STN centroid.
  - Compute phi values (contact angular positions relative to the motor STN centroid, for directional selectivity).
  - **Scoring**:
    - Raw score = rank(|distance|) + rank(|phi|) using SciPy's `stats.rankdata`.
    - Optional **clinical nudge** (from `review_weights.py`): Multiply by (weights * 0.5 + 0.5), where weights are derived from Excel-based clinical reviews (improvements in rigidity, akinesia, tremor, masked by side effects onset).
  - **Selection**:
    - Sort scores ascending (lower is better).
    - Select top 2-3 contacts: Include 3rd if its difference to 2nd is <= difference between 1st and 2nd, and total < best score.
    - For non-directional electrodes (4 contacts), reduce to 1-2.
- **Clinical Nudge Details** (in `review_weights.py`):
  - For each coarse contact evaluation: Sum symptom improvements, apply linear scaling (decreasing weights for later steps), cumulative bounded differences ("negative" improvements, with positive cumulative score differences, are clipped to 0).
  - Normalise across contacts, expand the 4 coarse contacts from the evaluations to 8 contacts for directional leads (<i>C<sub>0</sub></i> → C<sub>0</sub>, <i>C<sub>1a/b/c</sub></i> → C<sub>1</sub>, <i>C<sub>2a/b/c</sub></i> → C<sub>2</sub>, <i>C<sub>3</sub></i> → C<sub>3</sub>).
  - With missing or invalid clinical data input it will fall back to uniform weights (1/4 per contact) as a default.
  - Contact nudge = 2 - weights. (centre around 1, with improvements towards 0).
- **Rationale**: Balances anatomical proximity to motor STN with directional optimisation and clinical feedback. Inspired by sweet-spot methods in literature, but here anatomy-focused with optional empirical adjustment.

### 4. Current Estimation

- **Algorithm** (in `current_estimation.py`):
  - Simulate at 4 mA using OSS-DBS (via `contacts_mt.py`).
  - Load E-field lattice (`E_field_Lattice.csv`).
  - Transform field coordinates to atlas voxel space (using scaling/offset).
  - Filter points inside motor STN.
  - Compute median |E-field| in target.
  - Scale current = 0.2 / median (to achieve 0.2 V/mm threshold for axon activation).
  - Fallback: scale current by 1.0 if no overlap or low E-field magnitude (median < 1e-6, logs warning).
- **Rationale**: 0.2 V/mm is a typical threshold for VTA based on axon models. Ensures comparable activation across patients despite tissue variability.

### 5. VTA Simulation and Overlap Calculation

- **Simulation** (in `contacts_mt.py`):
  - Generate OSS-DBS JSON: Set contacts/currents (mono/bi-polar, case as ground if needed), brain region, encapsulation (e.g., 0.5 mm thickness), FEM settings.
  - Run OSS-DBS via subprocess.
  - Multi-threaded for left/right, multiple currents (e.g., 0.8-3.2 mA steps if enabled).
- **Overlap Algorithm** (in `contacts_mt.py`, `single_folder` function):
  - Load E-field from OSS-DBS (`E-field.vtu`) and transform to atlas space.
  - For each subregion (motor, associative, limbic):
    - Define subregion mesh using PyVista's `delaunay_3d` on atlas point clouds.
    - Identify E-field points where |E| > 0.2 V/mm (binary VTA approximation).
    - Use surface clipping to check containment of E-field volume within each subregion's convex hull.
    - Compute coverage as the ratio of activated volume (within hull and above threshold) to total subregion volume.
    - Compute the fraction of activated volume outside the motor STN sub-region as a spillover estimate.
  - Motor VTA: Fraction of total VTA volume within the motor STN hull.
- **Output**: Appends to `overlaps.csv` (identifier/patient number (ipn), contacts, side, current, motor/assoc/limbic overlaps, motor_VTA).
- **Rationale**: Uses FEM-based fields from OSS-DBS for accuracy. The 0.2 V/mm threshold aligns with literature for effective stimulation radius (2-5 mm). Mesh-based overlap calculations via convex hulls ensure computational efficiency and anatomical precision for quantifying sweet-spot targeting.

### 6. Single Patient Fine-Tuning

- **Algorithm** (in `VTA_start.py`):
  - Load/filter `overlaps.csv` for patient/side.
  - Interpolate vs. current (SciPy CubicSpline) for smooth curves (0.5-4.0 mA, 0.1 mA steps). An additional (0,0) point is also considered to avoid free extrapolation in the 0.5-0.8 mA range.
  - Slider updates: Plots coverage (bar: motor/assoc/limbic) and spillover (line: vs. current).
  - Harmonic mean (optional): VTA containment, limbic/associative optimisation.
- **Saving**: Exports HTML with plots (PyQtGraph exporters) and data snapshots.
- **Rationale**: Enables interactive exploration for clinical refinement, using pre-computed data for speed.

## Error Handling and Logging

- `woopsies.py`: Tracks errors (e.g., missing files, low fields) and info (e.g., scores, ratios). Outputs to logs.

## Limitations and Extensions

- Assumes isotropic conductivity; OSS-DBS can be extended for anisotropy.
- Clinical nudges require a formatted Excel sheet; missing data defaults to uniform weights.
- Simulations are time-intensive; caching reduces repeats.


For usage instructions, see the app GUI and the corresponding README sections.

References: 
- Lead-DBS: Horn et al. (2015, 2019) : https://doi.org/10.1016/j.neuroimage.2014.12.002, https://doi.org/10.1016/j.neuroimage.2018.08.068 
- OSS-DBS: Butenko et al. (2020) : https://doi.org/10.1371/journal.pcbi.1008023
- Contact Selection: Inspired by sweetspot/VTA-based methods (Dembek et al., 2019 : https://doi.org/10.1002/ana.25567)
- VTA Thresholds: Åström et al., 2015 : https://doi.org/10.1109/tbme.2014.2363494