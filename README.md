# LISAN — Layered Intensity Spread and Analysis for Night-sky structures

**Current version:** `v0.8.0-beta.1`

**LISAN** is a Python-based pipeline for constructing and analysing extended Point Spread Functions (PSFs) in wide-field astronomical imaging. It is designed for deep surface-photometry studies, where scattered light from bright sources can limit the detection and interpretation of very faint structures such as stellar halos, intracluster light (ICL), tidal streams, shells, and diffuse galaxy outskirts.

LISAN provides a modular, command-line-driven workflow that combines depth estimation, source masking, catalogue construction, Gaia-based stellar selection, magnitude-dependent PSF construction, interactive joining of radial PSF regimes, optional power-law extrapolation, and generation of a final circularized two-dimensional PSF.

The current implementation has been developed primarily for **INT/WFC** imaging, but much of the workflow is instrument-agnostic and can be adapted to other datasets after checking the assumptions described below.

---

## Table of Contents

- [Scientific Context](#-scientific-context)
- [Pipeline Overview](#-pipeline-overview)
- [Software Status and Compatibility](#-software-status-and-compatibility)
- [Installation](#-installation)
- [External Dependencies](#-external-dependencies)
- [Input Data Requirements](#-input-data-requirements)
- [Input FITS Format](#input-fits-format)
- [File Naming Convention](#file-naming-convention)
- [Basic Usage](#-basic-usage)
- [Recommended Execution Order](#-recommended-execution-order)
- [Stage 1 — Measure Image Depth](#1️⃣-stage-1--measure-image-depth)
- [Stage 2 — Build Masks](#2️⃣-stage-2--build-masks)
- [Stage 3 — Build Catalogs and Gaia Cross-matches](#3️⃣-stage-3--build-catalogs-and-gaia-cross-matches)
- [Stage 4 — Build the PSF](#4️⃣-stage-4--build-the-psf)
- [Stage 5 — Join the PSF](#5️⃣-stage-5--join-the-psf)
- [Output Directory Structure](#-output-directory-structure)
- [Command-Line Reference](#-command-line-reference)
- [Complete Example Workflow](#-complete-example-workflow)
- [Network Requirements](#-network-requirements)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🔬 Scientific Context

Deep astronomical images are often limited not only by photon statistics and sky subtraction but also by **scattered light**. Bright stars and galaxies redistribute light over very large angular scales through the instrumental and atmospheric PSF. The faint extended wings of the PSF can mimic, suppress, or distort real low-surface-brightness structures.

LISAN approaches this problem using a data-driven strategy:

1. sources are detected and segmented with GNUastro;
2. photometric measurements and size diagnostics are produced with `astmkcatalog`;
3. the source catalogue is cross-matched with Gaia DR3;
4. Gaia parallax and proper-motion information is used to identify reliable stellar sources;
5. the point-source branch is characterized in magnitude–half-sum-radius space;
6. stars are separated into magnitude-dependent PSF components;
7. stellar stamps are normalized, cleaned, masked, visually inspected, and stacked;
8. the resulting radial PSF components are joined interactively;
9. an optional power law can be fitted to the outer PSF and extrapolated to the requested final radius;
10. the final one-dimensional profile is circularized into a normalized 2D PSF.

---

## 🧱 Pipeline Overview

```text
Reduced + calibrated FITS images
            │
            ▼
    1. Measure depth
            │
            ▼
      2. Build masks
            │
            ▼
3. Build photometric catalogues
            │
            ▼
      Gaia DR3 query
            │
            ▼
      Gaia cross-match
            │
            ▼
      4. Build PSF
       ┌────┴────┐
       │         │
     INNER     OUTER
       │         │
       └────┬────┘
            ▼
  Individual PSF profiles
            │
            ▼
    5. Interactive join
            │
            ▼
   Joined radial PSF
            │
            ▼
 Optional power-law fit
            │
            ▼
Circularized final 2D PSF
```

The stages are modular and are controlled through `lisan.py`. The **catalogue stage must be completed before PSF building**, and the **PSF building stage must be completed before PSF joining**.

---

## 🧪 Software Status and Compatibility

LISAN is currently distributed as a beta release:

```text
LISAN v0.8.0-beta.1
```

The current version has been developed and tested with:

| Software | Tested version |
|---|---:|
| Python | **3.10.12** |
| GNUastro | **0.23** |
| NumPy | **1.26.4** |
| SciPy | **1.11.4** |
| Astropy | **5.3.4** |
| Matplotlib | **3.9.2** |
| Pandas | **2.2.3** |
| tqdm | **4.67.0** |

Newer versions may work, but the versions above correspond to the development/test environment for this beta release.

The current implementation is primarily tested under **Linux**. Some operations rely on Unix shell behaviour, GNUastro command-line programs, filename globbing, and utilities such as `pkill`.

Check the LISAN version with:

```bash
python3 lisan.py --version
```

Expected output:

```text
LISAN v0.8.0-beta.1
```

---

## 📦 Installation

A dedicated Python virtual environment is recommended.

```bash
python3 -m venv lisan_env
source lisan_env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

A minimal `requirements.txt` matching the tested environment is:

```text
numpy==1.26.4
scipy==1.11.4
astropy==5.3.4
matplotlib==3.9.2
pandas==2.2.3
tqdm==4.67.0
```

Verify the command-line interface with:

```bash
python3 lisan.py --help
```

---

## 🔧 External Dependencies

### GNUastro

LISAN has been developed and tested with **GNUastro 0.23**. The pipeline calls programs and scripts including:

```text
astnoisechisel
astsegment
astmkcatalog
astquery
astmatch
astmkprof
astfits
asttable
astarithmetic
astscript-psf-select-stars
astscript-psf-stamp
astscript-radial-profile
astscript-fits-view
```

Check the installation with:

```bash
astnoisechisel --version
```

### SAOImage DS9

The PSF-building stage opens selected stellar stamps for manual visual inspection through `astscript-fits-view`. The current workflow therefore expects **DS9** to be installed and available in the system PATH.

### LaTeX

The current interactive PSF-joining interface sets:

```python
plt.rcParams["text.usetex"] = True
```

A working LaTeX installation is therefore required by the current plotting configuration in `psf_joint.py`.

### Unix/Linux utilities

The current beta uses shell commands and utilities such as `pkill`. A Linux environment is strongly recommended.

---

## 📥 Input Data Requirements

LISAN expects a directory containing reduced astronomical FITS images. Different galaxies and different filters can coexist in the same input directory.

Example:

```text
data/
├── PGC10074_lum.fits
├── PGC10074_g.fits
├── PGC10074_r.fits
├── NGC1037_g.fits
├── NGC1037_r.fits
├── NGC1042_g.fits
└── NGC1042_r.fits
```

The directory itself can have any name. The galaxy name is **not** inferred from the directory name.

The input images should already have undergone the usual instrumental reduction. They should be bias-subtracted, flat-fielded, astrometrically calibrated, photometrically calibrated or associated with a known zeropoint, represented in linear image units, and accompanied by a valid celestial WCS.

LISAN is not a raw-data reduction pipeline.

---

## Input FITS Format

For the current beta release, the safest supported structure is:

```text
HDU 0 : Primary HDU
HDU 1 : 2D science image + valid celestial WCS
```

This is important because the current code explicitly uses HDU 1 in key stages, including depth measurement and PSF construction.

A conceptual input structure is therefore:

```text
No.    Name        Type          Dimensions
0      PRIMARY     PrimaryHDU
1      SCIENCE     ImageHDU      (Nx, Ny)
```

The extension name itself is not essential, but its position as **HDU 1** is important in the current implementation.

### WCS

The header associated with the science image must contain a valid celestial WCS because LISAN uses it for Gaia footprint queries, catalogue matching, and conversion between RA/Dec and image pixel coordinates.

### Image units

Input values should be in a consistent linear unit. The photometric zeropoint can be controlled independently for depth and catalogue measurements using:

```text
--depth-zeropoint
--cats-zeropoint
```

Both currently default to `22.5`.

---

## File Naming Convention

The recommended and effectively required convention for the complete workflow is:

```text
<GALAXY>_<FILTER>.fits
```

Examples:

```text
PGC10074_lum.fits
NGC1037_g.fits
NGC1037_r.fits
IC1101_g.fits
IC1101_r.fits
```

Several stages determine the object name using the first underscore-separated token. The PSF-joining stage interprets the first token as the galaxy and the second token as the filter.

Therefore, avoid galaxy identifiers containing additional underscores.

Recommended:

```text
NGC1037_g.fits
PGC10074_lum.fits
```

Avoid:

```text
NGC_1037_g.fits
PGC_10074_lum.fits
```

### Object-name resolution

During PSF construction, the galaxy identifier is passed to `SkyCoord.from_name`. The galaxy name encoded in the filename should therefore be resolvable by the Astropy name resolver.

---

## ▶️ Basic Usage

The main entry point is:

```bash
python3 lisan.py
```

Show all parameters:

```bash
python3 lisan.py --help
```

Show the version:

```bash
python3 lisan.py --version
```

The main global inputs are:

```text
--dir
--filters
```

Example:

```bash
python3 lisan.py \
    --dir ./data \
    --filters g,r \
    --measure-depth
```

Multiple filters are supplied as a comma-separated string, for example:

```text
--filters g,r,lum
```

---

## 🔄 Recommended Execution Order

For a complete reconstruction, use:

```text
1. Measure image depth
2. Build masks
3. Build photometric/Gaia catalogues
4. Build PSF components
5. Join PSF components
```

The critical dependency chain is:

```text
make-catalogs
      ↓
build-psf
      ↓
join-psf
```

---

# 1️⃣ Stage 1 — Measure Image Depth

Run:

```bash
python3 lisan.py \
    --dir ./data \
    --filters g,r \
    --measure-depth
```

Default settings:

```text
--depth-output-dir        Process_data/Measure_Depth
--depth-fits              depths.fits
--depth-zeropoint         22.5
--depth-sfmagarea         100
--depth-sfmagnsigma       3.0
--depth-upnsigma          3.0
```

Additional NoiseChisel parameters can be passed through `--depth-noisechisel-params`.

The depth stage uses `astnoisechisel`, `astmkprof`, `astmkcatalog`, and `astfits`.

### Outputs

Typical products are:

```text
Process_data/Measure_Depth/
├── depths.fits
└── Masks_noisechisel/
    └── <GALAXY>_<FILTER>/
        └── <GALAXY>_<FILTER>_masked.fits
```

The `depths.fits` file contains a binary table named `DEPTH` with:

```text
NAME
FILTER
SBLMAG
DATE
```

---

# 2️⃣ Stage 2 — Test and Tune Source Masks

The `--make-masks` stage is primarily intended as a **diagnostic and parameter-tuning step**.

It provides a convenient way to run GNUastro **NoiseChisel** and **Segment** directly on the input science images and inspect the resulting detection and segmentation maps before running the catalogue and PSF-construction stages.

In particular, this stage can be used to determine appropriate NoiseChisel and Segment parameters for a given dataset, instrument, image depth, or observing configuration.

Run:

```bash
python3 lisan.py \
    --dir ./data \
    --filters g,r \
    --make-masks
```

The default output root is:

```text
Process_data/Mask_data/
```

with:

```text
Process_data/Mask_data/
├── Mask_noisechisel/
│   ├── PGC10074_lum_noisechisel.fits
│   └── NGC1037_g_noisechisel.fits
└── Mask_segment/
    ├── PGC10074_lum_segment.fits
    └── NGC1037_g_segment.fits
```

### Important: these are diagnostic masks

The products generated by `--make-masks` are intended mainly for **visual inspection and parameter optimization**.

They are **not automatically reused by the `--make-catalogs` stage**.

When `--make-catalogs` is executed, LISAN runs NoiseChisel and Segment again and generates a new set of masks inside:

```text
Process_data/Make_catalogs/Mask_noisechisel/
Process_data/Make_catalogs/Mask_segment/
```

These latter products are the masks associated with the photometric catalogues and are the segmentation maps subsequently used during PSF construction.

Therefore, a typical workflow is:

```text
--make-masks
     │
     └── Test different NoiseChisel/Segment parameters
             │
             └── Inspect masks and choose suitable values
                         │
                         ▼
                  --make-catalogs
                         │
                         └── Generate the masks actually used
                             for catalogue and PSF construction
```

### Default diagnostic masking parameters

If no custom options are supplied, `--make-masks` currently uses the following NoiseChisel parameters:

```text
--tilesize=20,20
--interpnumngb=5
--dthresh=0.05
--snminarea=2
--rawoutput
--quiet
```

and the following Segment parameters:

```text
--tilesize=10,10
--interpnumngb=1
--gthresh=-10
--objbordersn=0
--minnumfalse=1
--quiet
```

These defaults are intended as starting values and should not necessarily be assumed to be optimal for every dataset.

Custom parameters can be passed with:

```text
--noisechisel-params
--segment-params
```

For example:

```bash
python3 lisan.py \
    --dir ./data \
    --filters g,r \
    --make-masks \
    --noisechisel-params="--tilesize=30,30 --interpnumngb=1 --qthresh=0.5 --minnumfalse=1" \
    --segment-params="--tilesize=20,20 --interpnumngb=1 --gthresh=-10 --objbordersn=15 --minnumfalse=1 --keepmaxnearriver"
```

Once a satisfactory detection and segmentation configuration has been identified, equivalent parameters should be supplied to the catalogue stage using `--cats-noisechisel-params` and `--cats-segment-params`.

---

# 3️⃣ Stage 3 — Build Catalogs and Gaia Cross-matches

This stage is **required before PSF construction**.

It is important to distinguish it from the previous `--make-masks` diagnostic stage.

`--make-catalogs` performs its **own NoiseChisel and Segment runs**. These runs generate the detection and segmentation products that are directly associated with the source catalogues and subsequently used by the PSF-building stage.

Run:

```bash
python3 lisan.py \
    --dir ./data \
    --filters g,r \
    --make-catalogs
```

The stage performs:

```text
Science image
     │
     ▼
 NoiseChisel
     │
     ▼
   Segment
     │
     ▼
 MakeCatalog
     │
     ▼
 Gaia DR3 query
     │
     ▼
Catalogue–Gaia match
     │
     ▼
Gaia parallax/proper-motion selection
     │
     ▼
Magnitude–size diagnostics
```

> **IMPORTANT — NoiseChisel and Segment configuration**
>
> The NoiseChisel and Segment parameters used in `--make-catalogs` are scientifically important because they determine the detections and segmentation maps from which the photometric source catalogue is constructed.
>
> In addition, the Segment products generated here are later used by LISAN during PSF construction when producing and masking the stellar PSF stamps.
>
> The masks previously generated with `--make-masks` are **not** automatically transferred to this stage.
>
> Users who have optimized their masking configuration with `--make-masks` should therefore explicitly provide the desired parameters to `--make-catalogs` using:
>
> ```text
> --cats-noisechisel-params
> --cats-segment-params
> ```

## Default NoiseChisel parameters used by `--make-catalogs`

If `--cats-noisechisel-params` is not specified, LISAN uses:

```text
--tilesize=20,20
--interpnumngb=1
--qthresh=0.5
--minnumfalse=1
```

These parameters are passed to `astnoisechisel` for each input science image.

The resulting files are stored in:

```text
Process_data/Make_catalogs/Mask_noisechisel/
```

with names of the form:

```text
nc-<INPUT_FILENAME>.fits
```

## Default Segment parameters used by `--make-catalogs`

Before segmentation, LISAN automatically creates a Gaussian convolution kernel using:

```text
astmkprof --kernel=gaussian,1,3 --oversample=1
```

which is saved as:

```text
Process_data/Make_catalogs/kernel_fat.fits
```

If `--cats-segment-params` is not specified, Segment then uses:

```text
--tilesize=20,20
--kernel=Process_data/Make_catalogs/kernel_fat.fits
--interpnumngb=1
--gthresh=-10
--objbordersn=15
--minnumfalse=1
--keepmaxnearriver
```

The resulting segmentation files are stored in:

```text
Process_data/Make_catalogs/Mask_segment/
```

with names of the form:

```text
seg-<INPUT_FILENAME>.fits
```

These segmentation products are particularly important because the PSF-building stage expects them at this location.

## Using optimized masking parameters

For example, after determining appropriate parameters during the `--make-masks` testing stage, they can be supplied to the catalogue stage as:

```bash
python3 lisan.py \
    --dir ./data \
    --filters g,r \
    --make-catalogs \
    --cats-noisechisel-params="--tilesize=30,30 --interpnumngb=1 --qthresh=0.5 --minnumfalse=1" \
    --cats-segment-params="--tilesize=20,20 --interpnumngb=1 --gthresh=-10 --objbordersn=15 --minnumfalse=1 --keepmaxnearriver"
```

### Parameter replacement behaviour

When `--cats-noisechisel-params` or `--cats-segment-params` is supplied, LISAN uses the values provided by the user instead of the corresponding internal default list.

Therefore, the custom argument should contain the **complete set of GNUastro parameters that the user wants to apply**, rather than only the parameter that is being changed.

For example, to modify only `--tilesize` while otherwise reproducing the current NoiseChisel defaults, use:

```bash
--cats-noisechisel-params="--tilesize=30,30 --interpnumngb=1 --qthresh=0.5 --minnumfalse=1"
```

rather than simply:

```bash
--cats-noisechisel-params="--tilesize=30,30"
```

unless the remaining parameters should intentionally revert to GNUastro's own defaults.

## Catalogue construction

After NoiseChisel and Segment, GNUastro `astmkcatalog` measures quantities including:

```text
ID
RA
DEC
MAGNITUDE
HALF_SUM_SB
SB
HALF_SUM_AREA
HALF_MAX_AREA
HALF_MAX_SB
HALF_SUM_RADIUS
SN
HALF_MAX_RADIUS
```

The catalogue is then cross-matched with Gaia.

The Gaia query requests:

```text
source_id
ra
dec
phot_g_mean_mag
parallax
parallax_error
pmra
pmdec
pmra_error
pmdec_error
```

The current catalogue defaults are:

```text
--cats-output-dir        Process_data/Make_catalogs
--cats-zeropoint         22.5
--cats-aperture-arcsec   10.0
--cats-gaia-dataset      dr3
--cats-gaia-sigma        3.0
```

Diagnostic plots can be disabled with:

```text
--cats-no-plots
```

### Outputs

```text
Process_data/Make_catalogs/
├── kernel_fat.fits
├── Mask_noisechisel/
│   └── nc-<INPUT_FILENAME>.fits
├── Mask_segment/
│   └── seg-<INPUT_FILENAME>.fits
├── Data_catalog_g/
│   └── <IMAGE_STEM>_cat_g.fits
├── Data_catalog_r/
│   └── <IMAGE_STEM>_cat_r.fits
├── Data_catalog_gaia_match/
│   ├── <GALAXY>_gaia_<FILTER>.fits
│   └── match_gaia_<GALAXY>_<FILTER>.fits
└── Plots_mag_vs_hr/
    └── Plots_<FILTER>/
        ├── <GALAXY>_<FILTER>.jpg
        └── <GALAXY>_<FILTER>_G_band.jpg
```

---

# 4️⃣ Stage 4 — Build the PSF

Run the default inner PSF construction with:

```bash
python3 lisan.py \
    --dir ./data \
    --filters g,r \
    --build-psf
```

Select which PSF regimes to construct with:

```text
--psf-select-parts I   # inner only
--psf-select-parts O   # outer only
--psf-select-parts B   # both
```

Example:

```bash
python3 lisan.py \
    --dir ./data \
    --filters g,r \
    --build-psf \
    --psf-select-parts B
```

## Inner PSF components

The default inner labels are:

```text
A,B,C
```

The point-like branch is estimated from a configurable Gaia magnitude interval:

```text
--psf-branch-mag-min 16.0
--psf-branch-mag-max 18.0
```

The current inner defaults are:

```text
--psf-parts             A,B,C
--psf-min-dist          0.015,0.015,0.015
--psf-norm-radii        5,10;10,20;20,40
--psf-width-image       200,200;400,400;800,800
--psf-selection-radii   90,90,90
```

Each inner star is selected, Gaia-matched, filtered in magnitude and half-sum radius, angularly restricted around the galaxy, recentered, stamped, masked, inspected, stacked, and converted to a radial profile.

### Inner centroid refinement

Inner stars are recentered using a local stamp and Lorentzian fits to the collapsed row and column profiles.

```text
--psf-center-sigma     5.0
--psf-center-max-iter  5
```

If `--psf-center-sigma 0` is used, the collapse is performed with `nanmean` rather than sigma-clipped statistics.

## Outer PSF components

Outer magnitude bins are generated automatically, approximately one magnitude wide, between the inner/outer transition and the brightest available Gaia-selected stars. They are named:

```text
Outer_1
Outer_2
Outer_3
...
```

The outer components do **not** apply the same half-sum-radius 2-sigma filtering used for the inner point-like branch after the magnitude selection. They also do **not** use the Lorentzian recentering step: Gaia RA/Dec is converted directly to pixel coordinates through the WCS.

Outer defaults:

```text
--psf-min-dist-outer          0.015
--psf-norm-radii-outer        40,80
--psf-width-image-outer       1600,1600
--psf-step-min-dist-outer     0.0
--psf-step-norm-radii-outer   0,0
--psf-step-width-image-outer  0,0
```

The step parameters allow the outer stamp properties to change progressively with outer magnitude bin.

# PSF stamp masking parameters

After each stellar PSF stamp has been extracted with `astscript-psf-stamp`, LISAN performs an additional masking step on the **individual stellar stamp** before it is accepted for stacking.

This masking step is independent of the NoiseChisel and Segment configuration used previously to create the full-image catalogues.

The four command-line parameters controlling this stage are:

```text
--psf-nc-inner-params
--psf-seg-inner-params
--psf-nc-outer-params
--psf-seg-outer-params
```

They allow the NoiseChisel and Segment configuration used to clean the individual stellar stamps to be adjusted independently for the inner and outer PSF components.

## What is masked at this stage?

The purpose of this masking step is **not to remove the star used to construct the PSF**.

Instead, LISAN attempts to preserve the central target star while masking contaminating sources falling within the same stamp, such as:

- neighbouring stars;
- compact galaxies;
- background sources;
- unrelated detections overlapping the PSF stamp.

The procedure applied to each stellar stamp is approximately:

```text
Extract normalized stellar stamp
             │
             ▼
 Convert zero-valued pixels to NaN
             │
             ▼
        NoiseChisel
             │
             ▼
          Segment
             │
             ▼
 Identify segmentation label(s)
 associated with the central star
             │
             ▼
 Preserve the central source
             │
             ▼
 Mask all other segmented sources
             │
             ▼
 Clean stellar stamp
             │
             ▼
     Visual inspection
             │
             ▼
          Stacking
```

LISAN identifies the central source from the segmentation map using both the label at the centre of the stamp and the dominant segmentation label in a small region around the centre. These central labels are excluded from the contaminant mask. All remaining non-zero segmentation labels are treated as contaminating sources and their corresponding pixels in the stellar stamp are replaced by `NaN`.

This is therefore an important step for preventing neighbouring sources from contributing flux to the final stacked PSF.

---

## Inner PSF stamp masking

The inner PSF components (`A`, `B`, `C`, or any alternative labels specified through `--psf-parts`) use:

```text
--psf-nc-inner-params
--psf-seg-inner-params
```

### `--psf-nc-inner-params`

Defines the GNUastro **NoiseChisel** parameters applied to each individual inner stellar stamp.

If this option is not supplied, LISAN uses:

```text
--tilesize=20,20
--outliernumngb=5
--interpnumngb=1
--qthresh=0.5
--minnumfalse=1
--rawoutput
```

### `--psf-seg-inner-params`

Defines the GNUastro **Segment** parameters used after NoiseChisel on each inner stellar stamp.

If this option is not supplied, LISAN uses:

```text
--tilesize=20,20
--snminarea=2
--interpnumngb=1
--gthresh=-10
--objbordersn=0
--minnumfalse=1
```

These parameters determine how contaminating objects surrounding the central PSF star are detected and segmented.

For example:

```bash
python3 lisan.py \
    --dir ./data \
    --filters g,r \
    --build-psf \
    --psf-select-parts I \
    --psf-nc-inner-params="--tilesize=20,20 --outliernumngb=5 --interpnumngb=1 --qthresh=0.5 --minnumfalse=1 --rawoutput" \
    --psf-seg-inner-params="--tilesize=20,20 --snminarea=2 --interpnumngb=1 --gthresh=-10 --objbordersn=0 --minnumfalse=1"
```

---

## Outer PSF stamp masking

The outer PSF components (`Outer_1`, `Outer_2`, ...) use:

```text
--psf-nc-outer-params
--psf-seg-outer-params
```

These options have the same purpose as their inner counterparts but make it possible to use a different masking configuration for the substantially larger and brighter-star stamps used to construct the outer PSF.

### `--psf-nc-outer-params`

Defines the NoiseChisel parameters applied to the individual outer stellar stamps.

### `--psf-seg-outer-params`

Defines the Segment parameters applied to the individual outer stellar stamps.

### Default behaviour for outer stamps

If no outer-specific parameters are supplied, LISAN **reuses the inner stamp-masking configuration**:

```text
outer NoiseChisel parameters = inner NoiseChisel parameters
outer Segment parameters      = inner Segment parameters
```

This inheritance also applies when the inner parameters have been customized.

For example:

```bash
--psf-nc-inner-params="..."
```

combined with no:

```bash
--psf-nc-outer-params
```

means that the same customized NoiseChisel configuration will also be used for the outer stamps.

Separate outer parameters only need to be supplied when a different masking behaviour is desired.

For example:

```bash
python3 lisan.py \
    --dir ./data \
    --filters g,r \
    --build-psf \
    --psf-select-parts B \
    --psf-nc-inner-params="--tilesize=20,20 --outliernumngb=5 --interpnumngb=1 --qthresh=0.5 --minnumfalse=1 --rawoutput" \
    --psf-seg-inner-params="--tilesize=20,20 --snminarea=2 --interpnumngb=1 --gthresh=-10 --objbordersn=0 --minnumfalse=1" \
    --psf-nc-outer-params="--tilesize=30,30 --outliernumngb=5 --interpnumngb=1 --qthresh=0.5 --minnumfalse=1 --rawoutput" \
    --psf-seg-outer-params="--tilesize=30,30 --snminarea=2 --interpnumngb=1 --gthresh=-10 --objbordersn=0 --minnumfalse=1"
```

> **IMPORTANT — These parameters are different from the catalogue masking parameters**
>
> There are three conceptually different NoiseChisel/Segment configurations in LISAN:
>
> ```text
> --noisechisel-params / --segment-params
>     → diagnostic full-image masks generated by --make-masks
>
> --cats-noisechisel-params / --cats-segment-params
>     → full-image detection and segmentation used to build the catalogues
>
> --psf-nc-*-params / --psf-seg-*-params
>     → masking of contaminating objects inside individual stellar PSF stamps
> ```
>
> They should not be confused with one another. In particular, optimizing the full-image segmentation does not necessarily imply that the same parameters are optimal for the much smaller stellar stamps.

### Parameter replacement behaviour

As in the catalogue stage, providing one of these options replaces the corresponding internal LISAN parameter list.

For example:

```bash
--psf-nc-inner-params="--tilesize=30,30"
```

does **not** mean “use the LISAN defaults but change `--tilesize`”.

It means that the NoiseChisel call receives the user-supplied parameter list instead of the LISAN default list. Any omitted options will therefore fall back to GNUastro's own behaviour.

To change only the tile size while retaining the rest of the current LISAN configuration, explicitly provide the complete desired set:

```bash
--psf-nc-inner-params="--tilesize=30,30 --outliernumngb=5 --interpnumngb=1 --qthresh=0.5 --minnumfalse=1 --rawoutput"
```

---

## Manual visual inspection

After automatic masking, the cleaned stamps are displayed using `astscript-fits-view`/DS9.

The user can visually inspect the selected stars and manually remove problematic stamps before the stacking stage.

This provides a final quality-control step for cases in which automatic masking is insufficient, for example:

- severely blended stars;
- residual extended galaxies;
- image artefacts;
- problematic saturation patterns;
- imperfect segmentation;
- stars whose PSF morphology is clearly anomalous.

Only the stamps remaining after this inspection are used to construct the corresponding stacked PSF component.
## Stacking

Accepted stellar stamps are combined with GNUastro `astarithmetic` using a sigma-clipped mean. The code checks image data in both HDU 0 and HDU 1 before validating the stack. A radial profile is then produced with `astscript-radial-profile` using `mean`, `std`, `area`, and `semi-major` measurements.

### Outputs

```text
Process_data/Building_PSF/
PSF_files/Inner_parts/
PSF_files/Outer_parts/
```

Typical final component directories are:

```text
PSF_files/
├── Inner_parts/
│   └── <GALAXY>/
│       ├── <GALAXY>_A_<FILTER>/
│       ├── <GALAXY>_B_<FILTER>/
│       └── <GALAXY>_C_<FILTER>/
└── Outer_parts/
    └── <GALAXY>/
        ├── <GALAXY>_Outer_1_<FILTER>/
        ├── <GALAXY>_Outer_2_<FILTER>/
        └── ...
```

Each completed component contains products such as:

```text
stack_<...>.fits
profile_psf_<...>.fits
```

---

# 5️⃣ Stage 5 — Join the PSF

The joining stage loads the available inner and outer PSF profiles and displays them in an interactive Matplotlib interface.

Run:

```bash
python3 lisan.py \
    --dir ./data \
    --filters g,r \
    --join-psf
```

The main program iterates over compatible FITS files inside `--dir`, obtains the galaxy and filter from each filename, and opens the joiner for each selected combination.

## Selecting galaxies

Use `--gals-to-use` to restrict the PSF-joining stage:

```bash
python3 lisan.py \
    --dir ./data \
    --filters g,r \
    --gals-to-use NGC1037,PGC10074 \
    --join-psf
```

In the current main implementation, `--gals-to-use` is applied when selecting galaxy/filter combinations for **PSF joining**.

## Interactive join interface

By default, the joiner searches:

```text
./PSF_files/Inner_parts
./PSF_files/Outer_parts
```

These roots can be changed with:

```text
--join-inner-root
--join-outer-root
```

The interface provides editable scale factors, overlap regions between consecutive profiles, axis-limit controls, overlap identifiers on the plot, and joining diagnostics.

The current joining procedure works sequentially from the **outermost PSF component toward the innermost component**. User-defined overlap intervals determine where adjacent components are allowed to match. Where standard-deviation information is available, S/N is used to select a matching radius; otherwise the implementation falls back to a signal-based criterion.

## External outer PSF

An optional external outer PSF can be supplied with:

```text
--join-external-outer-stack
--join-external-outer-profile
```

Example:

```bash
python3 lisan.py \
    --dir ./data \
    --filters g \
    --gals-to-use NGC1037 \
    --join-psf \
    --join-external-outer-stack ./external/outer_stack.fits \
    --join-external-outer-profile ./external/outer_profile.fits
```

When several galaxy/filter combinations are processed through the main script, the same external paths are forwarded to all selected join operations. For object-specific external PSFs, process the targets separately.

## Power-law extrapolation and circularized 2D PSF

Providing `--join-circular-radius` makes the main script request circular export from `psf_joint.py`.

Example:

```bash
python3 lisan.py \
    --dir ./data \
    --filters g \
    --gals-to-use NGC1037 \
    --join-psf \
    --join-circular-radius 1001
```

The requested radius **must be odd**.

Valid examples:

```text
1001
2001
6801
```

When circular export is requested, a second interactive window allows the user to choose the radial interval for fitting a power law and a PSF–power-law overlap interval. The fitted profile has the form:

```text
I(r) = A r^alpha
```

The fitted power law is matched to the measured PSF and extrapolated to the requested radius. The final radial profile is then converted to a circularized 2D PSF with GNUastro and normalized to unit total flux.

Final products are stored below:

```text
PSF_files/PSFs_complete/<GALAXY>_<FILTER>/
```

including:

```text
psf_<GALAXY>_<FILTER>.fits
psf_profile_<GALAXY>_<FILTER>.fits
<GALAXY>_<FILTER>_profile_tmp.fits
Intervals/
Custom_tables/
Circular_profiles/
```

---

## 📁 Output Directory Structure

After a complete run, the project can look approximately like:

```text
LISAN/
├── lisan.py
├── README.md
├── requirements.txt
├── LICENSE
├── modules/
│   ├── masking.py
│   ├── measure_depth.py
│   ├── making_catalogs.py
│   ├── psf_builder.py
│   └── psf_joint.py
├── data/
│   ├── PGC10074_lum.fits
│   ├── NGC1037_g.fits
│   └── NGC1037_r.fits
├── Process_data/
│   ├── Measure_Depth/
│   ├── Mask_data/
│   ├── Make_catalogs/
│   └── Building_PSF/
└── PSF_files/
    ├── Inner_parts/
    ├── Outer_parts/
    └── PSFs_complete/
```

Not every directory is generated by every stage.

---

## ⚙️ Command-Line Reference

The authoritative list is always available with:

```bash
python3 lisan.py --help
```

### Global input parameters

| Argument | Description | Default |
|---|---|---|
| `--version` | Print LISAN version and exit | — |
| `--dir` | Directory containing input FITS images | required |
| `--filters` | Comma-separated filters | required |
| `--gals-to-use` | Galaxy subset used by the PSF-joining selection | all |

### Masking stage

| Argument | Description | Default |
|---|---|---|
| `--make-masks` | Run masking | off |
| `--mask-output-dir` | Mask output root | `Process_data/Mask_data` |
| `--noisechisel-params` | Extra NoiseChisel arguments | module defaults |
| `--segment-params` | Extra Segment arguments | module defaults |

### Depth stage

| Argument | Description | Default |
|---|---|---|
| `--measure-depth` | Run depth measurement | off |
| `--depth-output-dir` | Depth output root | `Process_data/Measure_Depth` |
| `--depth-fits` | Depth FITS table | `depths.fits` |
| `--depth-zeropoint` | Photometric zeropoint | `22.5` |
| `--depth-sfmagarea` | Surface-brightness area parameter | `100` |
| `--depth-sfmagnsigma` | Limiting-magnitude sigma | `3.0` |
| `--depth-upnsigma` | Upper-limit sigma | `3.0` |
| `--depth-noisechisel-params` | Extra NoiseChisel arguments | module defaults |

### Catalogue stage

| Argument | Description | Default |
|---|---|---|
| `--make-catalogs` | Build photometric/Gaia catalogues | off |
| `--cats-output-dir` | Catalogue output root | `Process_data/Make_catalogs` |
| `--cats-zeropoint` | Photometric zeropoint | `22.5` |
| `--cats-noisechisel-params` | Extra NoiseChisel arguments | module defaults |
| `--cats-segment-params` | Extra Segment arguments | module defaults |
| `--cats-aperture-arcsec` | Catalogue/Gaia matching aperture | `10.0` |
| `--cats-gaia-dataset` | Gaia dataset | `dr3` |
| `--cats-gaia-sigma` | Astrometric significance threshold | `3.0` |
| `--cats-no-plots` | Disable catalogue diagnostic plots | off |

### PSF-building stage

| Argument | Description | Default |
|---|---|---|
| `--build-psf` | Run PSF construction | off |
| `--psf-select-parts` | `I`, `O`, or `B` | `I` |
| `--psf-parts` | Inner component labels | `A,B,C` |
| `--psf-min-dist` | Minimum source distance per inner component | `0.015,0.015,0.015` |
| `--psf-norm-radii` | Stamp normalization radii | `5,10;10,20;20,40` |
| `--psf-width-image` | Stamp dimensions | `200,200;400,400;800,800` |
| `--psf-selection-radii` | Angular selection radius in arcmin | `90,90,90` |
| `--psf-center-sigma` | Sigma clipping for centroid refinement; `0` disables clipping | `5.0` |
| `--psf-center-max-iter` | Maximum clipping iterations | `5` |
| `--psf-branch-mag-min` | Lower branch-search magnitude | `16.0` |
| `--psf-branch-mag-max` | Upper branch-search magnitude | `18.0` |

### PSF stamp masking

| Argument | Description |
|---|---|
| `--psf-nc-inner-params` | NoiseChisel arguments for inner stamps |
| `--psf-seg-inner-params` | Segment arguments for inner stamps |
| `--psf-nc-outer-params` | NoiseChisel arguments for outer stamps |
| `--psf-seg-outer-params` | Segment arguments for outer stamps |

### Outer PSF configuration

| Argument | Description | Default |
|---|---|---|
| `--psf-min-dist-outer` | Base outer minimum source distance | `0.015` |
| `--psf-norm-radii-outer` | Base outer normalization radii | `40,80` |
| `--psf-width-image-outer` | Base outer stamp dimensions | `1600,1600` |
| `--psf-step-min-dist-outer` | Increment per outer bin | `0.0` |
| `--psf-step-norm-radii-outer` | Normalization-radius increment per bin | `0,0` |
| `--psf-step-width-image-outer` | Stamp-size increment per bin | `0,0` |

### PSF joining

| Argument | Description | Default |
|---|---|---|
| `--join-psf` | Open interactive joiner | off |
| `--join-inner-root` | Root containing inner PSFs | `./PSF_files/Inner_parts` |
| `--join-outer-root` | Root containing outer PSFs | `./PSF_files/Outer_parts` |
| `--join-external-outer-stack` | Optional external 2D outer PSF | none |
| `--join-external-outer-profile` | Optional external radial outer profile | none |
| `--join-circular-radius` | Radius for power-law extrapolation and circular export; must be odd | none |

---

## 🧭 Complete Example Workflow

Given:

```text
data/
├── NGC1037_g.fits
├── NGC1037_r.fits
├── PGC10074_g.fits
└── PGC10074_r.fits
```

### 1. Measure depth

```bash
python3 lisan.py --dir ./data --filters g,r --measure-depth
```

### 2. Build masks

```bash
python3 lisan.py --dir ./data --filters g,r --make-masks
```

### 3. Build catalogues

```bash
python3 lisan.py --dir ./data --filters g,r --make-catalogs
```

### 4. Build inner and outer PSF parts

```bash
python3 lisan.py --dir ./data --filters g,r --build-psf --psf-select-parts B
```

### 5. Join only one galaxy

```bash
python3 lisan.py \
    --dir ./data \
    --filters g,r \
    --gals-to-use NGC1037 \
    --join-psf
```

### 6. Join and export circularized PSFs

```bash
python3 lisan.py \
    --dir ./data \
    --filters g,r \
    --gals-to-use NGC1037 \
    --join-psf \
    --join-circular-radius 1001
```

The joining GUI is opened separately for each selected galaxy/filter combination.

---

## 🌐 Network Requirements

Some LISAN stages require internet access.

`making_catalogs.py` uses GNUastro `astquery` to query Gaia over the WCS footprint of each image. In addition, `psf_builder.py` uses `SkyCoord.from_name` to obtain reference galaxy coordinates for angular selection.

Therefore, catalogue creation and normal PSF construction require a working network connection unless these external lookups are replaced by local data.

---

## 📚 Citation

If you use LISAN in scientific work, please cite the associated papers:

**Marrero-de la Rosa et al. 2026a**  
[https://doi.org/10.1051/0004-6361/202557193](https://doi.org/10.1051/0004-6361/202557193)

and:

**Marrero-de la Rosa et al. 2026b**  
[https://doi.org/10.48550/arXiv.2607.15340](https://doi.org/10.48550/arXiv.2607.15340)

Please also cite the specific LISAN software release used in your analysis and credit this repository.

---

## 📄 License

This project is released under the **MIT License**.

See the `LICENSE` file distributed with the repository for the complete license text.

---

## 🤝 Acknowledgements

Development of LISAN is supported by the **Instituto de Astrofísica de Canarias (IAC)**.

This work makes use of data from the **Gaia** mission and uses tools from the **GNU Astronomy Utilities (GNUastro)**.

The software additionally relies on the Python scientific ecosystem, particularly Astropy, NumPy, SciPy, Matplotlib, and related packages.

If LISAN is used in published scientific work, please cite the associated papers and the corresponding software release.

---

## Development note

LISAN `v0.8.0-beta.1` is an active beta release. The command-line interface, intermediate directory structure, and some implementation details may change before the first stable release. For reproducibility, use tagged releases for scientific analyses.
