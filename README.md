# LISAN — Layered Intensity Spread and Analysis for Night-sky structures

**LISAN** is a Python-based pipeline for constructing and analyzing extended Point Spread Functions (PSFs) in wide-field astronomical imaging. It is designed for deep surface photometry studies, where scattered light from bright sources limits the detection of faint structures such as stellar halos, intracluster light (ICL), or tidal features.

LISAN provides a modular, command-line–driven workflow that combines image processing, star selection, and multi-regime PSF construction in a reproducible and fully scriptable way.

## 🚀 Project Goals

LISAN is designed to:

- Build high-fidelity, radially extended PSF models from science images.
- Use Gaia DR3 to select reliable point sources over a wide dynamic range.
- Separate the PSF into radial regimes (inner, sub-intermediate, intermediate, outer).
- Refine stellar centroids using Lorentzian fits.
- Normalize and stack stellar cutouts to recover PSF wings over several orders of magnitude.
- Produce PSF stacks and radial profiles ready for convolution, subtraction, or scattered-light correction.

The pipeline is currently optimized for **INT/WFC** data, but the design is instrument-agnostic and can be adapted to other facilities.

## 🔬 Scientific Context

Deep imaging surveys are fundamentally limited by scattered light from bright stars and galaxies. This contamination originates in the extended PSF wings, which are often poorly characterized by standard pipelines.

LISAN addresses this problem by:

- Selecting clean stellar samples using Gaia parallaxes and proper motions.
- Identifying the point-like branch in magnitude–size space.
- Building PSFs in magnitude-dependent radial bins.
- Ensuring consistency between radial regimes through controlled normalization.
- Allowing user control over centroiding, masking, and selection parameters.

The resulting PSFs are fully data-driven and suitable for low surface-brightness science.

## 🧱 Pipeline Philosophy

LISAN is a **multi-stage pipeline** executed from the command line.  
Each stage produces intermediate products required by the following steps.

### Recommended execution order

1. Measure image depth  
2. Create masks  
3. Build photometric and Gaia-matched catalogs  
4. Construct the PSF  

Each stage can be executed independently via command-line flags.

## ▶️ How to Use LISAN

All pipeline stages are controlled through the main entry point:

```
python3 lisan.py --help

```
## 1️⃣ Measure Image Depth

Estimate limiting surface-brightness levels for each filter.

```
python3 lisan.py \
  --dir ./data/cutouts \
  --filters g,r \
  --measure-depth
```
## 2️⃣ Build Masks

Create segmentation masks for sources and background estimation.

```
python3 lisan.py \
  --dir ./data/cutouts \
  --filters g,r \
  --make-masks

```
## 3️⃣ Build Catalogs (Required for PSF)

Generate photometric catalogs and Gaia cross-matches.  
**This step must be completed before PSF construction.**

```
python3 lisan.py \
  --dir ./data/cutouts \
  --filters g,r \
  --make-catalogs

```
## 4️⃣ Build the PSF

Construct the extended PSF using magnitude-binned stellar samples.

```
python3 lisan.py \
  --dir ./data/cutouts \
  --filters g,r \
  --build-psf

```
## 📁 Directory Structure
LISAN/
├── lisan.py
├── modules/
│ ├── psf_builder.py
│ ├── masking.py
│ ├── making_catalogs.py
│ └── measure_depth.py
├── Process_data/
│ ├── Mask_data/
│ ├── Make_catalogs/
│ └── Building_PSF/
├── PSF_files/
│ └── Inner_parts/
└── README.md

## 📦 Dependencies

LISAN relies on:

- numpy  
- scipy  
- astropy  
- matplotlib  
- pandas  
- tqdm  
- **Gnuastro** (`astnoisechisel`, `astsegment`, `astscript-psf-stamp`, `astarithmetic`)

## 📄 License

This project is released under the **MIT License**.

## 🤝 Acknowledgements

Development supported by the **Instituto de Astrofísica de Canarias (IAC)**.  
Stellar catalogues are based on data from **Gaia DR3**.

If you use LISAN in your work, please cite the forthcoming associated paper and credit this repository.



