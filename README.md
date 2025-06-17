# LISAN — Layered Intensity Spread and Analysis for Night-sky structures

**LISAN** is a Python-based pipeline for constructing and analyzing extended Point Spread Functions (PSFs) in wide-field astronomical imaging. Designed for deep surface photometry studies, LISAN enables accurate modeling of the instrumental and atmospheric scattering that contaminates the outskirts of bright objects—especially stars—and biases the detection of faint features such as stellar halos, intracluster light (ICL), or tidal debris.

---

## 🚀 Project Goals

LISAN is specifically designed to:

- Build high-fidelity, radially extended PSF models from science and calibration fields.
- Combine information from both unsaturated and saturated stars to recover the PSF over several orders of magnitude in brightness.
- Separate the PSF into radial regimes based on stellar brightness and signal-to-noise.
- Ensure smooth transitions between radial PSF sections by matching S/N levels in overlapping zones.
- Output normalized PSF profiles that can be used for convolution, subtraction, or decontamination of scattered light.

It is particularly optimized for instruments like the **INT/WFC**, but can be adapted to other facilities with similar data.

---

## 🔬 Scientific Context

Deep imaging surveys are limited by scattered light from bright stars and the inner regions of galaxies. This contamination arises from the extended wings of the PSF, which are often poorly modeled in standard pipelines. LISAN tackles this problem by:

- Using Gaia DR3 to select reliable point sources across a wide dynamic range.
- Computing PSFs in concentric radial bins: inner, sub-intermediate, intermediate, and outer.
- Optionally including a fifth extrapolated region to reach arcminute scales.
- Employing centroid refinement via Lorentzian fits and flux normalization through annular scaling.

These steps ensure a realistic, data-driven PSF that can be applied to complex scenes where scattered light bias is critical.

---

## 📁 Project Structure (planned)

LISAN/
├── lisan_core.py # Core logic for radial PSF stacking
├── star_selection.py # Stellar filtering and Gaia crossmatch
├── centroid_refinement.py # Subpixel centering of image cutouts
├── psf_builder.py # Multi-bin radial stacking and joining
├── config.yaml # User configuration file
├── data/ # Raw input catalogues and image cutouts
├── output/ # Final PSF models and diagnostics
└── notebooks/ # Example Jupyter notebooks

---

## 📦 Dependencies

LISAN uses:

- `astropy`
- `photutils`
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `pyyaml`
- Gnuastro tools (optional, for segmentation and masking)

A `requirements.txt` and `environment.yml` will be provided in the first public release.

---

## 🛠️ Current Status

LISAN is under active development. The first functional version will:

- Accept science images and Gaia catalogues as input
- Apply selection cuts to build a stellar sample per magnitude bin
- Compute radial PSFs for each bin and join them with S/N-matched transitions
- Output normalized PSF profiles in physical and pixel units, with diagnostics

### Future features

- Instrument generalization
- Full integration with halo subtraction codes like MAHDI
- Interactive visualization of residuals and extrapolations

---
## 📄 License

This code will be released under the MIT license.

---

## 🤝 Acknowledgements

Development supported by the Instituto de Astrofísica de Canarias (IAC). Stellar catalogues are based on data from Gaia DR3.

If you use LISAN in your work, please cite the forthcoming associated paper and credit this repository.
