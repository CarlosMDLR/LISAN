"""
making_catalog.py

Create source catalogs from galaxy cutouts using GNUastro:
NoiseChisel -> Segment -> MakeCatalog -> Gaia query/match -> filtering -> plots.

This module is meant to be imported and called from the LISAN main script.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterable, Optional, List

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.ticker import AutoMinorLocator
from tqdm import tqdm


def _run(cmd: List[str], *, capture: bool = False) -> str:
    """Run external command (no shell). Return stdout if capture=True. Silence stdout/stderr."""
    p = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return (p.stdout or "").strip()


def make_catalogs(
    input_dir: Path,
    filters: Iterable[str],
    *,
    output_dir: Path = Path("Process_data/Make_catalogs"),
    zeropoint: float = 22.5,
    noisechisel_args: Optional[List[str]] = None,
    segment_args: Optional[List[str]] = None,
    aperture_arcsec: float = 10.0,
    gaia_dataset: str = "dr3",
    gaia_sigma: float = 3.0,
    make_plots: bool = True,
    show_progress: bool = True,
) -> None:
    """
    Build catalogs and Gaia matches for all FITS images in input_dir for each filter.

    Parameters
    ----------
    input_dir : Path
        Directory with galaxy FITS cutouts.
    filters : Iterable[str]
        Filters to process (e.g. ["g", "r"]).
    output_dir : Path
        Base output directory for products.
    zeropoint : float
        Zeropoint passed to astmkcatalog.
    noisechisel_args : list[str], optional
        Extra args for astnoisechisel (defaults mimic your original).
    segment_args : list[str], optional
        Extra args for astsegment (defaults mimic your original, using a gaussian kernel).
    aperture_arcsec : float
        Matching aperture in arcsec for astmatch (catalog vs Gaia).
    gaia_dataset : str
        Gaia dataset for astquery (e.g. "dr3").
    gaia_sigma : float
        Sigma threshold for parallax/PM filtering (default 3).
    make_plots : bool
        If True, generate mag vs half-sum-radius plots.
    show_progress : bool
        If True, show tqdm progress.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Directories
    dir_nc = output_dir / "Mask_noisechisel"
    dir_seg = output_dir / "Mask_segment"
    dir_gaia = output_dir / "Data_catalog_gaia_match"
    dir_nc.mkdir(parents=True, exist_ok=True)
    dir_seg.mkdir(parents=True, exist_ok=True)
    dir_gaia.mkdir(parents=True, exist_ok=True)

    if noisechisel_args is None:
        noisechisel_args = [
            "--tilesize=20,20",
            "--interpnumngb=1",
            "--qthresh=0.5",
            "--minnumfalse=1",
        ]

    # Kernel file (created once)
    kernel_fat = output_dir / "kernel_fat.fits"
    if not kernel_fat.exists():
        _run([
            "astmkprof",
            "--kernel=gaussian,1,3",
            "--oversample=1",
            f"--output={kernel_fat}",
        ])

    if segment_args is None:
        segment_args = [
            "--tilesize=20,20",
            f"--kernel={kernel_fat}",
            "--interpnumngb=1",
            "--gthresh=-10",
            "--objbordersn=15",
            "--minnumfalse=1",
            "--keepmaxnearriver",
        ]

    # Process per filter
    for flt in filters:
        # Match files like "...<filter>...fits" (same spirit as original)
        pat = re.compile(rf"{re.escape(flt)}.*\.fits$", re.IGNORECASE)
        files = sorted([p for p in input_dir.iterdir() if p.is_file() and pat.search(p.name)])
        if not files:
            continue

        dir_cat = output_dir / f"Data_catalog_{flt}"
        dir_plots = output_dir / "Plots_mag_vs_hr" / f"Plots_{flt}"
        dir_cat.mkdir(parents=True, exist_ok=True)
        if make_plots:
            dir_plots.mkdir(parents=True, exist_ok=True)

        it = tqdm(files, desc=f"Catalogs {flt}", disable=not show_progress)
        for image in it:
            obj = image.name.split("_")[0]

            nc_out = dir_nc / f"nc-{image.name}"
            seg_out = dir_seg / f"seg-{image.name}"

            # 1) NoiseChisel
            _run([
                "astnoisechisel",
                str(image),
                f"--output={nc_out}",
                *noisechisel_args,
            ])

            # 2) Segment (use INPUT-NO-SKY HDU like original)
            _run([
                "astsegment",
                str(nc_out),
                "-hINPUT-NO-SKY",
                f"--output={seg_out}",
                *segment_args,
            ])

            # 3) MakeCatalog
            cat_fits = dir_cat / f"{image.stem}_cat_{flt}.fits"
            _run([
                "astmkcatalog",
                str(seg_out),
                "--ids",
                "--ra",
                "--dec",
                "--magnitude",
                "--half-sum-sb",
                "--sb",
                "--half-sum-area",
                "--half-max-area",
                "--half-max-sb",
                "--half-sum-radius",
                "--sn",
                "--half-max-radius",
                f"--zeropoint={zeropoint}",
                "--clumpscat",
                f"--output={cat_fits}",
            ])

            # 4) Gaia query over the image footprint
            gaia_fits = dir_gaia / f"{obj}_gaia_{flt}.fits"
            _run([
                "astquery",
                "gaia",
                f"--dataset={gaia_dataset}",
                f"--overlapwith={image}",
                "-csource_id,ra,dec,phot_g_mean_mag,parallax,parallax_error,pmra,pmdec,pmra_error,pmdec_error",
                f"--output={gaia_fits}",
            ])

            # 5) Match catalog <-> Gaia
            match_fits = dir_gaia / f"match_gaia_{obj}_{flt}.fits"
            _run([
                "astmatch",
                str(cat_fits),
                str(gaia_fits),
                f"--aperture={aperture_arcsec}/3600",
                "--hdu=CLUMPS",
                "--hdu2=QUERY",
                "--ccol1=RA,DEC",
                "--ccol2=ra,dec",
                f"--output={match_fits}",
            ])

            # 6) Filter + plots (optional)
            if not make_plots:
                continue

            with fits.open(cat_fits) as hd:
                d = hd[1].data
                mag = d["MAGNITUDE"]
                hr = d["HALF_SUM_RADIUS"]

            with fits.open(match_fits) as hd:
                d1 = hd[1].data
                mag_gaia = d1["MAGNITUDE"]
                hr_gaia = d1["HALF_SUM_RADIUS"]

                dq = hd[2].data  # QUERY
                par = dq["parallax"]
                par_e = dq["parallax_error"]
                pmra = dq["pmra"]
                pmdec = dq["pmdec"]
                pmra_e = dq["pmra_error"]
                pmdec_e = dq["pmdec_error"]
                magG = dq["phot_g_mean_mag"]

            # Criteria (same logic, but parameterized by gaia_sigma)
            bol_par = par >= gaia_sigma * par_e
            cond_pm = (
                (~np.isnan(pmra) & (np.abs(pmra) >= gaia_sigma * pmra_e))
                | (~np.isnan(pmdec) & (np.abs(pmdec) >= gaia_sigma * pmdec_e))
            )
            sel = bol_par | cond_pm

            mag_gaia_sel = mag_gaia[sel]
            hr_gaia_sel = hr_gaia[sel]
            magG_sel = magG[sel]

            # Plot 1: detections + Gaia (filter mag)
            fig, ax = plt.subplots()
            ax.plot(np.log10(hr), mag, "k.", label="Detections")
            ax.plot(np.log10(hr_gaia), mag_gaia, "ro", markerfacecolor="none", label="Gaia detections")
            ax.plot(np.log10(hr_gaia_sel), mag_gaia_sel, "gs", markersize=3, label=f"Gaia ({gaia_sigma}σ par/PM)")
            ax.set_xlabel("log(Half sum radius [px])", fontsize=18)
            ax.set_ylabel(f"Magnitude in {flt} filter", fontsize=18)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(direction="in", which="minor", length=4)
            ax.set_title(f"{obj} Filter {flt}")
            ax.legend(loc="best")
            fig.savefig(dir_plots / f"{obj}_{flt}.jpg", bbox_inches="tight", pad_inches=0.05, dpi=400)
            plt.close(fig)

            # Plot 2: Gaia G band
            fig, ax = plt.subplots()
            ax.plot(np.log10(hr_gaia), magG, "ro", markerfacecolor="none", label="Gaia detections")
            ax.plot(np.log10(hr_gaia_sel), magG_sel, "gs", markersize=3, label=f"Gaia ({gaia_sigma}σ par/PM)")
            ax.set_xlabel("log(Half sum radius [px])", fontsize=18)
            ax.set_ylabel("Magnitude in Gaia G band", fontsize=18)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(direction="in", which="minor", length=4)
            ax.set_title(f"{obj} Filter {flt}")
            ax.legend(loc="best")
            fig.savefig(dir_plots / f"{obj}_{flt}_G_band.jpg", bbox_inches="tight", pad_inches=0.05, dpi=400)
            plt.close(fig)
