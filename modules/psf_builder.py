# =========================
# modules/psf_builder.py
# =========================
"""
psf_builder.py (module)

INNER-only PSF builder compatible with the pipeline main.py.

Keeps original calculations and logic, but:
- No interactive prompts
- No DS9/manual deletion
- Structured for argparse-driven execution
- Saves the same diagnostic Mag vs log(HR) plot

Expected inputs (same as original project structure):
- Gaia-match catalogs:
    ./Process_data/Make_catalogs/Data_catalog_gaia_match/match_gaia_<GAL>_<FLT>.fits
- Segmentation cutouts:
    ./Process_data/Make_catalogs/Mask_segment/seg-<CUTOUT_FILENAME>
- Science cutouts provided via `directorio` in the pipeline.

Note:
- OUTER building is intentionally not implemented yet. If select_parts includes O,
  this module will raise NotImplementedError.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.visualization as vis

from tqdm import tqdm
from scipy import stats
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip, sigma_clipped_stats
from matplotlib.ticker import AutoMinorLocator


# -----------------------------------------------------------------------------
# Helpers (faithful math)
# -----------------------------------------------------------------------------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def obtain_coords(name: str) -> Tuple[float, float]:
    c = SkyCoord.from_name(name)
    return float(c.ra.deg), float(c.dec.deg)


def lorentzian(x, amplitude, center, width):
    return amplitude / (1 + ((x - center) / (0.5 * width)) ** 2)


def ajuste_gaussiano(datos_raw, center_1):
    """
    Fit Lorentzian to 1D data (historical name 'gaussiano').

    Keeps original behavior as much as possible:
    - subtracts nanmin baseline
    - replaces NaNs by 0 before curve_fit (as in your snippet)

    Returns
    -------
    mean, stddev, fit_curve (+ baseline) evaluated on full x-grid.
    """
    datos_1 = np.array(datos_raw, dtype=float).flatten()

    # If all NaN, bail out safely
    if not np.isfinite(datos_1).any():
        x_full = np.arange(len(datos_1), dtype=float)
        return float(center_1), 50.0, np.full_like(x_full, np.nan, dtype=float)

    baseline = np.nanmin(datos_1)
    datos = datos_1 - baseline

    # put nans to 0 (faithful to your current logic)
    datos_nan_mask = np.isnan(datos)
    datos[datos_nan_mask] = 0.0

    x = np.arange(0, len(datos), dtype=float)
    initial_params = [1.0, float(center_1), 50.0]

    # curve_fit cannot handle inf/nan
    if not np.isfinite(datos).all():
        x_full = np.arange(len(datos_1), dtype=float)
        return float(center_1), 50.0, np.full_like(x_full, np.nan, dtype=float)

    params, _ = curve_fit(lorentzian, x, datos, p0=initial_params)
    amplitude, mean, stddev = params
    fit_curve = lorentzian(x, amplitude, mean, stddev)
    return float(mean), float(stddev), fit_curve + baseline


def graph_interactive_center(
    data,
    center,
    show_fig: bool,
    size: int,
    *,
    sigma: float = 5.0,
    max_iter: int = 5,
):
    """
    Center refinement using Lorentzian fits to sigma-clipped row/col medians,
    but computed on a *local stamp* around the star for speed.

    The returned center is in FULL-IMAGE pixel coordinates (same convention as input).

    Parameters
    ----------
    data : 2D ndarray
        Full image array.
    center : (x, y)
        Initial center (pixels) in full-image coordinates.
    show_fig : bool
        Whether to show the diagnostic stamp + marginals (uses the stamp).
    size : int
        Half-size of the stamp (pixels). Stamp shape ~ (2*size+1, 2*size+1).
    sigma : float
        Sigma for sigma-clipped-stats.
    max_iter : int
        Max iterations for sigma-clipped-stats.

    Returns
    -------
    np.ndarray [y_center_refined, x_center_refined] in full-image coordinates.
    """
    x_center = int(float(center[0]))
    y_center = int(float(center[1]))

    # ---- 1) define local stamp bounds (full-image coordinates) ----
    ny, nx = data.shape[0], data.shape[1]
    x0 = max(0, x_center - size)
    x1 = min(nx, x_center + size + 1)  # +1 because slicing is exclusive
    y0 = max(0, y_center - size)
    y1 = min(ny, y_center + size + 1)

    # If stamp is too small, return original
    if (x1 - x0) < 5 or (y1 - y0) < 5:
        return np.array([float(y_center), float(x_center)])

    stamp = data[y0:y1, x0:x1]

    # ---- 2) center within stamp coordinates ----
    # (what was x_center/y_center in full image becomes local indices)
    x_c_loc = x_center - x0
    y_c_loc = y_center - y0

    # ---- 3) estimate 1D medians using sigma clipping on the stamp ----
    # NOTE: now the computation is O(stamp) instead of O(full image)
    if float(sigma) == 0.0:
        # Simple collapse, no clipping
        median_row = np.nanmean(stamp, axis=1)
        median_col = np.nanmean(stamp, axis=0)
    else:
        # Sigma-clipped medians (original logic)
        _, median_row, _ = sigma_clipped_stats(
            stamp,
            mask=None,
            sigma=float(sigma),
            axis=1,
            maxiters=int(max_iter),
        )
        _, median_col, _ = sigma_clipped_stats(
            stamp,
            mask=None,
            sigma=float(sigma),
            axis=0,
            maxiters=int(max_iter),
        )

    # ---- 4) Lorentzian fits in local coordinates ----
    # The initial centers for the 1D fits must be in local pixel units
    center_row_loc, _, fit_row = ajuste_gaussiano(median_row, y_c_loc)
    center_col_loc, _, fit_col = ajuste_gaussiano(median_col, x_c_loc)

    # Convert back to full-image coordinates
    center_row = center_row_loc + y0
    center_col = center_col_loc + x0

    # ---- 5) optional diagnostic plot (show stamp only) ----
    if show_fig:
        fig = plt.figure(layout="constrained")
        ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
        ax.set(aspect=1)

        ax_histx = ax.inset_axes([0, 1.05, 1, 0.25])
        ax_histy = ax.inset_axes([1.05, 0, 0.25, 1])

        norm = vis.ImageNormalize(vmin=0, vmax=10, stretch=vis.LogStretch(10000))
        ax.imshow(stamp, cmap="nipy_spectral", norm=norm)

        # draw refined centers in *stamp* coords
        ax.axhline(center_row_loc, ls="--", color="k")
        ax.axvline(center_col_loc, ls="--", color="k")

        ax_histx.plot(median_col, "k-")
        ax_histx.plot(fit_col, "r--")
        ax_histx.axvline(center_col_loc, ls="--", color="k")
        ax_histx.set_xlim(0, stamp.shape[1] - 1)

        ax_histy.set_ylim(stamp.shape[0] - 1, 0)
        ax_histy.plot(median_row, np.arange(0, len(median_row)), "k-")
        ax_histy.plot(fit_row, np.arange(0, len(median_row)), "r--")
        ax_histy.axhline(center_row_loc, ls="--", color="k")

        ax_histx.set_xticks([])
        ax_histy.set_yticks([])

        ax.set_xlabel("x (px, stamp)", fontsize=16)
        ax.set_ylabel("y (px, stamp)", fontsize=16)

        plt.show()

    # Return same ordering as your original: [row(y), col(x)] in full coords
    return np.array([float(center_row), float(center_col)])

def silent_run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def fits_hdu_has_data(path: Path, hdu_index: int) -> bool:
    """True if HDU exists and contains a 2D image with non-zero dimensions."""
    try:
        with fits.open(path) as hdul:
            if hdu_index < 0 or hdu_index >= len(hdul):
                return False
            d = hdul[hdu_index].data
            if d is None:
                return False
            if getattr(d, "ndim", 0) != 2:
                return False
            if d.shape[0] == 0 or d.shape[1] == 0:
                return False
            return True
    except Exception:
        return False
# -----------------------------------------------------------------------------
# PSFBuilder (pipeline-facing)
# -----------------------------------------------------------------------------
class PSFBuilder:
    def __init__(
        self,
        filters,
        directorio,
        *,
        select_parts="I",
        parts=("A", "B", "C"),
        min_dist=("0.015", "0.015", "0.015"),
        norm_radii=("5,10", "10,20", "20,40"),
        width_image=("200,200", "400,400", "800,800"),
        selection_radii_arcmin=(15.0, 15.0, 15.0),
        min_dist_outer=("0.0"),
        norm_radii_outer=("5,10"),
        width_image_outer=("200,200"),
        step_min_dist_outer=0.015,
        step_norm_radii_outer=(5, 10),
        step_width_image_outer=(200, 200),
        branch_mag_min=16.0,
        branch_mag_max=18.0,
        nc_inner_args=None,
        seg_inner_args=None,
        nc_outer_args=None,
        seg_outer_args=None,
        center_sigma=5.0,
        center_max_iter=5,
    ):
        self.filters = list(filters)
        self.directorio = Path(directorio)
        self.select_parts = str(select_parts).upper()

        self.parts = list(parts)
        self.min_dist = list(min_dist)
        self.norm_radii = list(norm_radii)
        self.width_image = list(width_image)
        self.selection_radii = list(selection_radii_arcmin)
        
        self.min_dist_outer = list(min_dist_outer) if isinstance(min_dist_outer, (list, tuple)) else [min_dist_outer]
        self.norm_radii_outer = list(norm_radii_outer) if isinstance(norm_radii_outer, (list, tuple)) else [norm_radii_outer]
        self.width_image_outer = list(width_image_outer) if isinstance(width_image_outer, (list, tuple)) else [width_image_outer]

        self.step_min_dist_outer = float(step_min_dist_outer)
        self.step_norm_radii_outer = tuple(float(x) for x in step_norm_radii_outer)
        self.step_width_image_outer = tuple(float(x) for x in step_width_image_outer)

        self.branch_mag_min = float(branch_mag_min)
        self.branch_mag_max = float(branch_mag_max)

        self.center_sigma = float(center_sigma)
        self.center_max_iter = int(center_max_iter)

        self.nc_inner_args = nc_inner_args or [
            "--tilesize=20,20",
            "--outliernumngb=5",
            "--interpnumngb=1",
            "--qthresh=0.5",
            "--minnumfalse=1",
            "--rawoutput",
        ]
        self.seg_inner_args = seg_inner_args or [
            "--tilesize=20,20",
            "--snminarea=2",
            "--interpnumngb=1",
            "--gthresh=-10",
            "--objbordersn=0",
            "--minnumfalse=1",
        ]

        # Outer: si no se pasa nada, reutiliza los de inner
        self.nc_outer_args = nc_outer_args or self.nc_inner_args
        self.seg_outer_args = seg_outer_args or self.seg_inner_args

        # Dirs existentes
        self.dir_build = Path("./Process_data/Building_PSF")
        self.dir_gaia_match = Path("./Process_data/Make_catalogs/Data_catalog_gaia_match")
        self.seg_dir = Path("./Process_data/Make_catalogs/Mask_segment")
        self.psf_out_dir = Path("./PSF_files/Inner_parts")

        # NEW: dirs para OUTER (separados)
        self.dir_build_outer = self.dir_build / "Outer_parts"
        self.psf_out_dir_outer = Path("./PSF_files/Outer_parts")

        _ensure_dir(self.dir_build)
        _ensure_dir(self.psf_out_dir)
        _ensure_dir(self.dir_build_outer)
        _ensure_dir(self.psf_out_dir_outer)

    @staticmethod
    def nearest_position(x: float, x_array: np.ndarray) -> int:
        posicion_mas_cercana = -1
        diferencia_minima = float("inf")
        for indice, valor in enumerate(x_array):
            diferencia = abs(valor - x)
            if diferencia < diferencia_minima:
                diferencia_minima = diferencia
                posicion_mas_cercana = indice
        return posicion_mas_cercana



    
    def masks_maker_inner(self, stamp_path: Path):
        name = str(stamp_path)
        output = name.replace(".fits", "_noisechisel.fits")
        output2 = name.replace(".fits", "_tmp.fits")
        ruta_segment = name.replace(".fits", "_segment.fits")

        try:
            # ---- astarithmetic (silencioso) ----
            subprocess.run(
                [
                    "astarithmetic",
                    name,
                    name,
                    "--output=" + output2,
                    "0.0",
                    "eq",
                    "nan",
                    "where",
                    "-g1",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # ---- astnoisechisel (silencioso) ----
            cmd_nc = ["astnoisechisel", output2, *self.nc_inner_args, "--output=" + output]
            subprocess.run(
                cmd_nc,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # ---- astsegment (silencioso) ----
            cmd_seg = ["astsegment", output, *self.seg_inner_args, "--output=" + ruta_segment]
            subprocess.run(
                cmd_seg,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # ---- FITS processing (ya silencioso por defecto) ----
            with fits.open(ruta_segment) as hdul:
                objects = hdul[3].data

            n = int(len(objects) / 2)
            number = stats.mode(
                objects[n - 10 : n + 10, n - 10 : n + 10],
                axis=None,
                nan_policy="omit",
            ).mode
            number_2 = objects[n, n]

            objects[objects == number] = 0
            objects[objects == number_2] = 0
            objects[objects != 0] = 1

            imagen = fits.getdata(name)
            imagen[objects != 0] = np.nan
            fits.PrimaryHDU(imagen).writeto(name, overwrite=True)

            # ---- limpieza silenciosa ----
            for f in (output, output2, ruta_segment):
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass

            return imagen, objects

        except Exception:
            # Limpieza sin mensajes
            for f in (output, output2, ruta_segment):
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
            return 0, 0

    def masks_maker_outer(self, stamp_path: Path):
        return self.masks_maker_inner(stamp_path)
    
    def _save_diag_plot(
        self,
        outpath: Path,
        *,
        hr_gaia: np.ndarray,
        mag_gaia: np.ndarray,
        hr_gaia_select: np.ndarray,
        mag_gaia_select: np.ndarray,
        hr_branch: np.ndarray,
        mag_branch: np.ndarray,
        hr_final_selected: np.ndarray,
        mag_final_selected: np.ndarray,
        hr_prueba: np.ndarray,
        mag_prueba: np.ndarray,
        rejected_mask: np.ndarray,
        min_value: float,
        log_hr_branch_mean: float,
        lim_min: float,
        lim_max: float,
        title: str,
    ) -> None:
        _ensure_dir(outpath.parent)
        fig, ax = plt.subplots()

        ax.plot(np.log10(hr_gaia), mag_gaia, "ro", markerfacecolor="none", label="Gaia detections")
        ax.plot(np.log10(hr_gaia_select), mag_gaia_select, "gs", markersize=3, label="Gaia good parallax")
        ax.plot(np.log10(hr_branch), mag_branch, "b.", markersize=3, label="Gaia branch")
        ax.plot(np.log10(hr_final_selected), mag_final_selected, "ks", markersize=5, label="Selected stars")

        if rejected_mask.size == hr_prueba.size:
            ax.plot(np.log10(hr_prueba[rejected_mask]), mag_prueba[rejected_mask], "cs", markersize=5, label="Rejected stars")

        ax.set_xlabel("log(Half sum radius [px])", fontsize=18)
        ax.set_ylabel("Magnitude in Gaia G-band", fontsize=18)

        ax.axhline(min_value)
        ax.axvline(log_hr_branch_mean, lw=2, color="k", ls="--")
        ax.axvline(lim_min, color="k", lw=2, ls="--")
        ax.axvline(lim_max, color="k", lw=2, ls="--")
        ax.axvspan(lim_min, lim_max, alpha=0.5, color="gray")

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(direction="in", which="minor", length=4, color="k")

        ax.set_title(title)
        plt.legend(loc="best")
        plt.xlim(-0.1, 1.5)
        plt.ylim(10, 25)
        plt.savefig(outpath, format="jpg", bbox_inches="tight", pad_inches=0.05, dpi=400)
        plt.close(fig)

    def _stack_dir(self, stars_dir: Path, out_dir: Path, *, tag: str, rmax: int = 2500) -> bool:
        """
        Build stack + radial profile.
        Returns True only if the final stack exists and has image data.
        Tries input HDU=0, then HDU=1 (common case when primary HDU is empty).
        """
        stars = sorted(stars_dir.glob("*.fits"))
        conti = len(stars)
        if conti == 0:
            return False

        _ensure_dir(out_dir)

        stack_path = out_dir / f"stack_{tag}.fits"
        prof_path = out_dir / f"profile_psf_{tag}.fits"

        # Clean old outputs to avoid false positives
        for p in (stack_path, prof_path):
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

        # Try stacking reading stars from HDU 0, then 1
        stack_ok = False
        stack_hdu_used: int | None = None
        
        for in_hdu in (0, 1):
            try:
                # astarithmetic with explicit input HDU
                os.system(f"astarithmetic {stars_dir}/*.fits {str(conti)} \
                          2 0.2 sigclip-mean -g{in_hdu} --output={stack_path}\
                              --wcsfile=none --writeall > /dev/null 2>&1")

                # Verify stack really contains data (try both HDUs in the output)
                if fits_hdu_has_data(stack_path, 0) or fits_hdu_has_data(stack_path, 1):
                    stack_ok = True
                    stack_hdu_used = 0 if fits_hdu_has_data(stack_path, 0) else 1
                    break

            except Exception as e:
                # Try next HDU
                print(f"[warning] stacking failed for HDU={in_hdu} with error: {e}. Trying next HDU if available.")
                continue

        if not stack_ok:
            # ensure no stale file remains
            if stack_path.exists():
                try:
                    stack_path.unlink()
                except Exception as e:
                    print(f"[warning] failed to remove incomplete stack at {stack_path}: {e}")
                    pass
            return False
        
        # Radial profile: try with HDU=1 then HDU=0 (some stacks land in primary)
        prof_ok = False
        for prof_hdu in (1, 0):
            try:
                os.system(
                    f"astscript-radial-profile {stack_path} "
                    f"--quiet --hdu={prof_hdu} "
                    f"--measure=mean,std,area,semi-major "
                    f"--rmax={rmax} -o {prof_path} "
                    f"> /dev/null 2>&1"
                )
                if prof_path.exists():
                    prof_ok = True
                    break
            except Exception as e:
                print(f"[warning] radial profile failed for HDU={prof_hdu} with error: {e}. Trying next HDU if available.")
                continue

        # Check stack and profile existence
        return stack_path.exists() and (fits_hdu_has_data(stack_path, 0) or fits_hdu_has_data(stack_path, 1)) and prof_ok

    def build(self) -> None:
        do_inner = self.select_parts in ("I", "B")
        do_outer = self.select_parts in ("O", "B")

        for flt in self.filters:
            print("\n" + 40 * "=")
            print(f"\n Filter: {flt}")
            print("\n" + 40 * "=")

            fits_files = sorted(
                [p.name for p in self.directorio.iterdir() if p.is_file() and re.search(flt + r".*\.fits$", p.name)]
            )
            if not fits_files:
                continue

            for name in fits_files:
                name_gal = name.split("_")[0]
                print("\n" + 40 * "=")
                print(f"\n Galaxy: {name_gal}")
                print("\n" + 40 * "=")

                ruta_completa = self.directorio / name

                ruta_gaia_match = self.dir_gaia_match / f"match_gaia_{name_gal}_{flt}.fits"
                if not ruta_gaia_match.exists():
                    print(f"[skip] missing Gaia match: {ruta_gaia_match}")
                    continue

                ruta_seg_cut = self.seg_dir / f"seg-{name}"
                if not ruta_seg_cut.exists():
                    print(f"[skip] missing seg cutout: {ruta_seg_cut}")
                    continue

                # Load Gaia-match (faithful)
                hdul_gaia = fits.open(ruta_gaia_match)
                data_gaia = hdul_gaia[1].data
                mag_gaia = hdul_gaia[2].data["phot_g_mean_mag"]
                hr_gaia = data_gaia["HALF_SUM_RADIUS"]
                parallax_data = hdul_gaia[2].data["parallax"]
                parallax_error = hdul_gaia[2].data["parallax_error"]
                pm_ra_data = hdul_gaia[2].data["pmra"]
                pm_dec_data = hdul_gaia[2].data["pmdec"]
                pm_ra_error = hdul_gaia[2].data["pmra_error"]
                pm_dec_error = hdul_gaia[2].data["pmdec_error"]
                ra_gaia = data_gaia["RA"]
                dec_gaia = data_gaia["DEC"]
                hdul_gaia.close()

                # Gaia filter (faithful)
                bol_par = parallax_data >= 3 * parallax_error
                condicion_pmra_pmdec = (
                    (~np.isnan(pm_ra_data) & (abs(pm_ra_data) >= 3 * (pm_ra_error)))
                    | (~np.isnan(pm_dec_data) & (abs(pm_dec_data) >= 3 * (pm_dec_error)))
                )
                array_seleccion = bol_par | condicion_pmra_pmdec

                mag_gaia_select = mag_gaia[array_seleccion]
                hr_gaia_select = hr_gaia[array_seleccion]
                if mag_gaia_select.size == 0:
                    print("[skip] no Gaia-good sources after parallax/PM cut")
                    continue

                # Branch localization (faithful)
                bol_mag_gaia_min = mag_gaia_select > self.branch_mag_min
                bol_mag_gaia_max = mag_gaia_select < self.branch_mag_max
                bol_mag_gaia = np.invert(bol_mag_gaia_max ^ bol_mag_gaia_min)

                clipped_log_hr_branch = sigma_clip(np.log10(hr_gaia_select[bol_mag_gaia]), 2)
                clipped_log_hr_branch_values = clipped_log_hr_branch.data[np.invert(clipped_log_hr_branch.mask)]
                if clipped_log_hr_branch_values.size == 0:
                    print("[skip] empty branch mag-window after sigma_clip")
                    continue

                lim_min = np.mean(clipped_log_hr_branch_values) - 2 * np.std(clipped_log_hr_branch_values)
                lim_max = np.mean(clipped_log_hr_branch_values) + 2 * np.std(clipped_log_hr_branch_values)

                bol_min = np.log10(hr_gaia_select) > lim_min
                bol_max = np.log10(hr_gaia_select) < lim_max
                bol = bol_max ^ bol_min
                bol_selected = np.invert(bol)

                hr_branch = hr_gaia_select[bol_selected]
                mag_branch = mag_gaia_select[bol_selected]
                if mag_branch.size == 0:
                    print("[skip] empty branch after HR limits")
                    continue

                # min_value (faithful)
                Q1 = np.nanpercentile(mag_branch, 15)
                Q3 = np.nanpercentile(mag_branch, 55)
                IQR = Q3 - Q1
                lower_bound = Q1 - IQR
                upper_bound = Q3 + IQR
                filtered_data = mag_branch[(mag_branch >= lower_bound) & (mag_branch <= upper_bound)]
                if filtered_data.size == 0:
                    print("[skip] empty filtered_data for min_value")
                    continue

                min_value = np.round(np.nanmin(filtered_data), 1)

                upper_bounds = np.linspace(min_value, np.round(Q3) - 1, len(self.parts))
                lower_bounds = upper_bounds + 1
                upper_bounds = np.flip(upper_bounds)
                lower_bounds = np.flip(lower_bounds)

                # Open science cutout for recentering / WCS
                try:
                    hdu_list = fits.open(ruta_completa)
                    image_data = hdu_list[1].data
                    wcs = WCS(hdu_list[1].header)
                    hdu_list.close()
                except Exception as e:
                    print(f"[skip] could not open science cutout for WCS/image: {e}")
                    continue

                # =======================================================
                # =========================== INNER =====================
                # =======================================================
                if do_inner:
                    # Build each inner part
                    for i, part in enumerate(self.parts):
                        mag_sup_lim = str(np.round(lower_bounds[i], 1))
                        mag_inf_lim = str(np.round(upper_bounds[i], 1))

                        min_dist = self.min_dist[i]
                        norm_radii = self.norm_radii[i]
                        width_image = self.width_image[i]
                        selec_radii = float(self.selection_radii[i])

                        # 1) select stars
                        print(f"\n ▸ Building part {part} with mag limits {mag_inf_lim} - {mag_sup_lim} and min_dist {min_dist}")
                        dir_stars = _ensure_dir(self.dir_build / f"{part}_stars")
                        ruta_star = dir_stars / name.replace(".fits", f"_{part}_stars.fits")

                        os.system(
                            f"astscript-psf-select-stars {ruta_completa} "
                            f"--magnituderange={mag_inf_lim},{mag_sup_lim} "
                            f"--mindistdeg={min_dist} --output={ruta_star} "
                            f"> /dev/null 2>&1"
                        )

                        if not ruta_star.exists():
                            print(f"[skip] no star catalog produced for {name_gal} {flt} {part}")
                            continue

                        # 2) mix with Gaia via nearest_position (faithful)
                        print(f"\n ▸ Mixing with Gaia and applying HR/Mag cuts for {name_gal} {flt} {part}")
                        try:
                            hr_prueba = []
                            mag_prueba = []
                            ra_prueba = []
                            dec_prueba = []

                            hprueba = fits.open(ruta_star)
                            tab = hprueba[1].data

                            for num in range(0, len(tab)):
                                ra = float(tab["ra"][num])
                                dec = float(tab["dec"][num])
                                mag = float(tab["phot_g_mean_mag"][num])

                                pos_ra = self.nearest_position(ra, ra_gaia)
                                pos_dec = self.nearest_position(dec, dec_gaia)

                                if pos_ra == pos_dec:
                                    ra_prueba.append(float(ra_gaia[pos_ra]))
                                    dec_prueba.append(float(dec_gaia[pos_ra]))
                                    hr_prueba.append(float(hr_gaia[pos_ra]))
                                    mag_prueba.append(float(mag_gaia[pos_ra]))
                                else:
                                    ra_prueba.append(0.0)
                                    dec_prueba.append(0.0)
                                    hr_prueba.append(0.0)
                                    mag_prueba.append(0.0)

                            hr_prueba = np.array(hr_prueba)
                            mag_prueba = np.array(mag_prueba)
                            ra_prueba = np.array(ra_prueba)
                            dec_prueba = np.array(dec_prueba)

                            lim_mags_inf = mag_prueba > float(mag_inf_lim)
                            lim_mags_upper = mag_prueba < float(mag_sup_lim)
                            lim_mags = np.invert(lim_mags_inf ^ lim_mags_upper)

                            hr_prueba = hr_prueba[lim_mags]
                            mag_prueba = mag_prueba[lim_mags]
                            ra_prueba = ra_prueba[lim_mags]
                            dec_prueba = dec_prueba[lim_mags]

                            bol_min2 = np.log10(hr_prueba) > lim_min
                            bol_max2 = np.log10(hr_prueba) < lim_max
                            bol2 = bol_max2 ^ bol_min2
                            bol_selected2 = np.invert(bol2)

                            mag_final_selected = mag_prueba[bol_selected2]
                            hr_final_selected = hr_prueba[bol_selected2]
                            ra_final_selected = ra_prueba[bol_selected2]
                            dec_final_selected = dec_prueba[bol_selected2]

                            # angular cut (faithful)
                            ra_gal, dec_gal = obtain_coords(name_gal)
                            galaxy_coord = SkyCoord(ra=ra_gal * u.degree, dec=dec_gal * u.degree)
                            stars_coord = SkyCoord(ra=ra_final_selected * u.degree, dec=dec_final_selected * u.degree)
                            angular_distance = stars_coord.separation(galaxy_coord)
                            radius = selec_radii * u.arcminute
                            mask = angular_distance <= radius

                            ra_final_selected = ra_final_selected[mask]
                            dec_final_selected = dec_final_selected[mask]
                            mag_final_selected = mag_final_selected[mask]
                            hr_final_selected = hr_final_selected[mask]

                            # Save RA/DEC/MAG catalog (faithful)
                            data_combined = np.zeros(
                                len(mag_final_selected),
                                dtype=[("ra", ">f8"), ("dec", ">f8"), ("phot_g_mean_mag", ">f8")],
                            )
                            data_combined["ra"] = ra_final_selected
                            data_combined["dec"] = dec_final_selected
                            data_combined["phot_g_mean_mag"] = mag_final_selected
                            hprueba[1].header["NAXIS2"] = int(len(mag_final_selected))

                            new_hdul = fits.HDUList(
                                [
                                    fits.PrimaryHDU(header=hprueba[0].header),
                                    fits.BinTableHDU(data_combined, header=hprueba[1].header),
                                ]
                            )
                            hprueba.close()
                            new_hdul.writeto(ruta_star, overwrite=True)

                            # Diagnostic plot (same content as original)
                            diag_dir = _ensure_dir(self.dir_build / "Diagnostics")
                            diag_path = diag_dir / f"{name_gal}_{flt}_{part}_mag_hr.jpg"
                            self._save_diag_plot(
                                diag_path,
                                hr_gaia=hr_gaia,
                                mag_gaia=mag_gaia,
                                hr_gaia_select=hr_gaia_select,
                                mag_gaia_select=mag_gaia_select,
                                hr_branch=hr_branch,
                                mag_branch=mag_branch,
                                hr_final_selected=hr_final_selected,
                                mag_final_selected=mag_final_selected,
                                hr_prueba=hr_prueba,
                                mag_prueba=mag_prueba,
                                rejected_mask=~bol_selected2,
                                min_value=float(min_value),
                                log_hr_branch_mean=float(np.mean(clipped_log_hr_branch_values)),
                                lim_min=float(lim_min),
                                lim_max=float(lim_max),
                                title=f"{name_gal} Filter {flt}",
                            )

                            if len(mag_final_selected) == 0:
                                print(f"[skip] no stars after filters for {name_gal} {flt} {part}")
                                continue

                        except Exception as e:
                            print(f"[skip] failure in Gaia/gnuastro mixing stage: {e}")
                            try:
                                hprueba.close()
                            except Exception:
                                pass
                            continue

                        # 3) recenter (faithful) + sigma configurable
                        print(f"\n ▸ Recentering stars for {name_gal} {flt} {part} with sigma={self.center_sigma} and max_iter={self.center_max_iter}")
                        try:
                            h_center = fits.open(ruta_star)
                            ra = h_center[1].data["ra"]
                            dec = h_center[1].data["dec"]
                            h_center.close()

                            pixel_coords = wcs.world_to_pixel_values([r for r in ra], [d for d in dec])
                            pixel_coords = np.array(pixel_coords).T

                            y_new = []
                            x_new = []
                            for coord in tqdm(pixel_coords, desc=f"{name_gal} {part} centers", leave=False):
                                x, y = int(coord[0]), int(coord[1])

                                try:
                                    new_center = graph_interactive_center(
                                        image_data,
                                        (x, y),
                                        False,
                                        10,
                                        sigma=self.center_sigma,
                                        max_iter=self.center_max_iter,
                                    )
                                    y_new.append(float(new_center[0]))
                                    x_new.append(float(new_center[1]))
                                except Exception:
                                    y_new.append(float(y))
                                    x_new.append(float(x))

                            y_new = np.array(y_new)
                            x_new = np.array(x_new)

                            data_xy = np.zeros(
                                len(mag_final_selected),
                                dtype=[("x_px", ">f8"), ("y_px", ">f8"), ("phot_g_mean_mag", ">f8")],
                            )
                            data_xy["x_px"] = x_new
                            data_xy["y_px"] = y_new
                            data_xy["phot_g_mean_mag"] = mag_final_selected

                            fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(data_xy, name="STARS")]).writeto(
                                ruta_star, overwrite=True
                            )

                        except Exception as e:
                            print(f"[skip] center correction failed: {e}")
                            continue

                        # 4) stamps using seg-cutout as image and segment
                        #    NEW: create stamps first; then in Python, if the *image value at the stamp center* is negative,
                        #    multiply the whole stamp by -1 (without touching the center coordinates used for stamping).
                        print(f"\n ▸ Stamping and masking for {name_gal} {flt} {part}")
                        try:
                            crops_root = _ensure_dir(self.dir_build / "Star_crops" / part)
                            ruta_gal_folder = _ensure_dir(crops_root / f"{name_gal}_{flt}")

                            out_base = name.replace(".fits", "")
                            out_root = ruta_gal_folder / out_base
                            ruta_out = str(out_root)  # used later for manual deletion block (kept close to original style)

                            os.system(
                                "{ "
                                + "counter=1; "
                                + "asttable " + str(ruta_star)
                                + " | while read -r x_px y_px mag; do "
                                + "astscript-psf-stamp " + str(ruta_seg_cut) + " "
                                + "--mode=img "
                                + "--center=$x_px,$y_px "
                                + f"--normradii={norm_radii} "
                                + f"--widthinpix={width_image} "
                                + "--quiet "
                                + f"--segment={ruta_seg_cut} "
                                + f"--output={out_root}" + "_$counter.fits; "
                                + "counter=$((counter+1)); "
                                + "done; "
                                + "} > /dev/null 2>&1"
                            )


                            # ---------------------------------------------------------------------
                            # Post-process stamps in Python: if central pixel value < 0, flip stamp
                            # ---------------------------------------------------------------------
                            stamp_list = [p for p in ruta_gal_folder.iterdir() if p.is_file() and p.suffix == ".fits"]
                            for stamp in stamp_list:
                                try:
                                    with fits.open(stamp, mode="update") as hdul:
                                        # Robustly pick an image HDU
                                        if len(hdul) > 1 and isinstance(hdul[1], fits.ImageHDU) and hdul[1].data is not None:
                                            hdu = hdul[1]
                                        else:
                                            hdu = hdul[0]

                                        img = hdu.data
                                        if img is None or img.ndim != 2:
                                            continue

                                        cy = img.shape[0] // 2
                                        cx = img.shape[1] // 2
                                        vcen = img[cy, cx]

                                        # Only flip if finite and negative
                                        if np.isfinite(vcen) and (vcen < 0):
                                            hdu.data = -img
                                            hdul.flush()

                                except Exception:
                                    # If something goes wrong, don't kill the pipeline; optionally delete the stamp
                                    try:
                                        stamp.unlink()
                                    except Exception:
                                        pass

                            # ---------------------------------------------------------------------
                            # Mask stamps (same as before)
                            # ---------------------------------------------------------------------
                            stamp_list = [p for p in ruta_gal_folder.iterdir() if p.is_file() and p.suffix == ".fits"]
                            for stamp in stamp_list:
                                try:
                                    self.masks_maker_inner(stamp)
                                except Exception:
                                    try:
                                        stamp.unlink()
                                    except Exception:
                                        pass

                            # ---------------------------------------------------------------------
                            # Ensure image data lives in HDU=1 (if it's in HDU=0, move it to HDU=1)
                            # This avoids downstream tools failing with "HDU 0 has 0 dimensions".
                            # ---------------------------------------------------------------------
                            stamp_list = [p for p in ruta_gal_folder.iterdir() if p.is_file() and p.suffix == ".fits"]
                            for stamp in stamp_list:
                                try:
                                    with fits.open(stamp, mode="readonly") as hdul:
                                        # Case A: HDU 1 exists and has image -> OK
                                        if (
                                            len(hdul) > 1
                                            and isinstance(hdul[1], (fits.ImageHDU, fits.PrimaryHDU))
                                            and hdul[1].data is not None
                                            and getattr(hdul[1].data, "ndim", 0) == 2
                                        ):
                                            continue

                                        # Case B: HDU 0 has the image -> move/copy to HDU 1
                                        if (
                                            len(hdul) >= 1
                                            and isinstance(hdul[0], fits.PrimaryHDU)
                                            and hdul[0].data is not None
                                            and getattr(hdul[0].data, "ndim", 0) == 2
                                        ):
                                            img0 = np.array(hdul[0].data, copy=True)
                                            hdr0 = hdul[0].header.copy()

                                        else:
                                            # Nothing usable
                                            continue

                                    # Rewrite file: PrimaryHDU with header only (no data), ImageHDU with the data in ext=1
                                    # Keep original primary header keywords as much as possible.
                                    phdu = fits.PrimaryHDU(header=hdr0)
                                    ihdu = fits.ImageHDU(data=img0, name="IMAGE")
                                    new_hdul = fits.HDUList([phdu, ihdu])
                                    new_hdul.writeto(stamp, overwrite=True)

                                except Exception:
                                    try:
                                        stamp.unlink()
                                    except Exception:
                                        pass

                            # ---------------------------------------------------------------------
                            # Manual visual inspection + deletion (as requested; DS9 via astscript-fits-view)
                            # ---------------------------------------------------------------------
                            print("\n ▸ Number of stars selected: ")
                            os.system("echo " + str(ruta_gal_folder) + "/*" + flt + "*.fits | wc -w")
                            pause = print("\n ▸ This is just a pause to perform a visual inspection of the stars")

                            ds9_process = subprocess.Popen(
                                ["astscript-fits-view", str(ruta_gal_folder) + "/*" + flt + "*.fits"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                            delete = input("\n ▸ Select the stars to delete if any: ")
                            subprocess.run(["pkill", "-f", "ds9"])

                            if len(delete) != 0:
                                delete_list = np.array([int(num) for num in delete.split(",")])
                                for ii in delete_list:
                                    if ii == 0:
                                        ii = ""
                                    stack = ruta_out + "_" + str(ii) + ".fits"
                                    os.system("rm " + stack)

                        except Exception as e:
                            print(f"[skip] stamping/masking failed: {e}")
                            continue

                        # 5) stack per galaxy + part
                        print(f"\n ▸ Stacking for {name_gal} {flt} {part}")
                        
                        out_psf = _ensure_dir(self.psf_out_dir / name_gal / f"{name_gal}_{part}_{flt}")
                        tag = f"{part}_{flt}_{name_gal}_{mag_inf_lim}_{mag_sup_lim}"

                        ok = False
                        try:
                            ok = self._stack_dir(ruta_gal_folder, out_psf, tag=tag)
                        except Exception:
                            ok = False

                        stack_path = out_psf / f"stack_{tag}.fits"
                        prof_path = out_psf / f"profile_psf_{tag}.fits"

                        if ok and stack_path.exists():
                            print(f"[ok] {name_gal} {flt} {part} stacked")
                        else:
                            # Si quieres, aquí puedes reportar qué faltó sin volcar el error completo:
                            missing = []
                            if not stack_path.exists():
                                missing.append("stack")
                            if not prof_path.exists():
                                missing.append("profile")
                            missing_str = ",".join(missing) if missing else "unknown"
                            print(f"[skip] stacking failed ({missing_str})")
                            continue

                # ==============================================================
                # ============================== OUTER =========================
                # ==============================================================
                if do_outer:

                    min_gaia = float(np.nanmin(mag_gaia_select))
                    min_val  = float(min_value)

                    if not np.isfinite(min_gaia) or not np.isfinite(min_val) or (min_val <= min_gaia):
                        print(f"[skip] cannot build outer parts: min_val={min_val} min_gaia={min_gaia}")
                        break

                    # bins de 1 mag de ancho (máximo)
                    start_edge = min_val
                    n_bins = int(np.ceil(start_edge - min_gaia))
                    if n_bins < 1:
                        print(f"[skip] not enough mag range to define outer bins: min_val={min_val} min_gaia={min_gaia}")
                        break

                    edges = start_edge - np.arange(0, n_bins + 1, dtype=float)
                    mag_sup_arr = edges[:-1]  # 13.7, 12.7, ...
                    mag_inf_arr = edges[1:]   # 12.7, 11.7, ...

                    upper_bounds_outer = mag_inf_arr  # mag_inf_lim
                    lower_bounds_outer = mag_sup_arr  # mag_sup_lim
                    outer_parts = [f"Outer_{k}" for k in range(1, len(upper_bounds_outer) + 1)]
                    
                    for i, part_outer in enumerate(outer_parts):
                        mag_sup_lim = str(np.round(lower_bounds_outer[i], 1))
                        mag_inf_lim = str(np.round(upper_bounds_outer[i], 1))

                        # ----------------------------------------------------------
                        # OUTER params (NEW):
                        #   - base values from *_outer lists (index saturado)
                        #   - then apply step_* multiplied by i (outer bin index)
                        # ----------------------------------------------------------
                        j0 = min(i, len(self.min_dist_outer) - 1)
                        min_dist0 = float(self.min_dist_outer[j0])

                        j1 = min(i, len(self.norm_radii_outer) - 1)
                        nr_in0, nr_out0 = (float(x) for x in str(self.norm_radii_outer[j1]).split(","))

                        j2 = min(i, len(self.width_image_outer) - 1)
                        w_x0, w_y0 = (float(x) for x in str(self.width_image_outer[j2]).split(","))

                        # apply steps
                        min_dist = f"{(min_dist0 + i * self.step_min_dist_outer):.6f}"

                        nr_in  = nr_in0  + i * self.step_norm_radii_outer[0]
                        nr_out = nr_out0 + i * self.step_norm_radii_outer[1]
                        norm_radii = f"{nr_in:.2f},{nr_out:.2f}".replace(".00", "")

                        w_x = w_x0 + i * self.step_width_image_outer[0]
                        w_y = w_y0 + i * self.step_width_image_outer[1]
                        width_image = f"{int(round(w_x))},{int(round(w_y))}"

                        # selection radius: puedes reutilizar el último/único, o mantener el de inner
                        # aquí uso el último de inner (o el único si solo hay 1)
                        selec_radii = float(self.selection_radii[min(i, len(self.selection_radii) - 1)])

                        print(
                            f"\n ▸ [OUTER] Building {part_outer} with mag limits {mag_inf_lim} - {mag_sup_lim} "
                            f"and min_dist {min_dist}, norm_radii {norm_radii}, width {width_image}"
                        )

                        # 1) select stars (outer)
                        dir_stars_outer = _ensure_dir(self.dir_build_outer / f"{part_outer}_stars")
                        ruta_star = dir_stars_outer / name.replace(".fits", f"_{part_outer}_stars.fits")

                        os.system(
                            f"astscript-psf-select-stars {ruta_completa} "
                            f"--magnituderange={mag_inf_lim},{mag_sup_lim} "
                            f"--mindistdeg={min_dist} --output={ruta_star} "
                            f"> /dev/null 2>&1"
                        )
                        if not ruta_star.exists():
                            print(f"[skip] [OUTER] no star catalog produced for {name_gal} {flt} {part_outer}")
                            continue

                        # 2) mix con Gaia (idéntico a INNER hasta ra/dec/mag finales)
                        print(f"\n ▸ [OUTER] Mixing with Gaia and applying Mag/Angular cuts for {name_gal} {flt} {part_outer}")
                        try:
                            hr_prueba = []
                            mag_prueba = []
                            ra_prueba = []
                            dec_prueba = []

                            hprueba = fits.open(ruta_star)
                            tab = hprueba[1].data
                            
                            for num in range(0, len(tab)):
                                ra = float(tab["ra"][num])
                                dec = float(tab["dec"][num])
                                mag = float(tab["phot_g_mean_mag"][num])

                                pos_ra = self.nearest_position(ra, ra_gaia)
                                pos_dec = self.nearest_position(dec, dec_gaia)
                                _pos_mag = self.nearest_position(mag, mag_gaia)

                                if pos_ra == pos_dec:
                                    ra_prueba.append(float(ra_gaia[pos_ra]))
                                    dec_prueba.append(float(dec_gaia[pos_ra]))
                                    hr_prueba.append(float(hr_gaia[pos_ra]))
                                    mag_prueba.append(float(mag_gaia[pos_ra]))
                                else:
                                    ra_prueba.append(0.0)
                                    dec_prueba.append(0.0)
                                    hr_prueba.append(0.0)
                                    mag_prueba.append(0.0)

                            hr_prueba = np.array(hr_prueba, dtype=float)
                            mag_prueba = np.array(mag_prueba, dtype=float)
                            ra_prueba = np.array(ra_prueba, dtype=float)
                            dec_prueba = np.array(dec_prueba, dtype=float)

                            # --- corte en magnitud del bin OUTER ---
                            lim_mags_inf = mag_prueba > float(mag_inf_lim)
                            lim_mags_upper = mag_prueba < float(mag_sup_lim)
                            lim_mags = np.invert(lim_mags_inf ^ lim_mags_upper)
                            
                            hr_prueba = hr_prueba[lim_mags]
                            mag_prueba = mag_prueba[lim_mags]
                            ra_prueba = ra_prueba[lim_mags]
                            dec_prueba = dec_prueba[lim_mags]
                            
                            # --- OUTER: NO HR 2-sigma filtering ---
                            # (mantén todo lo que pase el corte en magnitud)
                            mag_final_selected = mag_prueba
                            hr_final_selected = hr_prueba
                            ra_final_selected = ra_prueba
                            dec_final_selected = dec_prueba
                            bol_selected2 = np.full_like(mag_final_selected, True, dtype=bool)  # keep all after mag cut

                            # --- corte angular ---
                            ra_gal, dec_gal = obtain_coords(name_gal)
                            galaxy_coord = SkyCoord(ra=ra_gal * u.degree, dec=dec_gal * u.degree)
                            stars_coord = SkyCoord(ra=ra_final_selected * u.degree, dec=dec_final_selected * u.degree)
                            angular_distance = stars_coord.separation(galaxy_coord)
                            radius = selec_radii * u.arcminute
                            mask = angular_distance <= radius

                            ra_final_selected = ra_final_selected[mask]
                            dec_final_selected = dec_final_selected[mask]
                            mag_final_selected = mag_final_selected[mask]
                            hr_final_selected = hr_final_selected[mask]

                            if len(mag_final_selected) == 0:
                                print(f"[skip] [OUTER] no stars after Mag/Angular cuts for {name_gal} {flt} {part_outer}")
                                hprueba.close()
                                #Diagnostic plot (same content as original)
                                diag_dir = _ensure_dir(self.dir_build_outer / "Diagnostics")
                                diag_path = diag_dir / f"{name_gal}_{flt}_{part_outer}_mag_hr.jpg"
                                self._save_diag_plot(
                                    diag_path,
                                    hr_gaia=hr_gaia,
                                    mag_gaia=mag_gaia,
                                    hr_gaia_select=hr_gaia_select,
                                    mag_gaia_select=mag_gaia_select,
                                    hr_branch=hr_branch,
                                    mag_branch=mag_branch,
                                    hr_final_selected=hr_final_selected,
                                    mag_final_selected=mag_final_selected,
                                    hr_prueba=hr_prueba,
                                    mag_prueba=mag_prueba,
                                    rejected_mask=~bol_selected2,
                                    min_value=float(min_value),
                                    log_hr_branch_mean=float(np.mean(clipped_log_hr_branch_values)),
                                    lim_min=float(lim_min),
                                    lim_max=float(lim_max),
                                    title=f"{name_gal} Filter {flt} {part_outer}",
                                )
                                continue

                            hprueba.close()
                            
                            #Diagnostic plot (same content as original)
                            diag_dir = _ensure_dir(self.dir_build_outer / "Diagnostics")
                            diag_path = diag_dir / f"{name_gal}_{flt}_{part_outer}_mag_hr.jpg"
                            self._save_diag_plot(
                                diag_path,
                                hr_gaia=hr_gaia,
                                mag_gaia=mag_gaia,
                                hr_gaia_select=hr_gaia_select,
                                mag_gaia_select=mag_gaia_select,
                                hr_branch=hr_branch,
                                mag_branch=mag_branch,
                                hr_final_selected=hr_final_selected,
                                mag_final_selected=mag_final_selected,
                                hr_prueba=hr_prueba,
                                mag_prueba=mag_prueba,
                                rejected_mask=~bol_selected2,
                                min_value=float(min_value),
                                log_hr_branch_mean=float(np.mean(clipped_log_hr_branch_values)),
                                lim_min=float(lim_min),
                                lim_max=float(lim_max),
                                title=f"{name_gal} Filter {flt} {part_outer}",
                            )
                            # 3) OUTER: NO recenter. Gaia RA/DEC -> pix directamente
                            xpix, ypix = wcs.world_to_pixel_values(
                                ra_final_selected.astype(float),
                                dec_final_selected.astype(float),
                            )
                            xpix = np.array(xpix, dtype=float)
                            ypix = np.array(ypix, dtype=float)

                            data_xy = np.zeros(
                                len(mag_final_selected),
                                dtype=[("x_px", ">f8"), ("y_px", ">f8"), ("phot_g_mean_mag", ">f8")],
                            )
                            data_xy["x_px"] = xpix
                            data_xy["y_px"] = ypix
                            data_xy["phot_g_mean_mag"] = mag_final_selected

                            fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(data_xy, name="STARS")]).writeto(
                                ruta_star, overwrite=True
                            )
                            hprueba.close()

                        except Exception as e:
                            print(f"[skip] [OUTER] failure in Gaia/gnuastro mixing stage: {e}")
                            try:
                                hprueba.close()
                            except Exception:
                                pass
                            continue

                        # 4) stamping/masking OUTER
                        print(f"\n ▸ [OUTER] Stamping and masking for {name_gal} {flt} {part_outer}")
                        try:
                            crops_root = _ensure_dir(self.dir_build_outer / "Star_crops" / part_outer)
                            ruta_gal_folder = _ensure_dir(crops_root / f"{name_gal}_{flt}")

                            out_base = name.replace(".fits", "")
                            out_root = ruta_gal_folder / out_base
                            ruta_out = str(out_root)

                            os.system(
                                "{ "
                                + "counter=1; "
                                + "asttable " + str(ruta_star)
                                + " | while read -r x_px y_px mag; do "
                                + "astscript-psf-stamp " + str(ruta_seg_cut) + " "
                                + "--mode=img "
                                + "--center=$x_px,$y_px "
                                + f"--normradii={norm_radii} "
                                + f"--widthinpix={width_image} "
                                + "--quiet "
                                + f"--segment={ruta_seg_cut} "
                                + f"--output={out_root}" + "_$counter.fits; "
                                + "counter=$((counter+1)); "
                                + "done; "
                                + "} > /dev/null 2>&1"
                            )

                            # flip si pixel central negativo
                            stamp_list = [p for p in ruta_gal_folder.iterdir() if p.is_file() and p.suffix == ".fits"]
                            for stamp in stamp_list:
                                try:
                                    with fits.open(stamp, mode="update") as hdul:
                                        if len(hdul) > 1 and isinstance(hdul[1], fits.ImageHDU) and hdul[1].data is not None:
                                            hdu = hdul[1]
                                        else:
                                            hdu = hdul[0]
                                        img = hdu.data
                                        if img is None or img.ndim != 2:
                                            continue
                                        cy = img.shape[0] // 2
                                        cx = img.shape[1] // 2
                                        vcen = img[cy, cx]
                                        if np.isfinite(vcen) and (vcen < 0):
                                            hdu.data = -img
                                            hdul.flush()
                                except Exception:
                                    try:
                                        stamp.unlink()
                                    except Exception:
                                        pass

                            # masks (outer)
                            stamp_list = [p for p in ruta_gal_folder.iterdir() if p.is_file() and p.suffix == ".fits"]
                            for stamp in stamp_list:
                                try:
                                    self.masks_maker_outer(stamp)
                                except Exception:
                                    try:
                                        stamp.unlink()
                                    except Exception:
                                        pass

                            # enforce data in HDU=1
                            stamp_list = [p for p in ruta_gal_folder.iterdir() if p.is_file() and p.suffix == ".fits"]
                            for stamp in stamp_list:
                                try:
                                    with fits.open(stamp, mode="readonly") as hdul:
                                        if len(hdul) > 1 and hdul[1].data is not None and getattr(hdul[1].data, "ndim", 0) == 2:
                                            continue
                                        if len(hdul) >= 1 and hdul[0].data is not None and getattr(hdul[0].data, "ndim", 0) == 2:
                                            img0 = np.array(hdul[0].data, copy=True)
                                            hdr0 = hdul[0].header.copy()
                                        else:
                                            continue
                                    phdu = fits.PrimaryHDU(header=hdr0)
                                    ihdu = fits.ImageHDU(data=img0, name="IMAGE")
                                    fits.HDUList([phdu, ihdu]).writeto(stamp, overwrite=True)
                                except Exception:
                                    try:
                                        stamp.unlink()
                                    except Exception:
                                        pass

                            # ---------------------------------------------------------------------
                            # Manual visual inspection + deletion (OUTER)  [NEW: integrado aquí]
                            # ---------------------------------------------------------------------
                            print("\n ▸ Number of stars selected: ")
                            os.system("echo " + str(ruta_gal_folder) + "/*" + flt + "*.fits | wc -w")
                            pause = print("\n ▸ This is just a pause to perform a visual inspection of the stars")

                            ds9_process = subprocess.Popen(
                                ["astscript-fits-view", str(ruta_gal_folder) + "/*" + flt + "*.fits"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                            delete = input("\n ▸ Select the stars to delete if any: ")
                            subprocess.run(["pkill", "-f", "ds9"])

                            if len(delete) != 0:
                                delete_list = np.array([int(num) for num in delete.split(",")])
                                for ii in delete_list:
                                    if ii == 0:
                                        ii = ""
                                    stack = ruta_out + "_" + str(ii) + ".fits"
                                    os.system("rm " + stack)

                        except Exception as e:
                            print(f"[skip] [OUTER] stamping/masking failed: {e}")
                            continue

                        # 5) stack OUTER
                        print(f"\n ▸ [OUTER] Stacking for {name_gal} {flt} {part_outer}")
                        out_psf = _ensure_dir(self.psf_out_dir_outer / name_gal / f"{name_gal}_{part_outer}_{flt}")
                        tag = f"{part_outer}_{flt}_{name_gal}_{mag_inf_lim}_{mag_sup_lim}"

                        ok = False
                        try:
                            ok = self._stack_dir(ruta_gal_folder, out_psf, tag=tag)
                        except Exception:
                            ok = False

                        stack_path = out_psf / f"stack_{tag}.fits"
                        prof_path = out_psf / f"profile_psf_{tag}.fits"

                        if ok and stack_path.exists() and prof_path.exists():
                            print(f"[ok] [OUTER] {name_gal} {flt} {part_outer} stacked")
                        else:
                            missing = []
                            if not stack_path.exists():
                                missing.append("stack")
                            if not prof_path.exists():
                                missing.append("profile")
                            print(f"[skip] [OUTER] stacking failed ({','.join(missing) if missing else 'unknown'})")
                            continue