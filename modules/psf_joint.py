#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psf_joint.py

Interactive tool to:
1) Load PSF parts (INNER + OUTER) previously built by LISAN for a given galaxy+filter.
2) Optionally load an external OUTER PSF (2D stack + radial profile).
3) Display all radial profiles in an interactive matplotlib figure:
   - LEFT: sliders to scale each part (multiplicative factor on the profile).
   - RIGHT: text boxes to define overlap regions (in pixels) between consecutive parts.
4) "Join" parts by scaling them using the overlap region, selecting the radius where S/N is maximal
   within the overlap (using mean/std columns if present).

Notes
-----
- This script is intentionally "phase 1": interactive plotting + scaling + overlap definition + join logic.
- It does NOT yet write out the final joined PSF (you can add export once you like the behavior).
"""

from __future__ import annotations

import os, argparse, subprocess
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button

# -----------------------------------------------------------------------------
# Set plotting style (for better aesthetics in interactive use)
# -----------------------------------------------------------------------------

plt.rc('xtick', labelsize=21)    
plt.rc('ytick', labelsize=21)
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["font.serif"] = ["Times"]
plt.rcParams["mathtext.fontset"] = "stix"

plt.rcParams.update({
    'axes.linewidth': 1.5,           
    'xtick.major.width': 1.5,      
    'ytick.major.width': 1.5,      
    'xtick.color': 'black',       
    'ytick.color': 'black',       
    'axes.edgecolor': 'black',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def _find_profile_and_stack(part_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Given a part output directory, try to locate:
      - profile_psf_*.fits
      - stack_*.fits
    Returns (profile_path, stack_path) or (None, None) if not found.
    """
    prof = None
    stack = None

    # Prefer exact naming if exists; otherwise glob.
    profs = sorted(part_dir.glob("profile_psf_*.fits"))
    stacks = sorted(part_dir.glob("stack_*.fits"))

    if profs:
        prof = profs[0]
    if stacks:
        stack = stacks[0]

    return prof, stack


def _read_profile_fits(profile_path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Read an astscript-radial-profile output FITS and return:
      r_px, mean, std (std can be None if not found)

    Tries to be robust to different HDU/column naming.
    """
    with fits.open(profile_path) as hdul:
        # Find first table HDU with data
        tab = None
        for h in hdul:
            if isinstance(h, (fits.BinTableHDU, fits.TableHDU)) and h.data is not None:
                tab = h.data
                break
        if tab is None:
            raise ValueError(f"No table HDU found in profile FITS: {profile_path}")

        colnames = [c.lower() for c in tab.columns.names]

        # Radius: "semi-major" is typical for astscript-radial-profile with --measure=...semi-major
        r = None
        for key in ("semi-major", "semimajor", "r", "radius"):
            if key in colnames:
                r = np.array(tab[tab.columns.names[colnames.index(key)]], dtype=float)
                break
        if r is None:
            raise ValueError(f"Could not find radius column in: {profile_path} (cols={tab.columns.names})")

        # Mean:
        mean = None
        for key in ("mean", "avg", "average"):
            if key in colnames:
                mean = np.array(tab[tab.columns.names[colnames.index(key)]], dtype=float)
                break
        if mean is None:
            raise ValueError(f"Could not find mean column in: {profile_path} (cols={tab.columns.names})")

        # Std (optional):
        std = None
        for key in ("std", "sigma", "stdev", "stddev"):
            if key in colnames:
                std = np.array(tab[tab.columns.names[colnames.index(key)]], dtype=float)
                break

    # Clean obvious junk
    m = np.isfinite(r) & np.isfinite(mean)
    r = r[m]
    mean = mean[m]
    if std is not None:
        std = std[m]
    return r, mean, std


def _read_stack_image(stack_path: Path) -> np.ndarray:
    """
    Read a stack FITS image. Prefer HDU=1 if present, else HDU=0.
    """
    with fits.open(stack_path) as hdul:
        if len(hdul) > 1 and hdul[1].data is not None and getattr(hdul[1].data, "ndim", 0) == 2:
            img = hdul[1].data
        else:
            img = hdul[0].data
        if img is None or getattr(img, "ndim", 0) != 2:
            raise ValueError(f"No 2D image data found in stack FITS: {stack_path}")
        return np.array(img, dtype=float)


def _parse_overlap(text: str) -> Optional[Tuple[float, float]]:
    """
    Parse overlap "rmin-rmax" or "rmin,rmax" or "rmin rmax".
    Returns (rmin, rmax) in px, or None if invalid.
    """
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None

    # allow "20-30", "20,30", "20 30"
    s = s.replace(",", " ").replace("-", " ")
    parts = [p for p in s.split() if p]
    if len(parts) != 2:
        return None
    try:
        r1 = float(parts[0])
        r2 = float(parts[1])
    except Exception:
        return None
    if not np.isfinite(r1) or not np.isfinite(r2):
        return None
    rmin, rmax = (r1, r2) if r1 <= r2 else (r2, r1)
    if rmax <= rmin:
        return None
    return (rmin, rmax)


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------
@dataclass
class PSFPart:
    name: str
    profile_path: Optional[Path]
    stack_path: Optional[Path]
    r: np.ndarray
    mean: np.ndarray
    std: Optional[np.ndarray]
    scale: float = 1.0

    def snr(self) -> Optional[np.ndarray]:
        if self.std is None:
            return None
        s = np.array(self.std, dtype=float)
        m = np.array(self.mean, dtype=float)
        sn = np.full_like(m, np.nan, dtype=float)
        ok = np.isfinite(m) & np.isfinite(s) & (s > 0)
        sn[ok] = m[ok] / s[ok]
        return sn


# -----------------------------------------------------------------------------
# Loading parts
# -----------------------------------------------------------------------------
def load_lisan_parts(
    gal: str,
    flt: str,
    inner_root: Path,
    outer_root: Path,
) -> List[PSFPart]:
    """
    Load available parts from LISAN outputs.

    Expected (by current PSFBuilder):
      inner_root/<gal>/<gal>_<PART>_<flt>/{stack_*.fits,profile_psf_*.fits}
      outer_root/<gal>/<gal>_<Outer_k>_<flt>/{stack_*.fits,profile_psf_*.fits}
    But we keep it flexible and just search.

    Returns a list sorted as:
      A,B,C,... then Outer_1,Outer_2,...
    """
    parts: List[PSFPart] = []

    def _collect(root: Path, pattern: str) -> List[Path]:
        if not root.exists():
            return []
        return sorted([p for p in root.rglob("*") if p.is_dir() and re.search(pattern, str(p))])
    
    gal_inner_dir = inner_root / gal
    gal_outer_dir = outer_root / gal

    # Direct children dirs are usually "<gal>_<part>_<flt>"
    inner_dirs = []
    if gal_inner_dir.exists():
        inner_dirs = sorted([p for p in gal_inner_dir.iterdir() if p.is_dir() and p.name.endswith(f"_{flt}")])

    outer_dirs = []
    if gal_outer_dir.exists():
        outer_dirs = sorted([p for p in gal_outer_dir.iterdir() if p.is_dir() and p.name.endswith(f"_{flt}")])

    # If structure differs, fallback to rglob
    if not inner_dirs:
        inner_dirs = _collect(inner_root, rf"{re.escape(gal)}_.*_{re.escape(flt)}$")
    if not outer_dirs:
        outer_dirs = _collect(outer_root, rf"{re.escape(gal)}_.*_{re.escape(flt)}$")
    breakpoint()
    # Helper to build PSFPart from dir
    def _make_part_from_dir(d: Path) -> Optional[PSFPart]:
        prof, stack = _find_profile_and_stack(d)
        if prof is None:
            return None
        r, mean, std = _read_profile_fits(prof)
        name = d.name.replace(f"_{flt}", "")
        return PSFPart(name=name, profile_path=prof, stack_path=stack, r=r, mean=mean, std=std, scale=1.0)

    for d in inner_dirs:
        p = _make_part_from_dir(d)
        if p is not None:
            parts.append(p)

    for d in outer_dirs:
        p = _make_part_from_dir(d)
        if p is not None:
            parts.append(p)

    # Sort: inner single-letter first if possible, then Outer_k
    def _sort_key(p: PSFPart):
        n = p.name
        # If inner like "GAL_A" or just "A"
        # Prefer trailing single letter:
        m = re.search(r"(^|_)([A-Z])$", n)
        if m and "OUTER" not in n.upper():
            return (0, m.group(2))
        # Outer_#
        m2 = re.search(r"outer[_\- ]?(\d+)", n, flags=re.IGNORECASE)
        if m2:
            return (1, int(m2.group(1)))
        return (2, n)

    parts = sorted(parts, key=_sort_key)
    return parts


def load_external_outer(external_stack: Optional[Path], external_profile: Optional[Path]) -> Optional[PSFPart]:
    """
    Load external OUTER as a PSFPart named "Outer_EXT".
    """
    if external_profile is None:
        return None
    r, mean, std = _read_profile_fits(external_profile)
    return PSFPart(
        name="Outer_EXT",
        profile_path=external_profile,
        stack_path=external_stack,
        r=r,
        mean=mean,
        std=std,
        scale=1.0,
    )


# -----------------------------------------------------------------------------
# Joining logic (phase 1)
# -----------------------------------------------------------------------------
def find_best_overlap_point(
    p_left: PSFPart,
    p_right: PSFPart,
    rmin: float,
    rmax: float,
) -> Optional[float]:
    """
    Select a radius within [rmin,rmax] where S/N is maximal (requires std for both).
    If std missing, fallback to max(mean_left*mean_right) heuristic.

    Returns r_best (px) or None.
    """
    # common radius grid via interpolation onto left r
    rL = p_left.r
    mL = p_left.mean * p_left.scale
    snL = p_left.snr()
    rmask = (rL >= rmin) & (rL <= rmax) & np.isfinite(mL)

    if not np.any(rmask):
        return None

    r_grid = rL[rmask]

    # interpolate right to left grid
    mR = np.interp(r_grid, p_right.r, p_right.mean * p_right.scale, left=np.nan, right=np.nan)

    if snL is not None and p_right.std is not None:
        snL_g = snL[rmask]
        snR = p_right.snr()
        snR_g = np.interp(r_grid, p_right.r, snR, left=np.nan, right=np.nan)

        sn = np.nanmin(np.vstack([snL_g, snR_g]), axis=0)  # conservative
        if np.all(~np.isfinite(sn)):
            return None
        idx = int(np.nanargmax(sn))
        return float(r_grid[idx])

    # Fallback heuristic if std is missing
    score = mL[rmask] * mR
    if np.all(~np.isfinite(score)):
        return None
    idx = int(np.nanargmax(score))
    return float(r_grid[idx])


def compute_scale_to_match_at_r(
    p_left: PSFPart,
    p_right: PSFPart,
    r_match: float,
    eps: float = 1e-30,
) -> Optional[float]:
    """
    Compute multiplicative factor to apply to RIGHT so that:
      right(r_match) == left(r_match)
    using current scales.
    Returns new_scale_right (absolute), not delta.
    """
    yL = np.interp(r_match, p_left.r, p_left.mean * p_left.scale, left=np.nan, right=np.nan)
    yR = np.interp(r_match, p_right.r, p_right.mean * p_right.scale, left=np.nan, right=np.nan)

    if not np.isfinite(yL) or not np.isfinite(yR):
        return None
    if abs(yR) < eps:
        return None

    factor = yL / yR
    if not np.isfinite(factor) or factor <= 0:
        return None
    return float(p_right.scale * factor)


def join_parts_sequential(
    parts: List[PSFPart],
    overlaps: Dict[Tuple[str, str], Tuple[float, float]],
    direction: str = "inner_to_outer",
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Join sequentially either:
      - inner_to_outer: escala RIGHT para igualar LEFT (tu versión original).
      - outer_to_inner: escala RIGHT (más interno) para igualar LEFT (más externo),
                        recorriendo desde el más externo hacia el más interno.

    direction:
      "inner_to_outer" or "outer_to_inner"

    Updates part.scale in-place.

    Returns diagnostics:
      (left,right): {"r_best":..., "new_scale_right":...}
    """
    info: Dict[Tuple[str, str], Dict[str, float]] = {}

    if len(parts) < 2:
        return info

    if direction not in ("inner_to_outer", "outer_to_inner"):
        raise ValueError("direction must be 'inner_to_outer' or 'outer_to_inner'")

    if direction == "inner_to_outer":
        idx_pairs = [(k, k + 1) for k in range(len(parts) - 1)]
    else:
        # outer_to_inner: recorre desde el final hacia el principio
        idx_pairs = [(k, k - 1) for k in range(len(parts) - 1, 0, -1)]
        # left = parts[k] (outer), right = parts[k-1] (inner)

    for i_left, i_right in idx_pairs:
        left = parts[i_left]
        right = parts[i_right]
        key = (left.name, right.name)

        # OJO: tus overlaps se definen como (parts[i], parts[i+1]) en orden "natural".
        # Si vamos outer_to_inner, debemos buscar también el overlap equivalente.
        if key not in overlaps:
            # intenta el par inverso si el usuario lo definió en el orden opuesto
            key_inv = (right.name, left.name)
            if key_inv not in overlaps:
                continue
            rmin, rmax = overlaps[key_inv]
        else:
            rmin, rmax = overlaps[key]

        r_best = find_best_overlap_point(left, right, rmin, rmax)
        if r_best is None:
            continue

        # Escala RIGHT para que coincida con LEFT en r_best
        new_scale = compute_scale_to_match_at_r(left, right, r_best)
        if new_scale is None:
            continue

        right.scale = new_scale
        info[(left.name, right.name)] = {"r_best": float(r_best), "new_scale_right": float(new_scale)}

    return info

def build_joined_profile_outer_to_inner(
    parts: List[PSFPart],
    join_diag: Dict[Tuple[str, str], Dict[str, float]],
    eps: float = 1e-30,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, str]]]:
    """
    Construye un perfil final (r, y) uniendo desde OUTER -> INNER.
    Asume que parts está ordenado: INNER ... OUTER (como tu sorting).
    join_diag debe venir de join_parts_sequential_direction(..., direction="outer_to_inner").

    Devuelve:
      r_final, y_final, cuts = [(r_cut, "INNERNAME|OUTERNAME"), ...]
    """
    if len(parts) == 0:
        raise ValueError("No parts")
    if len(parts) == 1:
        r = np.array(parts[0].r, dtype=float)
        y = np.clip(parts[0].mean * parts[0].scale, eps, np.inf)
        return r, y, []

    # Identifica cortes usando r_best entre (outer, inner) consecutivos
    # Para cada k: outer=parts[k], inner=parts[k-1]
    cuts = []
    for k in range(len(parts) - 1, 0, -1):
        outer = parts[k]
        inner = parts[k - 1]
        key = (outer.name, inner.name)
        if key not in join_diag:
            # si no se unió, no hay corte fiable -> lo ignoramos
            continue
        r_cut = float(join_diag[key]["r_best"])
        cuts.append((r_cut, f"{inner.name}|{outer.name}"))

    # Ordena cortes crecientes en r
    cuts = sorted(cuts, key=lambda x: x[0])

    # Construye una malla de radios (unión de radios de todas las partes)
    r_all = np.unique(np.concatenate([np.array(p.r, dtype=float) for p in parts]))
    r_all = r_all[np.isfinite(r_all)]
    r_all = np.sort(r_all)

    # Interpola cada parte a r_all
    y_parts = {}
    for p in parts:
        y = np.interp(r_all, p.r, p.mean * p.scale, left=np.nan, right=np.nan)
        y_parts[p.name] = y

    # Segmentación por cortes:
    # - desde r_min hasta cut1 -> usa INNER más interno (parts[0])
    # - entre cut_i y cut_{i+1} -> usa la parte correspondiente
    # - desde último corte -> OUTER más externo (parts[-1])
    #
    # Definimos un “índice de tramo” en base a cuántos cortes hemos pasado.
    # tramo 0 -> parts[0], tramo 1 -> parts[1], ..., tramo N -> parts[N]
    #
    # Nota: si faltan cortes intermedios, el comportamiento será “best effort”.
    cut_radii = [c[0] for c in cuts]
    y_final = np.full_like(r_all, np.nan, dtype=float)

    for i, r in enumerate(r_all):
        # cuántos cortes ha superado este radio
        t = 0
        for rc in cut_radii:
            if r >= rc:
                t += 1
        # tramo t -> parts[t], saturado
        t = min(t, len(parts) - 1)
        y_final[i] = y_parts[parts[t].name][i]

    # Limpieza
    m = np.isfinite(r_all) & np.isfinite(y_final)
    r_final = r_all[m]
    y_final = np.clip(y_final[m], eps, np.inf)

    return r_final, y_final, cuts


# -----------------------------------------------------------------------------
# Interactive GUI
# -----------------------------------------------------------------------------

def interactive_plot(parts: List[PSFPart]) -> None:
    if len(parts) == 0:
        raise ValueError("No PSF parts loaded.")

    fig = plt.figure(figsize=(14.5, 7.5))
    try:
        manager = plt.get_current_fig_manager()
        manager.window.attributes("-fullscreen", True)
        print("\n == PSF Joint Interactive Tool ==")
        print("\n INFO: Fullscreen mode enabled. Press ESC to exit fullscreen.")
        print(" - Use the LEFT sliders to adjust scale factors for each part.")
        print(" - Use the RIGHT text boxes to define overlap regions between parts (e.g., '20-30').")
        print(" - Click 'Join' to apply scaling based on overlaps. Diagnostics will appear in the right panel.\n")
        def _exit_fullscreen(event):
            if event.key == "escape":
                manager.window.attributes("-fullscreen", False)

        fig.canvas.mpl_connect("key_press_event", _exit_fullscreen)

    except Exception:
        pass  # Not critical if fullscreen fails
    # -------------------------------------------------------------------------
    # Widget styling helpers (bigger + bold, consistent with your rcParams)
    # -------------------------------------------------------------------------
    def _style_textbox(tb: TextBox, fontsize: int = 18):
        # Editable text
        tb.text_disp.set_fontsize(fontsize)
        tb.text_disp.set_fontweight("bold")

        # Label text (if any)
        if tb.label is not None:
            tb.label.set_fontsize(fontsize)
            tb.label.set_fontweight("bold")

        # Make the cursor more visible
        try:
            tb.cursor.set_color("black")
            tb.cursor.set_linewidth(2.0)
        except Exception:
            pass

    def _style_button(btn: Button, fontsize: int = 18):
        btn.label.set_fontsize(fontsize)
        btn.label.set_fontweight("bold")

    # -------------------------------------------------------------------------
    # Main plot
    # -------------------------------------------------------------------------
    ax = fig.add_axes([0.26, 0.10, 0.50, 0.84])
    ax.set_xlabel("Radius (px)", fontsize=21)
    ax.set_ylabel("Profile (mean) [arb]", fontsize=21)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.2)

    # ---- Plot lines ----
    lines: Dict[str, any] = {}
    for p in parts:
        y = np.clip(p.mean * p.scale, 1e-30, np.inf)
        (ln,) = ax.plot(p.r, y, lw=1.7, label=p.name)
        lines[p.name] = ln

    # Legend INSIDE plot, bottom-left
    ax.legend(loc="lower left", fontsize=18, frameon=True)

    # ---- Overlap shaded regions + labels + vlines ----
    overlap_patches: Dict[int, any] = {}
    overlap_texts: Dict[int, any] = {}
    overlap_vlines: Dict[int, Tuple[any, any]] = {}

    # ---- LEFT: Compact TextBoxes para escalas ----
    scale_boxes: Dict[str, TextBox] = {}

    n = len(parts)

    left_x = 0.02
    box_width = 0.16
    label_height = 0.025
    box_height = 0.045
    pad = 0.015
    y_start = 0.90

    ax_scale_title = fig.add_axes([left_x, 0.95, box_width, 0.03])
    ax_scale_title.axis("off")
    ax_scale_title.text(0.0, 0.3, "Scale factors", fontsize=18, fontweight="bold")

    for i, p in enumerate(parts):
        y_top = y_start - i * (label_height + box_height + pad)

        ax_label = fig.add_axes([left_x, y_top, box_width, label_height])
        ax_label.axis("off")
        ax_label.text(0.0, 0.0, p.name, fontsize=16, fontweight="bold", ha="left", va="bottom")

        ax_box = fig.add_axes([left_x, y_top - box_height, box_width, box_height])
        tb = TextBox(ax_box, "", initial=f"{p.scale:.6g}")
        _style_textbox(tb, fontsize=18)
        scale_boxes[p.name] = tb

    def _apply_scales():
        for p in parts:
            txt = scale_boxes[p.name].text.strip()
            try:
                val = float(txt)
                if not np.isfinite(val) or val <= 0:
                    continue
                p.scale = val
            except Exception:
                continue

            y = np.clip(p.mean * p.scale, 1e-30, np.inf)
            lines[p.name].set_ydata(y)

        fig.canvas.draw_idle()

    ax_apply = fig.add_axes([left_x, 0.02, box_width, 0.05])
    btn_apply = Button(ax_apply, "Apply")
    _style_button(btn_apply, fontsize=18)
    btn_apply.on_clicked(lambda _evt: _apply_scales())

    for tb in scale_boxes.values():
        tb.on_submit(lambda _txt: _apply_scales())

    # -------------------------------------------------------------------------
    # TOP: Axis limits controls (compact: 2 boxes total) + Set button
    # -------------------------------------------------------------------------
    top_x = 0.26
    top_y = 0.95
    h = 0.035

    ax_lims_title = fig.add_axes([top_x, top_y, 0.12, h])
    ax_lims_title.axis("off")
    ax_lims_title.text(0.0, 0.15, "Axes limits:", fontsize=18, fontweight="bold")

    # One box for x: "xmin,xmax"
    ax_xlim = fig.add_axes([top_x + 0.12, top_y, 0.14, h])
    tb_xlim = TextBox(ax_xlim, "x:", initial="")
    _style_textbox(tb_xlim, fontsize=16)

    # One box for y: "ymin,ymax"
    ax_ylim = fig.add_axes([top_x + 0.28, top_y, 0.14, h])
    tb_ylim = TextBox(ax_ylim, "y:", initial="")
    _style_textbox(tb_ylim, fontsize=16)

    ax_lims_apply = fig.add_axes([top_x + 0.44, top_y, 0.08, h])
    btn_lims = Button(ax_lims_apply, "Set")
    _style_button(btn_lims, fontsize=16)

    def _parse_pair(text: str) -> Optional[Tuple[float, float]]:
        if text is None:
            return None

        s = text.strip()
        if not s:
            return None

        # Separar SOLO por coma o espacio (NO tocar el "-")
        parts = re.split(r"[,\s]+", s)

        if len(parts) != 2:
            return None

        try:
            v1 = float(parts[0])
            v2 = float(parts[1])
        except Exception:
            return None

        if not (np.isfinite(v1) and np.isfinite(v2)):
            return None

        if v2 <= v1:
            return None

        return (v1, v2)

    def _apply_limits(_evt=None):
        x_pair = _parse_pair(tb_xlim.text)
        y_pair = _parse_pair(tb_ylim.text)

        if x_pair is not None:
            ax.set_xlim(*x_pair)
        if y_pair is not None:
            ax.set_ylim(*y_pair)

        fig.canvas.draw_idle()

    btn_lims.on_clicked(_apply_limits)
    tb_xlim.on_submit(lambda _txt: _apply_limits())
    tb_ylim.on_submit(lambda _txt: _apply_limits())

    # -------------------------------------------------------------------------
    # RIGHT: Compact TextBoxes para overlaps (numerados)
    # -------------------------------------------------------------------------
    textboxes: Dict[int, TextBox] = {}
    overlaps: Dict[Tuple[str, str], Tuple[float, float]] = {}
    pair_by_idx: Dict[int, Tuple[str, str]] = {}

    pairs = [(parts[i].name, parts[i + 1].name) for i in range(len(parts) - 1)]
    n_pairs = len(pairs)

    right_x = 0.79
    right_w = 0.19
    ov_label_h = 0.03
    ov_box_h = 0.045
    ov_pad = 0.015
    ov_y_start = 0.90

    ax_ov_title = fig.add_axes([right_x, 0.95, right_w, 0.03])
    ax_ov_title.axis("off")
    ax_ov_title.text(0.0, 0.3, "Overlaps (rmin-rmax px)", fontsize=18, fontweight="bold")

    for i, (a, b) in enumerate(pairs, start=1):
        pair_by_idx[i] = (a, b)
        y_top = ov_y_start - (i - 1) * (ov_label_h + ov_box_h + ov_pad)

        ax_lab = fig.add_axes([right_x, y_top, right_w, ov_label_h])
        ax_lab.axis("off")
        ax_lab.text(0.0, 0.0, f"{i}) {a} → {b}", fontsize=16, fontweight="bold", ha="left", va="bottom")

        ax_box = fig.add_axes([right_x, y_top - ov_box_h, right_w, ov_box_h])
        tb = TextBox(ax_box, "", initial="")
        _style_textbox(tb, fontsize=18)
        textboxes[i] = tb

    def _clear_overlap_artists(idx: int):
        if idx in overlap_patches:
            try:
                overlap_patches[idx].remove()
            except Exception:
                pass
            overlap_patches.pop(idx, None)

        if idx in overlap_texts:
            try:
                overlap_texts[idx].remove()
            except Exception:
                pass
            overlap_texts.pop(idx, None)

        if idx in overlap_vlines:
            v1, v2 = overlap_vlines[idx]
            try:
                v1.remove()
            except Exception:
                pass
            try:
                v2.remove()
            except Exception:
                pass
            overlap_vlines.pop(idx, None)

    def _set_overlap_idx(idx: int, text: str):
        rng = _parse_overlap(text)
        _clear_overlap_artists(idx)

        a, b = pair_by_idx[idx]
        key = (a, b)

        if rng is None:
            overlaps.pop(key, None)
            fig.canvas.draw_idle()
            return

        rmin, rmax = rng
        overlaps[key] = (rmin, rmax)

        # shaded region
        patch = ax.axvspan(rmin, rmax, alpha=0.10)
        overlap_patches[idx] = patch

        # dotted boundary lines
        v1 = ax.axvline(rmin, ls=":", lw=1.3, alpha=0.7)
        v2 = ax.axvline(rmax, ls=":", lw=1.3, alpha=0.7)
        overlap_vlines[idx] = (v1, v2)

        # label uses NUMBER
        y_top = ax.get_ylim()[1]
        txt = ax.text(
            x=np.sqrt(rmin * rmax),
            y=y_top * 0.8,
            s=f"{idx}",
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="top",
            alpha=0.9,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
        )
        overlap_texts[idx] = txt

        fig.canvas.draw_idle()

    for idx, tb in textboxes.items():
        tb.on_submit(lambda text, i=idx: _set_overlap_idx(i, text))

    # -------------------------------------------------------------------------
    # Join button + info panel (right-bottom)
    # -------------------------------------------------------------------------
    ax_btn = fig.add_axes([right_x, 0.10, right_w, 0.08])
    btn = Button(ax_btn, "Join (S/N match in overlaps)")
    _style_button(btn, fontsize=18)

    ax_info = fig.add_axes([right_x, 0.20, right_w, 0.30])
    ax_info.axis("off")
    info_text = ax_info.text(0, 1, "", va="top", fontsize=14, fontweight="bold", family="monospace")

    # Guardamos lo último que se calcule al pulsar Join
    last_diag = {}

    def _do_join(_evt):
        nonlocal last_diag
        _apply_scales()

        # JOIN outer->inner (lo que quieres)
        last_diag = join_parts_sequential(parts, overlaps, direction="outer_to_inner")

        # refresh scale boxes + curves
        for p in parts:
            try:
                scale_boxes[p.name].set_val(f"{p.scale:.6g}")
            except Exception:
                pass
            y = np.clip(p.mean * p.scale, 1e-30, np.inf)
            lines[p.name].set_ydata(y)

        # diagnostics
        lines_out = []
        for (a, b), d in last_diag.items():
            lines_out.append(f"{a} - {b}\n  r_best={d['r_best']:.2f}\n  scale={d['new_scale_right']:.6g}")
        if not lines_out:
            lines_out = ["(no joins applied)"]
        info_text.set_text("\n\n".join(lines_out))

        fig.canvas.draw_idle()

    btn.on_clicked(_do_join)

    plt.show()

    # Tras cerrar la ventana, construimos perfil final si hubo join
    if isinstance(last_diag, dict) and len(last_diag) > 0:
        r_final, y_final, cuts = build_joined_profile_outer_to_inner(parts, last_diag)
    else:
        r_final = np.array([])
        y_final = np.array([])
        cuts = []

    return overlaps, last_diag, r_final, y_final, cuts

# -----------------------------------------------------------------------------
# Power-law interactive fitter (integrated)
# -----------------------------------------------------------------------------
def _parse_pair_allow_sci(text: str) -> Optional[Tuple[float, float]]:
    """
    Parse "vmin,vmax" or "vmin vmax" allowing scientific notation (1e-4).
    Does NOT replace '-' to avoid breaking 1e-4.
    """
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    parts = re.split(r"[,\s]+", s)
    if len(parts) != 2:
        return None
    try:
        v1 = float(parts[0])
        v2 = float(parts[1])
    except Exception:
        return None
    if not (np.isfinite(v1) and np.isfinite(v2)) or v2 <= v1:
        return None
    return (v1, v2)


def _fit_powerlaw(r: np.ndarray, y: np.ndarray, rmin: float, rmax: float) -> Optional[Tuple[float, float]]:
    """
    Fit y = A r^alpha in log-log within [rmin,rmax].
    Returns (alpha, A).
    """
    m = np.isfinite(r) & np.isfinite(y) & (r > 0) & (y > 0) & (r >= rmin) & (r <= rmax)
    if np.count_nonzero(m) < 2:
        return None
    x = np.log10(r[m])
    z = np.log10(y[m])
    alpha, c = np.polyfit(x, z, 1)  # z = c + alpha x
    A = 10.0 ** c
    return float(alpha), float(A)


def _best_overlap_point(r: np.ndarray, y: np.ndarray, std: Optional[np.ndarray], rmin: float, rmax: float) -> Optional[float]:
    m = np.isfinite(r) & np.isfinite(y) & (r >= rmin) & (r <= rmax) & (r > 0) & (y > 0)
    if std is not None:
        m = m & np.isfinite(std) & (std > 0)
    if np.count_nonzero(m) < 1:
        return None
    rr = r[m]
    if std is not None:
        sn = y[m] / std[m]
        idx = int(np.nanargmax(sn))
    else:
        idx = int(np.nanargmax(y[m]))
    return float(rr[idx])


def interactive_powerlaw(
    r: np.ndarray,
    y: np.ndarray,
    std: Optional[np.ndarray] = None,
    *,
    extrap_radius: int,
    return_radius_grid: str = "union",  # "union" or "dense"
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, float]]:
    """
    Interactive power-law fit + PSF–PL overlap matching.

    Returns
    -------
    r_final : np.ndarray
        Final radius grid (px). By default = union of input radii and extrapolated tail.
    y_final : np.ndarray
        Final joined profile: PSF up to r_match (or max r if no overlap), then power-law tail to extrap_radius.
    std_final : Optional[np.ndarray]
        Final std profile (keeps PSF std where available; tail std=None -> filled with NaN).
    meta : dict
        Diagnostics: alpha, A, r_match, rfit_min, rfit_max, rov_min, rov_max.

    Notes
    -----
    - extrap_radius is the target radius for exporting the circular PSF (e.g. 6801). Must be > max(r).
    - The power-law is extrapolated to extrap_radius even if the PSF doesn't reach there.
    - The join point is chosen in the overlap region by maximal S/N (if std available) else maximal y.
    - You must click "Fit + Match" and then close the window to return values.
    """
    if extrap_radius is None:
        raise ValueError("extrap_radius must be provided (e.g. 6801).")
    extrap_radius = int(extrap_radius)
    if extrap_radius <= 0:
        raise ValueError("extrap_radius must be > 0.")
    if not np.all(np.isfinite(r)):
        raise ValueError("r contains non-finite values.")
    if np.nanmax(r) >= extrap_radius:
        raise ValueError(f"extrap_radius={extrap_radius} must be > max(r)={np.nanmax(r)}")

    # Ensure sorted and clean
    r = np.asarray(r, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(r) & np.isfinite(y) & (r > 0)
    r = r[m]
    y = y[m]
    if std is not None:
        std = np.asarray(std, dtype=float)
        std = std[m]

    order = np.argsort(r)
    r = r[order]
    y = y[order]
    if std is not None:
        std = std[order]

    yclip = np.clip(y, 1e-30, np.inf)

    # -------------------------------------------------------------------------
    # helpers local (use your existing ones if already defined in file)
    # -------------------------------------------------------------------------
    def _parse_pair_allow_sci(text: str) -> Optional[Tuple[float, float]]:
        if text is None:
            return None
        s = text.strip()
        if not s:
            return None
        parts = re.split(r"[,\s]+", s)
        if len(parts) != 2:
            return None
        try:
            v1 = float(parts[0])
            v2 = float(parts[1])
        except Exception:
            return None
        if not (np.isfinite(v1) and np.isfinite(v2)):
            return None
        vmin, vmax = (v1, v2) if v1 <= v2 else (v2, v1)
        if vmax <= vmin:
            return None
        return (vmin, vmax)

    def _fit_powerlaw(rr: np.ndarray, yy: np.ndarray, rmin: float, rmax: float) -> Optional[Tuple[float, float]]:
        """
        Fit y = A * r^alpha in log-log space.
        Returns (alpha, A). A is in linear units.
        """
        sel = (rr >= rmin) & (rr <= rmax) & np.isfinite(yy) & (yy > 0)
        if np.count_nonzero(sel) < 3:
            return None
        x = np.log10(rr[sel])
        z = np.log10(yy[sel])
        # linear regression: z = b + alpha*x
        A_mat = np.vstack([np.ones_like(x), x]).T
        try:
            b, alpha = np.linalg.lstsq(A_mat, z, rcond=None)[0]
        except Exception:
            return None
        A_lin = 10.0 ** b
        if not (np.isfinite(alpha) and np.isfinite(A_lin) and A_lin > 0):
            return None
        return float(alpha), float(A_lin)

    def _best_overlap_point(
        rr: np.ndarray,
        yy: np.ndarray,
        ss: Optional[np.ndarray],
        rmin: float,
        rmax: float,
    ) -> Optional[float]:
        sel = (rr >= rmin) & (rr <= rmax) & np.isfinite(yy) & (yy > 0)
        if np.count_nonzero(sel) < 1:
            return None
        rr_sel = rr[sel]
        yy_sel = yy[sel]

        if ss is not None:
            ss_sel = ss[sel]
            ok = np.isfinite(ss_sel) & (ss_sel > 0)
            if np.count_nonzero(ok) >= 1:
                sn = np.full_like(yy_sel, np.nan, dtype=float)
                sn[ok] = yy_sel[ok] / ss_sel[ok]
                if np.any(np.isfinite(sn)):
                    return float(rr_sel[int(np.nanargmax(sn))])

        # fallback: maximum signal
        return float(rr_sel[int(np.nanargmax(yy_sel))])

    # Styling helpers you already use
    def _style_textbox(tb: TextBox, fontsize: int = 18):
        tb.text_disp.set_fontsize(fontsize)
        tb.text_disp.set_fontweight("bold")
        if tb.label is not None:
            tb.label.set_fontsize(fontsize)
            tb.label.set_fontweight("bold")

    def _style_button(btn: Button, fontsize: int = 18):
        btn.label.set_fontsize(fontsize)
        btn.label.set_fontweight("bold")

    # -------------------------------------------------------------------------
    # State to return after window closes
    # -------------------------------------------------------------------------
    state = {
        "alpha": np.nan,
        "A": np.nan,
        "r_match": np.nan,
        "rfit_min": np.nan,
        "rfit_max": np.nan,
        "rov_min": np.nan,
        "rov_max": np.nan,
        "has_solution": False,
    }

    # We'll update these when user clicks Fit+Match
    ypl_current = np.full_like(r, np.nan, dtype=float)

    # -------------------------------------------------------------------------
    # GUI
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(14.5, 7.5))

    # Fullscreen (ESC exits fullscreen)
    try:
        manager = plt.get_current_fig_manager()
        manager.window.attributes("-fullscreen", True)

        def _exit_fullscreen(event):
            if event.key == "escape":
                manager.window.attributes("-fullscreen", False)

        fig.canvas.mpl_connect("key_press_event", _exit_fullscreen)
    except Exception:
        pass

    ax = fig.add_axes([0.20, 0.10, 0.60, 0.84])
    ax.set_xlabel("Radius (px)", fontsize=21)
    ax.set_ylabel("Profile (mean) [arb]", fontsize=21)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.2)

    (ln_psf,) = ax.plot(r, yclip, lw=2.0, label="PSF (joined)")
    (ln_pl,) = ax.plot(r, np.clip(ypl_current, 1e-30, np.inf), lw=2.0, ls="--", label="Power law")

    ax.legend(loc="lower left", fontsize=18, frameon=True)

    # Right panel
    right_x = 0.82
    right_w = 0.16

    ax_title = fig.add_axes([right_x, 0.95, right_w, 0.03])
    ax_title.axis("off")
    ax_title.text(0.0, 0.3, "Power-law fit", fontsize=18, fontweight="bold")

    ax_fit_lab = fig.add_axes([right_x, 0.88, right_w, 0.03])
    ax_fit_lab.axis("off")
    ax_fit_lab.text(0.0, 0.0, "Fit range (rmin rmax)", fontsize=16, fontweight="bold", va="bottom")

    ax_fit = fig.add_axes([right_x, 0.84, right_w, 0.045])
    tb_fit = TextBox(ax_fit, "", initial="")
    _style_textbox(tb_fit, 18)

    ax_ov_lab = fig.add_axes([right_x, 0.77, right_w, 0.03])
    ax_ov_lab.axis("off")
    ax_ov_lab.text(0.0, 0.0, "Overlap (rmin rmax)", fontsize=16, fontweight="bold", va="bottom")

    ax_ov = fig.add_axes([right_x, 0.73, right_w, 0.045])
    tb_ov = TextBox(ax_ov, "", initial="")
    _style_textbox(tb_ov, 18)

    # Axes limits (top, compact 2 boxes)
    top_x = 0.20
    top_y = 0.95
    hh = 0.035

    ax_lims_title = fig.add_axes([top_x, top_y, 0.12, hh])
    ax_lims_title.axis("off")
    ax_lims_title.text(0.0, 0.15, "Axes limits:", fontsize=18, fontweight="bold")

    ax_xlim = fig.add_axes([top_x + 0.12, top_y, 0.14, hh])
    tb_xlim = TextBox(ax_xlim, "x:", initial="")
    _style_textbox(tb_xlim, fontsize=16)

    ax_ylim = fig.add_axes([top_x + 0.28, top_y, 0.14, hh])
    tb_ylim = TextBox(ax_ylim, "y:", initial="")
    _style_textbox(tb_ylim, fontsize=16)

    ax_set = fig.add_axes([top_x + 0.44, top_y, 0.08, hh])
    btn_set = Button(ax_set, "Set")
    _style_button(btn_set, fontsize=16)

    def _apply_limits(_evt=None):
        xp = _parse_pair_allow_sci(tb_xlim.text)
        yp = _parse_pair_allow_sci(tb_ylim.text)
        if xp is not None:
            ax.set_xlim(*xp)
        if yp is not None:
            ax.set_ylim(*yp)
        fig.canvas.draw_idle()

    btn_set.on_clicked(_apply_limits)
    tb_xlim.on_submit(lambda _t: _apply_limits())
    tb_ylim.on_submit(lambda _t: _apply_limits())

    # Info
    ax_info = fig.add_axes([right_x, 0.35, right_w, 0.35])
    ax_info.axis("off")
    info = ax_info.text(0, 1, "", va="top", fontsize=14, fontweight="bold", family="monospace")

    # Apply button
    ax_btn = fig.add_axes([right_x, 0.10, right_w, 0.08])
    btn = Button(ax_btn, "Fit + Match")
    _style_button(btn, 18)

    fit_patch = None
    ov_patch = None
    join_vline = None

    def _update_patches(rfit, rov, rmatch):
        nonlocal fit_patch, ov_patch, join_vline
        for obj in (fit_patch, ov_patch, join_vline):
            if obj is not None:
                try:
                    obj.remove()
                except Exception:
                    pass
        fit_patch = None
        ov_patch = None
        join_vline = None

        if rfit is not None:
            fit_patch = ax.axvspan(rfit[0], rfit[1], alpha=0.08)
        if rov is not None:
            ov_patch = ax.axvspan(rov[0], rov[1], alpha=0.10)
        if rmatch is not None and np.isfinite(rmatch):
            join_vline = ax.axvline(rmatch, ls=":", lw=2.0, alpha=0.8)

    def _do_fit(_evt=None):
        nonlocal ypl_current

        rfit = _parse_pair_allow_sci(tb_fit.text)
        rov = _parse_pair_allow_sci(tb_ov.text)

        if rfit is None:
            state.update({"has_solution": False})
            info.set_text("Provide fit range.")
            _update_patches(None, rov, None)
            fig.canvas.draw_idle()
            return

        sol = _fit_powerlaw(r, yclip, rfit[0], rfit[1])
        if sol is None:
            state.update({"has_solution": False})
            info.set_text("Fit failed (few points).")
            _update_patches(rfit, rov, None)
            fig.canvas.draw_idle()
            return

        alpha, A = sol
        ypl = A * (r ** alpha)

        r_match = None
        if rov is not None:
            r_match = _best_overlap_point(r, yclip, std, rov[0], rov[1])
            if r_match is not None:
                y_psf = float(np.interp(r_match, r, yclip))
                y_pl = float(np.interp(r_match, r, ypl))
                if np.isfinite(y_psf) and np.isfinite(y_pl) and y_pl > 0:
                    fac = y_psf / y_pl
                    A *= fac
                    ypl *= fac

        ypl_current = ypl
        ln_pl.set_data(r, np.clip(ypl_current, 1e-30, np.inf))

        # update state for return
        state["alpha"] = float(alpha)
        state["A"] = float(A)
        state["rfit_min"] = float(rfit[0])
        state["rfit_max"] = float(rfit[1])
        if rov is not None:
            state["rov_min"] = float(rov[0])
            state["rov_max"] = float(rov[1])
        else:
            state["rov_min"] = np.nan
            state["rov_max"] = np.nan

        if r_match is not None:
            state["r_match"] = float(r_match)
        else:
            state["r_match"] = np.nan

        state["has_solution"] = True

        out = [f"alpha = {alpha:.6g}", f"A     = {A:.6g}"]
        if r_match is not None:
            out.append(f"r_match = {r_match:.6g}")
        out.append(f"extrap_radius = {extrap_radius:d}")
        info.set_text("\n".join(out))

        _update_patches(rfit, rov, r_match)
        fig.canvas.draw_idle()

    btn.on_clicked(_do_fit)
    tb_fit.on_submit(lambda _t: _do_fit())
    tb_ov.on_submit(lambda _t: _do_fit())

    plt.show()

    # -------------------------------------------------------------------------
    # Build FINAL union profile after window closes
    # -------------------------------------------------------------------------
    if not state["has_solution"]:
        # If user never fit: return original profile, truncated/extended? -> keep original.
        meta = {k: float(v) for k, v in state.items() if k != "has_solution"}
        return r.copy(), yclip.copy(), (std.copy() if std is not None else None), meta

    alpha = float(state["alpha"])
    A = float(state["A"])
    r_match = float(state["r_match"]) if np.isfinite(state["r_match"]) else np.nan

    # Decide join radius:
    # - If r_match defined: PSF up to r_match, PL from r_match to extrap_radius
    # - Else: default to last radius of PSF (continuous tail from the end)
    if np.isfinite(r_match):
        r_join = r_match
    else:
        r_join = float(r[-1])

    # Build target radius grid
    if return_radius_grid == "dense":
        # 1 px step tail beyond max(r), keep original points as well
        r_tail = np.arange(int(np.floor(np.nanmax(r))) + 1, extrap_radius + 1, 1, dtype=float)
        r_final = np.unique(np.concatenate([r, r_tail]))
        r_final.sort()
    else:
        # "union": keep original r, append integer radii tail if needed
        r_tail = np.arange(int(np.ceil(np.nanmax(r))) + 1, extrap_radius + 1, 1, dtype=float)
        r_final = np.unique(np.concatenate([r, r_tail]))
        r_final.sort()

    # Evaluate PL on r_final
    ypl_final = A * (r_final ** alpha)
    ypl_final = np.clip(ypl_final, 1e-30, np.inf)

    # Interpolate PSF onto r_final for inner part
    ypsf_final = np.interp(r_final, r, yclip, left=np.nan, right=np.nan)

    # Compose final profile
    y_final = np.empty_like(r_final, dtype=float)
    # PSF region
    m_psf = (r_final <= r_join) & np.isfinite(ypsf_final)
    y_final[m_psf] = ypsf_final[m_psf]
    # Tail region
    m_tail = ~m_psf
    y_final[m_tail] = ypl_final[m_tail]

    # std_final: keep PSF std where available, else NaN
    std_final = None
    if std is not None:
        std_interp = np.interp(r_final, r, std, left=np.nan, right=np.nan)
        std_final = np.full_like(r_final, np.nan, dtype=float)
        std_final[m_psf] = std_interp[m_psf]
        # tail stays NaN

    meta = {
        "alpha": alpha,
        "A": A,
        "r_match": float(r_join),
        "rfit_min": float(state["rfit_min"]),
        "rfit_max": float(state["rfit_max"]),
        "rov_min": float(state["rov_min"]),
        "rov_max": float(state["rov_max"]),
        "extrap_radius": float(extrap_radius),
    }
    return r_final, y_final, std_final, meta



def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"{' '.join(cmd)}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}"
        )

def _ensure_odd_radius(radius: int) -> int:
    radius = int(radius)
    if radius <= 0:
        raise ValueError("circular_radius must be > 0")
    if radius % 2 == 0:
        raise ValueError(
            f"circular_radius must be ODD so the center is exactly at the image center. Got {radius}."
        )
    return radius

def write_profile_to_fits_table(path: Path, r_px: np.ndarray, mean: np.ndarray) -> None:
    """
    Writes a simple 2-column table FITS with columns: RADIUS, MEAN
    (fits.BinTableHDU).
    """
    from astropy.table import Table
    tab = Table()
    tab["RADIUS"] = np.asarray(r_px, dtype=float)
    tab["MEAN"] = np.asarray(mean, dtype=float)
    tab.write(str(path), overwrite=True)

def circularize_profile_to_2d(
    *,
    r_px: np.ndarray,
    mean: np.ndarray,
    out_dir: Path,
    name_control: str,
    radius: int,
) -> Tuple[Path, Path]:
    """
    Creates:
      - psf_<name_control>.fits (2D circularized, normalized)
      - psf_profile_<name_control>.fits (radial profile of the 2D PSF)

    Uses GNUAstro:
      asttable (sorted-to-interval) and astmkprof --customtable
      astscript-radial-profile to verify/save profile.
    """
    radius = _ensure_odd_radius(radius)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    interval_dir = out_dir / "Intervals"
    custom_table_dir = out_dir / "Custom_tables"
    circular_profiles_dir = out_dir / "Circular_profiles"
    interval_dir.mkdir(parents=True, exist_ok=True)
    custom_table_dir.mkdir(parents=True, exist_ok=True)
    circular_profiles_dir.mkdir(parents=True, exist_ok=True)

    # 1) Write temporary table with RADIUS/MEAN (so we don't depend on a prior FITS)
    tmp_profile_table = out_dir / f"{name_control}_profile_tmp.fits"
    write_profile_to_fits_table(tmp_profile_table, r_px, mean)

    # 2) Make interval + custom table (GNUAstro expects "arith RADIUS sorted-to-interval,MEAN")
    interval_path = interval_dir / f"{name_control}_interval_tmp.fits"
    custom_table_path = custom_table_dir / f"{name_control}_custom_table.fits"

    _run([
        "asttable", str(tmp_profile_table),
        f"--output={interval_path}",
        '-c', 'arith RADIUS sorted-to-interval,MEAN',
        "-o", str(custom_table_path),
    ])

    # 3) Build 2D image size (2*radius+1) and center (radius, radius) in 0-based indexing
    size = (2 * radius)-1
    center = radius  # center pixel index

    path_circular = out_dir / f"psf_{name_control}.fits"

    # Build astmkprof via stdin line:
    # "1 <xcenter> <ycenter> 8 <xsize> <ysize> 0 0 1 0 1"
    # Here xcenter=ycenter=center, xsize=ysize=size
    os.system(f'echo "1 {center} {center} 8 {size} 0 0 1 0 1" \
                    | astmkprof --customtable={custom_table_path} \
                                --mergedsize={size},{size} \
                                --output={path_circular} \
                                --mcolnocustprof \
                                --oversample=1 \
                                --clearcanvas \
                                --mode=img')


    # 4) Normalize 2D PSF to sum=1
    img2d = fits.getdata(path_circular)
    s = np.nansum(img2d)
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError(f"Invalid PSF sum after astmkprof: sum={s}")
    img2d = img2d / s
    fits.writeto(path_circular, img2d, overwrite=True)

    # 5) Measure radial profile of the 2D PSF (up to radius)
    profile_out = out_dir / f"psf_profile_{name_control}.fits"
    _run([
        "astscript-radial-profile", str(path_circular),
        "--quiet",
        "--hdu=0",
        "--measure=mean,std,area,semi-major",
        f"--rmax={radius}",
        "-o", str(profile_out),
    ])

    return path_circular, profile_out

def parse_args():
    p = argparse.ArgumentParser(description="Interactive PSF joiner (LISAN).")
    p.add_argument("--dir", required=True, help="Galaxy name (e.g., PGC10074).")
    p.add_argument("--filter", required=True, help="Filter label (e.g., g,r,lum).")
    p.add_argument("--inner-root", default="./PSF_files/Inner_parts", help="Root for INNER outputs.")
    p.add_argument("--outer-root", default="./PSF_files/Outer_parts", help="Root for OUTER outputs.")
    p.add_argument("--external-outer-stack", default=None, help="Optional external OUTER 2D stack FITS.")
    p.add_argument("--external-outer-profile", default=None, help="Optional external OUTER profile FITS.")

    p.add_argument(
        "--export-circular",
        action="store_true",
        help="After joining parts (+ optional powerlaw), export a circularized 2D PSF using astmkprof.",
    )
    p.add_argument(
        "--circular-radius",
        type=int,
        default=None,
        help="Radius in pixels for circular PSF (must be ODD, e.g. 6801).",
    )
    p.add_argument(
        "--final-root",
        default="./PSF_files",
        help="Root folder where PSFs_complete/<gal>_<filter>/... will be created (default: ./PSF_files).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # -----------------------
    # Validación CLI coherente
    # -----------------------
    if args.export_circular and args.circular_radius is None:
        raise SystemExit("--export-circular requiere --circular-radius (ODD).")
    if args.circular_radius is not None:
        # fuerza impar aquí (falla con error claro)
        _ensure_odd_radius(args.circular_radius)

    # -----------------------
    # Load parts
    # -----------------------
    
    parts = load_lisan_parts(
        gal=str(args.dir),
        flt=str(args.filter),
        inner_root=Path(args.inner_root),
        outer_root=Path(args.outer_root),
    )
    
    ext = load_external_outer(
        external_stack=Path(args.external_outer_stack) if args.external_outer_stack else None,
        external_profile=Path(args.external_outer_profile) if args.external_outer_profile else None,
    )
    if ext is not None:
        parts.append(ext)

    # Re-sort (keeps your convention)
    def _sort_key(p: PSFPart):
        n = p.name
        m = re.search(r"(^|_)([A-Z])$", n)
        if m and "OUTER" not in n.upper():
            return (0, m.group(2))
        m2 = re.search(r"outer[_\- ]?(\d+)", n, flags=re.IGNORECASE)
        if m2:
            return (1, int(m2.group(1)))
        if n.lower() == "outer_ext":
            return (2, 10**9)
        return (3, n)

    parts = sorted(parts, key=_sort_key)

    # -----------------------
    # Interactive join -> returns joined profile
    # -----------------------
    overlaps, join_diag, r_joined, y_joined, cuts = interactive_plot(parts)

    if r_joined is None or len(r_joined) == 0:
        raise SystemExit("No joined profile available. Click 'Join' at least once and close the window.")

    # -----------------------
    # OPTIONAL: powerlaw + merge to extrap_radius
    # Only if user wants export (or at least provided circular_radius)
    # -----------------------
    r_final, y_final = r_joined, y_joined
    std_final = None
    meta = {}

    if args.export_circular:
        # powerlaw GUI returns profile merged PSF+PL out to circular_radius
        r_final, y_final, std_final, meta = interactive_powerlaw(
            r_joined, y_joined, None,
            extrap_radius=int(args.circular_radius),
        )

        # -----------------------
        # Circularize to 2D
        # -----------------------
        name_control = f"{args.dir}_{args.filter}"
        final_dir = Path(args.final_root) / "PSFs_complete" / name_control
        final_dir.mkdir(parents=True, exist_ok=True)

        psf2d_path, psfprof_path = circularize_profile_to_2d(
            r_px=r_final,
            mean=y_final,
            out_dir=final_dir,
            name_control=name_control,
            radius=int(args.circular_radius),
        )

        print(f"\nSaved circularized 2D PSF: {psf2d_path}")
        print(f"Saved radial profile of 2D PSF: {psfprof_path}")

    # Diagnostics (siempre)
    if r_final.size > 0:
        print("\n[OK] Joined PSF profile built.")
        if cuts:
            print("Cuts (r_cut px, inner|outer):")
            for rc, lab in cuts:
                print(f"  {rc:.3f}  {lab}")

        if meta:
            print("\nPowerlaw meta:")
            for k, v in meta.items():
                print(f"  {k}: {v}")

if __name__ == "__main__":
    main()