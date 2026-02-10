"""
masking.py

Utilities to create image masks using GNUastro (NoiseChisel + Segment).

This module is meant to be imported and called from the LISAN main script.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, List

from tqdm import tqdm


def _run(cmd: List[str]) -> None:
    """Run external command, silencing GNUastro warnings."""
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def make_masks(
    input_dir: Path,
    *,
    output_dir: Path,
    noisechisel_args: Optional[List[str]] = None,
    segment_args: Optional[List[str]] = None,
) -> None:
    """
    Create masks for all FITS files in a directory.

    A mask is considered completed when both astnoisechisel and astsegment
    finish successfully.

    Parameters
    ----------
    input_dir : Path
        Directory containing FITS images.
    output_dir : Path
        Base output directory for mask products.
    noisechisel_args : list of str, optional
        Extra parameters for astnoisechisel.
    segment_args : list of str, optional
        Extra parameters for astsegment.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    mask_nc_dir = output_dir / "Mask_noisechisel"
    mask_seg_dir = output_dir / "Mask_segment"

    mask_nc_dir.mkdir(parents=True, exist_ok=True)
    mask_seg_dir.mkdir(parents=True, exist_ok=True)

    if noisechisel_args is None:
        noisechisel_args = [
            "--tilesize=20,20",
            "--interpnumngb=5",
            "--dthresh=0.05",
            "--snminarea=2",
            "--rawoutput",
            "--quiet",
        ]

    if segment_args is None:
        segment_args = [
            "--tilesize=10,10",
            "--interpnumngb=1",
            "--gthresh=-10",
            "--objbordersn=0",
            "--minnumfalse=1",
            "--quiet",
        ]

    fits_files = sorted(input_dir.glob("*.fits"))

    # Progress bar: one step == one completed mask
    for image in tqdm(
        fits_files,
        desc="Creating masks",
        unit="mask",
    ):
        name = image.stem

        out_nc = mask_nc_dir / f"{name}_noisechisel.fits"
        out_seg = mask_seg_dir / f"{name}_segment.fits"

        # NoiseChisel
        _run([
            "astnoisechisel",
            str(image),
            *noisechisel_args,
            f"--output={out_nc}",
        ])

        # Segment
        _run([
            "astsegment",
            str(out_nc),
            *segment_args,
            f"--output={out_seg}",
        ])
