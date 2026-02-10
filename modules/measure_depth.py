"""
lisan_depth.py

Auxiliary functions (called from lisan.py) to measure image depth with GNUastro.
Adds incremental persistence of depths into a FITS binary table.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, List

from astropy.io import fits
from astropy.table import Table, vstack
from tqdm import tqdm


def _run(cmd: List[str], *, capture: bool = False, text: str | None = None) -> str:
    """Run external command (no shell). Return stdout if capture=True."""
    p = subprocess.run(
        cmd,
        input=text,
        text=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )
    return (p.stdout or "").strip()


def _parse_sblmag(astfits_stdout: str) -> float:
    """Mimic: grep SBLMAG | awk 'NR==2 {print $3}'."""
    lines = [ln for ln in astfits_stdout.splitlines() if "SBLMAG" in ln]
    if len(lines) < 2:
        raise ValueError("Could not find the expected SBLMAG lines in astfits output.")
    return float(lines[1].split()[2])


def _append_depth_fits(fits_path: Path, name: str, flt: str, sblmag: float) -> None:
    """Append one row to DEPTH table in a FITS file (create if missing)."""
    fits_path = Path(fits_path)
    row = Table(
        {
            "NAME": [name],
            "FILTER": [flt],
            "SBLMAG": [float(sblmag)],
            "DATE": [datetime.utcnow().isoformat(timespec="seconds")],
        }
    )

    if not fits_path.exists():
        fits_path.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList(
            [fits.PrimaryHDU(), fits.BinTableHDU(row, name="DEPTH")]
        ).writeto(fits_path)
        return

    with fits.open(fits_path, mode="update") as hdul:
        if "DEPTH" not in hdul:
            hdul.append(fits.BinTableHDU(row, name="DEPTH"))
        else:
            tab = Table(hdul["DEPTH"].data)
            hdul["DEPTH"].data = vstack(
                [tab, row], metadata_conflicts="silent"
            ).as_array()
        hdul.flush()


def calculate_depth(
    directorio: str | Path,
    filters: Iterable[str],
    *,
    output_dir: str | Path,
    depth_fits: str | Path,
    noisechisel_args: Optional[List[str]] = None,
    zeropoint: float = 22.5,
    sfmagarea: int = 100,
    sfmagnsigma: float = 3.0,
    upnsigma: float = 3.0,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Measure depth (SBLMAG) for each FITS in `directorio` matching each filter, and
    append results incrementally to `depth_fits` (FITS binary table 'DEPTH').

    Returns dict: { "<obj>_<filter>": sblmag }
    """
    directorio = Path(directorio)
    output_dir = Path(output_dir)
    depth_fits = os.path.join(output_dir,Path(depth_fits))

    masks_root = output_dir / "Masks_noisechisel"
    masks_root.mkdir(parents=True, exist_ok=True)

    if noisechisel_args is None:
        noisechisel_args = [
            "--hdu=1",
            "--tilesize=20,20",
            "--interpnumngb=5",
            "--dthresh=0.1",
            "--snminarea=2",
            "--quiet",
        ]

    results: Dict[str, float] = {}

    for flt in filters:
        pat = re.compile(rf"_{re.escape(flt)}.*\.fits$", re.IGNORECASE)
        files = sorted(p for p in directorio.iterdir() if p.is_file() and pat.search(p.name))

        it = tqdm(files, desc=f"Depth {flt}", disable=not show_progress)
        for ruta_imagen in it:
            obj = ruta_imagen.name.split("_")[0]
            name = f"{obj}_{flt}"

            out_dir = masks_root / name
            out_dir.mkdir(parents=True, exist_ok=True)
            maskdir = out_dir / f"{name}_masked.fits"

            with tempfile.TemporaryDirectory(prefix=f"lisan_depth_{name}_") as tmp:
                tmp = Path(tmp)
                lab = tmp / "lab.fits"
                sbl = tmp / "sbl.fits"

                _run(
                    ["astnoisechisel", str(ruta_imagen), *noisechisel_args, f"--output={maskdir}"]
                )

                _run(
                    [
                        "astmkprof",
                        f"--background={maskdir}",
                        "--clearcanvas",
                        "--mforflatpix",
                        "--quiet",
                        "--type=uint8",
                        f"--output={lab}",
                    ],
                    text="1 2000 2000 5 50 0 0 1 1 1\n",
                )

                _run(
                    [
                        "astmkcatalog",
                        str(lab),
                        "-h1",
                        f"--zeropoint={zeropoint}",
                        f"-o{sbl}",
                        f"--sfmagarea={sfmagarea}",
                        f"--sfmagnsigma={sfmagnsigma}",
                        "--forcereadstd",
                        f"--valuesfile={maskdir}",
                        "--valueshdu=INPUT-NO-SKY",
                        f"--upmaskfile={maskdir}",
                        "--upmaskhdu=DETECTIONS",
                        f"--upnsigma={upnsigma}",
                        "--checkuplim=1",
                        "--quiet",
                        "--upnum=10000",
                        "--ids",
                        "--upperlimit-sb",
                        "--upperlimit-mag",
                        "--envseed",
                    ]
                )

                sblmag = _parse_sblmag(_run(["astfits", str(sbl), "-h1"], capture=True))

            results[name] = sblmag
            _append_depth_fits(depth_fits, name=name, flt=flt, sblmag=sblmag)

    return results
