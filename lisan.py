"""
Main LISAN pipeline script.

Reads parameters from CLI (argparse) and runs selected pipeline stages.
"""

import sys
import argparse
import shlex
from pathlib import Path

modules_folder = "./modules"
modules_path = str(Path(modules_folder).resolve())
if modules_path not in sys.path:
    sys.path.append(modules_path)

from measure_depth import calculate_depth
from masking import make_masks
from making_catalogs import make_catalogs
from psf_builder import PSFBuilder


def parse_args():
    parser = argparse.ArgumentParser(
        description="LISAN pipeline: depth, masks, catalogs, and PSF building."
    )

    # -------------------------------------------------------------------------
    # Global
    # -------------------------------------------------------------------------
    parser.add_argument("--dir", required=True, help="Directory with FITS cutouts.")
    parser.add_argument("--filters", required=True, help="Comma-separated filters, e.g. g,r,i.")

    # -------------------------------------------------------------------------
    # MASKING
    # -------------------------------------------------------------------------
    parser.add_argument("--make-masks", action="store_true")
    parser.add_argument("--mask-output-dir", default="Process_data/Mask_data")
    parser.add_argument("--noisechisel-params", type=str, default=None)
    parser.add_argument("--segment-params", type=str, default=None)

    # -------------------------------------------------------------------------
    # DEPTH
    # -------------------------------------------------------------------------
    parser.add_argument("--measure-depth", action="store_true")
    parser.add_argument("--depth-output-dir", default="Measure_Depth")
    parser.add_argument("--depth-fits", default="depths.fits")
    parser.add_argument("--depth-zeropoint", type=float, default=22.5)
    parser.add_argument("--depth-sfmagarea", type=int, default=100)
    parser.add_argument("--depth-sfmagnsigma", type=float, default=3.0)
    parser.add_argument("--depth-upnsigma", type=float, default=3.0)
    parser.add_argument("--depth-noisechisel-params", type=str, default=None)

    # -------------------------------------------------------------------------
    # CATALOGS
    # -------------------------------------------------------------------------
    parser.add_argument("--make-catalogs", action="store_true")
    parser.add_argument("--cats-output-dir", default="Process_data/Make_catalogs")
    parser.add_argument("--cats-zeropoint", type=float, default=22.5)
    parser.add_argument("--cats-noisechisel-params", type=str, default=None)
    parser.add_argument("--cats-segment-params", type=str, default=None)
    parser.add_argument("--cats-aperture-arcsec", type=float, default=10.0)
    parser.add_argument("--cats-gaia-dataset", default="dr3")
    parser.add_argument("--cats-gaia-sigma", type=float, default=3.0)
    parser.add_argument("--cats-no-plots", action="store_true")

    # -------------------------------------------------------------------------
    # PSF BUILDING
    # -------------------------------------------------------------------------
    parser.add_argument("--build-psf", action="store_true", help="Enable PSF building stage.")
    parser.add_argument(
        "--psf-select-parts",
        default="I",
        help="I (inner), O (outer), B (both). Default I.",
    )

    # Optional configuration for INNER parts (comma-separated lists)
    parser.add_argument("--psf-parts", default="A,B,C", help="Inner part labels. Default A,B,C.")
    parser.add_argument(
        "--psf-min-dist",
        default="0.015,0.015,0.015",
        help="mindistdeg per part (comma-separated).",
    )
    parser.add_argument(
        "--psf-norm-radii",
        default="5,10;10,20;20,40",
        help="normradii per part: 'a,b;c,d;...'.",
    )
    parser.add_argument(
        "--psf-width-image",
        default="200,200;400,400;800,800",
        help="widthinpix per part: 'x,y;x,y;...'.",
    )
    parser.add_argument(
        "--psf-selection-radii",
        default="90,90,90",
        help="Selection radius (arcmin) per part.",
    )

    # NEW: center-refinement params (wired to PSFBuilder -> graph_interactive_center)
    parser.add_argument(
        "--psf-center-sigma",
        type=float,
        default=5.0,
        help="Sigma for sigma-clipped stats in star-center refinement (default: 5.0).",
    )
    parser.add_argument(
        "--psf-center-max-iter",
        type=int,
        default=5,
        help="Max iterations for sigma-clipped stats in star-center refinement (default: 5).",
    )

    # -------------------------------------------------------------------------
    # PSF magnitude limits for branch selection
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--psf-branch-mag-min",
        type=float,
        default=16.0,
        help="Lower magnitude limit for point-like branch estimation in PSF stage (default: 16.0).",
    )
    parser.add_argument(
        "--psf-branch-mag-max",
        type=float,
        default=18.0,
        help="Upper magnitude limit for point-like branch estimation in PSF stage (default: 18.0).",
    )

    # -------------------------------------------------------------------------
    # PSF mask-making params
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--psf-nc-inner-params",
        type=str,
        default=None,
        help="Extra params for astnoisechisel when masking INNER star-stamps (PSF stage). "
             "Example: '--tilesize=20,20 --qthresh=0.6'. If omitted, defaults in psf_builder.py are used.",
    )
    parser.add_argument(
        "--psf-seg-inner-params",
        type=str,
        default=None,
        help="Extra params for astsegment when masking INNER star-stamps (PSF stage). "
             "If omitted, defaults in psf_builder.py are used.",
    )
    parser.add_argument(
        "--psf-nc-outer-params",
        type=str,
        default=None,
        help="Extra params for astnoisechisel when masking OUTER star-stamps (PSF stage). "
             "If omitted, defaults in psf_builder.py are used.",
    )
    parser.add_argument(
        "--psf-seg-outer-params",
        type=str,
        default=None,
        help="Extra params for astsegment when masking OUTER star-stamps (PSF stage). "
             "If omitted, defaults in psf_builder.py are used.",
    )

    return parser.parse_args()


def _parse_filters(filters: str) -> list[str]:
    return [f.strip() for f in filters.split(",") if f.strip()]


def _parse_list_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_pairs_semicolon(s: str) -> list[str]:
    # "5,10;10,20;20,40" -> ["5,10","10,20","20,40"]
    return [x.strip() for x in s.split(";") if x.strip()]


def main():
    args = parse_args()

    input_dir = Path(args.dir)
    filters = _parse_filters(args.filters)

    # -------------------------------------------------------------------------
    # MASKING
    # -------------------------------------------------------------------------
    if args.make_masks:
        nc_args = shlex.split(args.noisechisel_params) if args.noisechisel_params else None
        seg_args = shlex.split(args.segment_params) if args.segment_params else None
        make_masks(
            input_dir,
            output_dir=Path(args.mask_output_dir),
            noisechisel_args=nc_args,
            segment_args=seg_args,
        )

    # -------------------------------------------------------------------------
    # DEPTH
    # -------------------------------------------------------------------------
    if args.measure_depth:
        depth_nc_args = shlex.split(args.depth_noisechisel_params) if args.depth_noisechisel_params else None
        calculate_depth(
            directorio=input_dir,
            filters=filters,
            output_dir=Path(args.depth_output_dir),
            depth_fits=Path(args.depth_fits),
            zeropoint=args.depth_zeropoint,
            sfmagarea=args.depth_sfmagarea,
            sfmagnsigma=args.depth_sfmagnsigma,
            upnsigma=args.depth_upnsigma,
            noisechisel_args=depth_nc_args,
            show_progress=True,
        )

    # -------------------------------------------------------------------------
    # CATALOGS
    # -------------------------------------------------------------------------
    if args.make_catalogs:
        cats_nc_args = shlex.split(args.cats_noisechisel_params) if args.cats_noisechisel_params else None
        cats_seg_args = shlex.split(args.cats_segment_params) if args.cats_segment_params else None
        make_catalogs(
            input_dir=input_dir,
            filters=filters,
            output_dir=Path(args.cats_output_dir),
            zeropoint=args.cats_zeropoint,
            noisechisel_args=cats_nc_args,
            segment_args=cats_seg_args,
            aperture_arcsec=args.cats_aperture_arcsec,
            gaia_dataset=args.cats_gaia_dataset,
            gaia_sigma=args.cats_gaia_sigma,
            make_plots=not args.cats_no_plots,
            show_progress=True,
        )

    # -------------------------------------------------------------------------
    # PSF BUILDING
    # -------------------------------------------------------------------------
    if args.build_psf:
        parts = _parse_list_csv(args.psf_parts)
        min_dist = _parse_list_csv(args.psf_min_dist)
        norm_radii = _parse_pairs_semicolon(args.psf_norm_radii)
        width_image = _parse_pairs_semicolon(args.psf_width_image)
        selection_radii = [float(x) for x in _parse_list_csv(args.psf_selection_radii)]

        psf_nc_inner = shlex.split(args.psf_nc_inner_params) if args.psf_nc_inner_params else None
        psf_seg_inner = shlex.split(args.psf_seg_inner_params) if args.psf_seg_inner_params else None
        psf_nc_outer = shlex.split(args.psf_nc_outer_params) if args.psf_nc_outer_params else None
        psf_seg_outer = shlex.split(args.psf_seg_outer_params) if args.psf_seg_outer_params else None

        builder = PSFBuilder(
            filters=filters,
            directorio=input_dir,
            select_parts=args.psf_select_parts,
            parts=parts,
            min_dist=min_dist,
            norm_radii=norm_radii,
            width_image=width_image,
            selection_radii_arcmin=selection_radii,
            # Branch selection:
            branch_mag_min=args.psf_branch_mag_min,
            branch_mag_max=args.psf_branch_mag_max,
            # Masking params for stamps:
            nc_inner_args=psf_nc_inner,
            seg_inner_args=psf_seg_inner,
            nc_outer_args=psf_nc_outer,
            seg_outer_args=psf_seg_outer,
            # NEW: center-refinement params:
            center_sigma=args.psf_center_sigma,
            center_max_iter=args.psf_center_max_iter,
        )
        builder.build()


if __name__ == "__main__":
    main()
