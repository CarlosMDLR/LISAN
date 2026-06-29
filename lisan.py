"""
Main LISAN pipeline script.

Reads parameters from CLI (argparse) and runs selected pipeline stages.
"""

import sys
import argparse
import shlex
from pathlib import Path

__version__ = "0.8.0-beta.1"
__release_tag__ = f"v{__version__}"

modules_folder = "./modules"
modules_path = str(Path(modules_folder).resolve())
if modules_path not in sys.path:
    sys.path.append(modules_path)

from measure_depth import calculate_depth
from masking import make_masks
from making_catalogs import make_catalogs
from psf_builder import PSFBuilder
from psf_joint import main as psf_joint_main


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"LISAN {__release_tag__}: masking, depth measurement, catalog construction, PSF building, and PSF joining.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    global_group = parser.add_argument_group("Global input parameters")
    global_group.add_argument("--version", action="version", version=f"LISAN {__release_tag__}")
    global_group.add_argument("--dir", required=True, help="Directory with FITS cutouts.")
    global_group.add_argument("--filters", required=True, help="Comma-separated filters, e.g. g,r,i,lum.")
    global_group.add_argument(
        "--gals-to-use",
        default=None,
        help="Comma-separated galaxy names to process in selected stages, e.g. PGC10074,NGC1037.",
    )

    masking_group = parser.add_argument_group("Masking stage")
    masking_group.add_argument("--make-masks", action="store_true", help="Run the masking stage.")
    masking_group.add_argument("--mask-output-dir", default="Process_data/Mask_data")
    masking_group.add_argument("--noisechisel-params", type=str, default=None)
    masking_group.add_argument("--segment-params", type=str, default=None)

    depth_group = parser.add_argument_group("Depth measurement stage")
    depth_group.add_argument("--measure-depth", action="store_true", help="Run the depth-measurement stage.")
    depth_group.add_argument("--depth-output-dir", default="Process_data/Measure_Depth")
    depth_group.add_argument("--depth-fits", default="depths.fits")
    depth_group.add_argument("--depth-zeropoint", type=float, default=22.5)
    depth_group.add_argument("--depth-sfmagarea", type=int, default=100)
    depth_group.add_argument("--depth-sfmagnsigma", type=float, default=3.0)
    depth_group.add_argument("--depth-upnsigma", type=float, default=3.0)
    depth_group.add_argument("--depth-noisechisel-params", type=str, default=None)

    catalogs_group = parser.add_argument_group("Catalog construction stage")
    catalogs_group.add_argument("--make-catalogs", action="store_true", help="Run the catalog-construction stage.")
    catalogs_group.add_argument("--cats-output-dir", default="Process_data/Make_catalogs")
    catalogs_group.add_argument("--cats-zeropoint", type=float, default=22.5)
    catalogs_group.add_argument("--cats-noisechisel-params", type=str, default=None)
    catalogs_group.add_argument("--cats-segment-params", type=str, default=None)
    catalogs_group.add_argument("--cats-aperture-arcsec", type=float, default=10.0)
    catalogs_group.add_argument("--cats-gaia-dataset", default="dr3")
    catalogs_group.add_argument("--cats-gaia-sigma", type=float, default=3.0)
    catalogs_group.add_argument("--cats-no-plots", action="store_true")

    psf_group = parser.add_argument_group("PSF building stage")
    psf_group.add_argument("--build-psf", action="store_true", help="Run the PSF-building stage.")
    psf_group.add_argument("--psf-select-parts", default="I", help="I (inner), O (outer), B (both).")
    psf_group.add_argument("--psf-parts", default="A,B,C", help="Inner PSF part labels.")
    psf_group.add_argument("--psf-min-dist", default="0.015,0.015,0.015")
    psf_group.add_argument("--psf-norm-radii", default="5,10;10,20;20,40")
    psf_group.add_argument("--psf-width-image", default="200,200;400,400;800,800")
    psf_group.add_argument("--psf-selection-radii", default="90,90,90")
    psf_group.add_argument("--psf-center-sigma", type=float, default=5.0)
    psf_group.add_argument("--psf-center-max-iter", type=int, default=5)
    psf_group.add_argument("--psf-branch-mag-min", type=float, default=16.0)
    psf_group.add_argument("--psf-branch-mag-max", type=float, default=18.0)

    psf_mask_group = parser.add_argument_group("PSF stamp masking parameters")
    psf_mask_group.add_argument("--psf-nc-inner-params", type=str, default=None)
    psf_mask_group.add_argument("--psf-seg-inner-params", type=str, default=None)
    psf_mask_group.add_argument("--psf-nc-outer-params", type=str, default=None)
    psf_mask_group.add_argument("--psf-seg-outer-params", type=str, default=None)

    psf_outer_group = parser.add_argument_group("PSF outer-part configuration")
    psf_outer_group.add_argument("--psf-min-dist-outer", default="0.015")
    psf_outer_group.add_argument("--psf-norm-radii-outer", default="40,80")
    psf_outer_group.add_argument("--psf-width-image-outer", default="1600,1600")
    psf_outer_group.add_argument("--psf-step-min-dist-outer", type=float, default=0.0)
    psf_outer_group.add_argument("--psf-step-norm-radii-outer", default="0,0")
    psf_outer_group.add_argument("--psf-step-width-image-outer", default="0,0")

    psf_join_group = parser.add_argument_group("PSF joining stage")
    psf_join_group.add_argument("--join-psf", action="store_true", help="Run the interactive PSF-joining stage.")
    psf_join_group.add_argument("--join-inner-root", default="./PSF_files/Inner_parts")
    psf_join_group.add_argument("--join-outer-root", default="./PSF_files/Outer_parts")
    psf_join_group.add_argument("--join-external-outer-stack", default=None)
    psf_join_group.add_argument("--join-external-outer-profile", default=None)
    psf_join_group.add_argument("--join-circular-radius", type=int, default=None)

    return parser.parse_args()


def _parse_filters(filters: str) -> list[str]:
    return [f.strip() for f in filters.split(",") if f.strip()]


def _parse_list_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_pairs_semicolon(s: str) -> list[str]:
    return [x.strip() for x in s.split(";") if x.strip()]


def _parse_float_pair_csv(s: str) -> tuple[float, float]:
    a, b = [x.strip() for x in s.split(",")]
    return float(a), float(b)


def main():
    args = parse_args()

    input_dir = Path(args.dir)
    filters = _parse_filters(args.filters)
    gals_to_use = _parse_list_csv(args.gals_to_use) if args.gals_to_use else None

    if args.make_masks:
        nc_args = shlex.split(args.noisechisel_params) if args.noisechisel_params else None
        seg_args = shlex.split(args.segment_params) if args.segment_params else None
        make_masks(
            input_dir,
            output_dir=Path(args.mask_output_dir),
            noisechisel_args=nc_args,
            segment_args=seg_args,
        )

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

    if args.build_psf:
        parts = _parse_list_csv(args.psf_parts)
        min_dist = _parse_list_csv(args.psf_min_dist)
        norm_radii = _parse_pairs_semicolon(args.psf_norm_radii)
        width_image = _parse_pairs_semicolon(args.psf_width_image)
        selection_radii = [float(x) for x in _parse_list_csv(args.psf_selection_radii)]

        min_dist_outer = _parse_list_csv(args.psf_min_dist_outer)
        norm_radii_outer = _parse_pairs_semicolon(args.psf_norm_radii_outer)
        width_image_outer = _parse_pairs_semicolon(args.psf_width_image_outer)

        step_norm_radii_outer = _parse_float_pair_csv(args.psf_step_norm_radii_outer)
        step_width_image_outer = _parse_float_pair_csv(args.psf_step_width_image_outer)

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
            branch_mag_min=args.psf_branch_mag_min,
            branch_mag_max=args.psf_branch_mag_max,
            nc_inner_args=psf_nc_inner,
            seg_inner_args=psf_seg_inner,
            nc_outer_args=psf_nc_outer,
            seg_outer_args=psf_seg_outer,
            center_sigma=args.psf_center_sigma,
            center_max_iter=args.psf_center_max_iter,
            min_dist_outer=min_dist_outer,
            norm_radii_outer=norm_radii_outer,
            width_image_outer=width_image_outer,
            step_min_dist_outer=args.psf_step_min_dist_outer,
            step_norm_radii_outer=step_norm_radii_outer,
            step_width_image_outer=step_width_image_outer,
        )
        builder.build()

    if args.join_psf:
        if not filters:
            raise SystemExit("--filters is required when using --join-psf")

        fits_files = sorted([
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".fits"
        ])

        if not fits_files:
            raise SystemExit(f"No FITS files found in {input_dir}")

        for image in fits_files:
            name_parts = image.stem.split("_")

            if len(name_parts) < 2:
                print(f"[skip] Cannot infer galaxy/filter from filename: {image.name}")
                continue

            gal = name_parts[0]
            flt = name_parts[1]

            if gals_to_use is not None and gal not in gals_to_use:
                continue

            if flt not in filters:
                continue

            print("\n" + 60 * "=")
            print(f"Joining PSF for galaxy={gal}, filter={flt}")
            print(60 * "=")

            argv = [
                "psf_joint.py",
                "--dir", str(gal),
                "--filter", str(flt),
                "--inner-root", str(args.join_inner_root),
                "--outer-root", str(args.join_outer_root),
            ]

            if args.join_external_outer_stack:
                argv += ["--external-outer-stack", str(args.join_external_outer_stack)]

            if args.join_external_outer_profile:
                argv += ["--external-outer-profile", str(args.join_external_outer_profile)]

            if args.join_circular_radius is not None:
                argv += ["--circular-radius", str(int(args.join_circular_radius))]
                argv += ["--export-circular"]

            old_argv = sys.argv
            try:
                sys.argv = argv
                psf_joint_main()
            finally:
                sys.argv = old_argv


if __name__ == "__main__":
    main()