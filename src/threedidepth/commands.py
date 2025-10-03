# -*- coding: utf-8 -*-

import argparse

from osgeo import gdal

from threedidepth.calculate import calculate_waterdepth
from threedidepth.calculate import calculate_water_quality
from threedidepth.calculate import MODE_CONSTANT
from threedidepth.calculate import MODE_CONSTANT_VAR
from threedidepth.calculate import MODE_LIZARD
from threedidepth.calculate import MODE_LIZARD_VAR

# keep old constants for backwards compatibility
from threedidepth.calculate import MODE_CONSTANT_S1  # noqa
from threedidepth.calculate import MODE_LIZARD_S1  # noqa


def threedidepth(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gridadmin_path", metavar="gridadmin", help="path to gridadmin.h5 file"
    )
    parser.add_argument(
        "results_3di_path",
        metavar="results_3di",
        help="path to (aggregate_)results_3di.nc file",
    )
    parser.add_argument(
        "dem_path", metavar="dem", help="path to bathymetry file"
    )
    parser.add_argument(
        "waterdepth_path",
        metavar="waterdepth",
        help="path to resulting geotiff"
    )
    calculation_type_group = parser.add_mutually_exclusive_group()
    calculation_type_group.add_argument(
        "-s",
        "--steps",
        nargs="+",
        type=int,
        dest="calculation_steps",
        help="simulation result step(s)",
    )
    calculation_type_group.add_argument(
        "--maximum",
        action="store_true",
        dest="calculate_maximum_waterlevel",
        help="calculate maximum waterlevel instead of waterlevel per timestep",
    )
    parser.add_argument(
        "-c",
        "--constant",
        action="store_true",
        help="disable interpolation and use constant waterlevel per grid cell",
    )
    parser.add_argument(
        "-p",
        "--progress",
        action="store_const",
        dest="progress_func",
        const=gdal.TermProgress_nocb,
        help="Show progress.",
    )
    parser.add_argument(
        "-n",
        "--netcdf",
        action="store_true",
        help="export the waterdepth as a netcdf"
    )
    kwargs = vars(parser.parse_args())
    if kwargs.pop("constant"):
        kwargs["mode"] = MODE_CONSTANT
    else:
        kwargs["mode"] = MODE_LIZARD
    calculate_waterdepth(**kwargs)


def threediwq(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gridadmin_path", metavar="gridadmin", help="path to gridadmin.h5 file"
    )
    parser.add_argument(
        "water_quality_results_3di_path",
        metavar="water_quality_results_3di",
        help="path to water_quality_results_3di.nc file",
    )
    parser.add_argument(
        "variable",
        type=str,
        help="Substance variable in netcdf",
    )
    for output_extent_element in ("xmin", "ymin", "xmax", "ymax"):
        parser.add_argument(
            "output_extent",
            metavar=output_extent_element,
            type=int,
            action="append",
            help=f"{output_extent_element} of output extent",
        )
    parser.add_argument(
        "output_path",
        help="path to resulting geotiff"
    )
    calculation_type_group = parser.add_mutually_exclusive_group()
    calculation_type_group.add_argument(
        "-s",
        "--steps",
        nargs="+",
        type=int,
        metavar="STEP",
        dest="calculation_steps",
        help="simulation result step(s)",
    )
    calculation_type_group.add_argument(
        "--maximum",
        action="store_true",
        dest="calculate_maximum_concentration",
        help=("calculate maximum concentration instead of"
              "concentration per timestep"),
    )
    parser.add_argument(
        "-c",
        "--constant",
        action="store_true",
        help=("disable interpolation and use "
              "constant concentration per grid cell"),
    )
    parser.add_argument(
        "-p",
        "--progress",
        action="store_const",
        dest="progress_func",
        const=gdal.TermProgress_nocb,
        help="Show progress.",
    )
    parser.add_argument(
        "-n",
        "--netcdf",
        action="store_true",
        help="export the concentration as a netcdf"
    )
    kwargs = vars(parser.parse_args())
    if kwargs.pop("constant"):
        kwargs["mode"] = MODE_CONSTANT_VAR
    else:
        kwargs["mode"] = MODE_LIZARD_VAR
    calculate_water_quality(**kwargs)
