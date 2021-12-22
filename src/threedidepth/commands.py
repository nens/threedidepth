# -*- coding: utf-8 -*-

import argparse
import collections
import sys

from osgeo import gdal

from threedidepth.calculate import calculate_waterdepth
from threedidepth.calculate import MODE_LIZARD
from threedidepth.calculate import MODE_LIZARD_S1
from threedidepth.calculate import MODE_CONSTANT
from threedidepth.calculate import MODE_CONSTANT_S1
from threedidepth.calculate import MODE_BILINEAR
from threedidepth.calculate import MODE_BILINEAR_S1


Choice = collections.namedtuple("Choice", ["c", "w", "b"])

MODE = {
    Choice(b=False, c=False, w=False): MODE_LIZARD,
    Choice(b=False, c=False, w=True): MODE_LIZARD_S1,
    Choice(b=False, c=True, w=False): MODE_CONSTANT,
    Choice(b=False, c=True, w=True): MODE_CONSTANT_S1,
    Choice(b=True, c=False, w=False): MODE_BILINEAR,
    Choice(b=True, c=False, w=True): MODE_BILINEAR_S1,
    Choice(b=True, c=True, w=False): MODE_CONSTANT,
    Choice(b=True, c=True, w=True): MODE_CONSTANT_S1,
}


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
    parser.add_argument(
        "-s",
        "--steps",
        nargs="+",
        type=int,
        dest="calculation_steps",
        help="simulation result step(s)",
    )
    parser.add_argument(
        "-c",
        "--constant",
        action="store_true",
        help="disable interpolation and use constant waterlevel per grid cell",
    )
    parser.add_argument(
        "-w",
        "--waterlevel",
        action="store_true",
        help="export the waterlevel instead of the waterdepth"
    )
    parser.add_argument(
        "-b",
        "--bilinear",
        action="store_true",
        help="When using interplation, use the bilinear method."
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        dest="threshold",
        help="exponent of 10 for the au threhsold; e.g. '-4' gives 1e-4.",
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

    threshold = kwargs.pop("threshold")
    if threshold is not None:
        kwargs["threshold"] = float('1e' + str(threshold))

    kwargs["mode"] = MODE[Choice(
        b=kwargs.pop("bilinear"),
        c=kwargs.pop("constant"),
        w=kwargs.pop("waterlevel"),
    )]
    calculate_waterdepth(**kwargs)


if __name__ == '__main__':
    threedidepth(sys.argv)
