# -*- coding: utf-8 -*-

from contextlib import contextmanager
from unittest import mock
import random
import string
import sys

from numpy.testing import assert_almost_equal
from osgeo import gdal
from osgeo import osr
from pytest import fixture
from pytest import mark
from pytest import raises
import numpy as np

from threedidepth.calculate import Calculator
from threedidepth.calculate import GeoTIFFConverter
from threedidepth.calculate import calculate_waterdepth
from threedidepth.calculate import calculator_classes
from threedidepth.calculate import MODE_COPY
from threedidepth.calculate import MODE_NODGRID
from threedidepth.calculate import MODE_CONSTANT_S1
from threedidepth.calculate import MODE_INTERPOLATED_S1
from threedidepth.calculate import MODE_CONSTANT
from threedidepth.calculate import MODE_INTERPOLATED
from threedidepth.calculate import SUBSET_2D_OPEN_WATER
from threedidepth.commands import threedidepth

RD = osr.GetUserInputAsWKT("EPSG:28992")
NDV = -9  # no data value of the test dem


def random_slug(size):
    """Return a random slug of length `size`.

    The returned slug consists of digits and lowercase letters.
    """
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(size))


@contextmanager
def vsimem_path():
    """ An autoremoving temporary path in vsimem. """
    while True:
        vsimem_path = "/vsimem/" + random_slug(6)
        if gdal.VSIStatL(vsimem_path) is None:
            break
    yield vsimem_path
    gdal.Unlink(vsimem_path)


@fixture
def target_path():
    with vsimem_path() as path:
        yield path


@fixture(scope="module")
def source_path(request):
    """Provides a GeoTIFF file in the vsimem virtual filesystem.

    Can be parametrized to add the creation option "tiled=yes".
    """
    bands, height, width, data_type = 1, 128, 512, gdal.GDT_Float32
    array = np.arange(height * width).reshape(height, width)

    source = gdal.GetDriverByName("mem").Create(
        "", width, height, bands, data_type,
    )
    source.SetGeoTransform((0, 1, 0, 128, 0, -1))
    source.SetProjection(RD)
    source_band = source.GetRasterBand(1)
    source_band.SetNoDataValue(-9)
    source_band.WriteArray(array)

    options = ["compress=deflate"]
    if getattr(request, "param", False):
        options.append("tiled=yes")

    with vsimem_path() as path:
        gdal.GetDriverByName("gtiff").CreateCopy(path, source, options=options)
        yield path


@fixture
def admin():
    grid_h5_result_admin = mock.Mock()
    import_path = "threedidepth.calculate.GridH5ResultAdmin"
    with mock.patch(import_path) as GridH5ResultAdmin:
        GridH5ResultAdmin.return_value = grid_h5_result_admin
        yield grid_h5_result_admin


@mark.parametrize("source_path", [False, True], indirect=True)
def test_tiff_converter(source_path, target_path):
    progress_func = mock.Mock()
    converter_kwargs = {
        "source_path": source_path,
        "target_path": target_path,
        "progress_func": progress_func,
    }

    def calculator(indices, values, no_data_value):
        """Return input values unmodified."""
        return values

    with GeoTIFFConverter(**converter_kwargs) as converter:
        converter.convert_using(calculator)

        assert len(converter) == len(progress_func.call_args_list)
        assert progress_func.call_args_list[0][0][0] < 1
        assert progress_func.call_args_list[-1][0][0] == 1

    source = gdal.Open(source_path)
    source_band = source.GetRasterBand(1)
    target = gdal.Open(target_path)
    target_band = target.GetRasterBand(1)

    assert np.equal(source.ReadAsArray(), target.ReadAsArray()).all()
    assert source.GetGeoTransform() == target.GetGeoTransform()
    assert source.GetProjection() == target.GetProjection()
    assert source_band.GetNoDataValue() == target_band.GetNoDataValue()
    assert source_band.GetBlockSize() == target_band.GetBlockSize()


def test_tiff_converter_existing_target(tmpdir):
    target_path = tmpdir.join("target.tif")
    target_path.ensure(file=True)  # "touch" the file
    with raises(OSError, match="exists"):
        GeoTIFFConverter(
            source_path=None, target_path=target_path, progress_func=None)


def test_calculate_waterdepth_wrong_mode():
    with raises(ValueError, match="ode"):
        calculate_waterdepth(
            gridadmin_path="dummy",
            results_3di_path='dummy',
            dem_path="dummy",
            waterdepth_path="dummy",
            mode="wrong",
        )


def test_command(tmpdir):
    depth_path = tmpdir.join("waterdepth.tif")
    depth_path.ensure(file=True)  # "touch" the file
    with mock.patch("threedidepth.commands.calculate_waterdepth") as wd:
        args = ["threedidepth,", "a", "b", "c", "d"]
        with mock.patch.object(sys, "argv", args):
            threedidepth()
        assert wd.called_with(
            gridadmin_path="a",
            resultes_3di_path="b",
            dem_path="c",
            waterdepth_path="d",
            calculation_step=-1,
            mode=MODE_INTERPOLATED,
        )
        args.append("--constant")
        with mock.patch.object(sys, "argv", args):
            threedidepth()
        assert wd.called_with(
            gridadmin_path="a",
            resultes_3di_path="b",
            dem_path="c",
            waterdepth_path="d",
            calculation_step=-1,
            mode=MODE_CONSTANT,
        )


def test_calculate_waterdepth(source_path, target_path, admin):
    with mock.patch("threedidepth.calculate.fix_gridadmin"):
        calculate_waterdepth(
            gridadmin_path="dummy",
            results_3di_path='dummy',
            dem_path=source_path,
            waterdepth_path=target_path,
            mode=MODE_COPY,
        )


data = (
    (MODE_COPY, np.array([[NDV, 2.0], [2.0, 2.0]])),
    (MODE_NODGRID, np.array([[6, 7], [8, 9]])),
    (MODE_CONSTANT_S1, np.array([[2.3, 2.2], [2.1, 2.0]])),
    (MODE_INTERPOLATED_S1, np.array([[2.3, 2.2], [2.1, 2.1]])),
    (MODE_CONSTANT, np.array([[NDV, 0.2], [0.1, NDV]])),
    (MODE_INTERPOLATED, np.array([[NDV, 0.2], [0.1, 0.1]])),
)


@mark.parametrize("mode, expected", data)
def test_calculators(mode, expected, admin):
    # indices and geo_transform make the values lie in a (6, 4), (10, 8) bbox
    indices = ((1, 2), (3, 4))
    geo_transform = (2, 2, 0, 10, 0, -2)
    nodgrid = np.array([
        [1, 2, 3, 4],
        [5, 5, 6, 7],
        [5, 5, 8, 9],
    ])
    s1 = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 2.3, 2.2, 2.1, 2.0]])
    ids = np.arange(1, 10)
    coordinates = np.array([
        [3, 5, 7, 9, 4, 7, 9, 7, 9],
        [9, 9, 9, 9, 6, 7, 7, 5, 3],  # last node pulled down one cell
    ])

    # dem values
    values = np.array([
        [NDV, 2.0],
        [2.0, 2.0],
    ])

    # prepare gridadmin uses
    if mode in (MODE_NODGRID, MODE_CONSTANT_S1, MODE_CONSTANT):
        admin.cells.get_nodgrid.return_value = nodgrid[1:3, 2:4]
    if mode in (MODE_INTERPOLATED_S1, MODE_INTERPOLATED):
        data = {"coordinates": coordinates, "s1": s1}
        admin.nodes.subset().timeseries().only().data = data
    if mode in (MODE_CONSTANT_S1, MODE_CONSTANT):
        data = {"id": ids, "s1": s1}
        admin.nodes.subset().timeseries().only().data = data

    calculator_kwargs = {
        "gridadmin_path": "dummy",
        "results_3di_path": "dummy",
        "calculation_step": -1,
        "dem_shape": (3, 4),
        "dem_geo_transform": geo_transform,
    }

    call_kwargs = {
        "no_data_value": NDV,
        "values": values,
        "indices": indices,
    }

    CalculatorClass = calculator_classes[mode]
    with CalculatorClass(**calculator_kwargs) as calculator:
        result = calculator(**call_kwargs)
    assert_almost_equal(result, expected)

    # check gridadmin uses
    if mode in (MODE_NODGRID, MODE_CONSTANT_S1, MODE_CONSTANT):
        # get_nodgrid() wants (1, 2), (3, 4) flipped & transposed
        admin.cells.get_nodgrid.assert_called_once_with(
            [2, 0, 4, 2],
            subset_name=SUBSET_2D_OPEN_WATER,
        )
    if mode in (MODE_INTERPOLATED_S1, MODE_INTERPOLATED):
        admin.nodes.subset.assert_called_with(SUBSET_2D_OPEN_WATER)
        admin.nodes.subset().timeseries.assert_called_with(indexes=[-1])
        admin.nodes.subset().timeseries().only.assert_called_with(
            "s1", "coordinates",
        )
    if mode in (MODE_CONSTANT_S1, MODE_CONSTANT):
        admin.nodes.subset.assert_called_with(SUBSET_2D_OPEN_WATER)
        admin.nodes.subset().timeseries.assert_called_with(indexes=[-1])
        admin.nodes.subset().timeseries().only.assert_called_with(
            "s1", "id",
        )


def test_calculator_not_implemented():
    with raises(NotImplementedError):
        Calculator("", "", "", "", "")("", "", "")
