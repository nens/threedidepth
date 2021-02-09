# -*- coding: utf-8 -*-

from contextlib import contextmanager
from unittest import mock
import random
import string

from numpy.testing import assert_allclose
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
from threedidepth.calculate import MODE_LINEAR_S1
from threedidepth.calculate import MODE_LIZARD_S1
from threedidepth.calculate import MODE_CONSTANT
from threedidepth.calculate import MODE_LINEAR
from threedidepth.calculate import MODE_LIZARD
from threedidepth.calculate import SUBSET_2D_OPEN_WATER

RD = osr.GetUserInputAsWKT("EPSG:28992")
NDV = -9  # no data value of the test dem
NLV = -9999.  # "no level value"


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


def test_calculate_waterdepth(source_path, target_path, admin):
    with mock.patch("threedidepth.calculate.fix_gridadmin"):
        calculate_waterdepth(
            gridadmin_path="dummy",
            results_3di_path='dummy',
            dem_path=source_path,
            waterdepth_path=target_path,
            mode=MODE_COPY,
        )


def test_depth_from_water_level():
    dem = np.array([[7.0, 2.0], [3.0, 4.0]])
    fillvalue = 7.0
    waterlevel = np.array([[4.0, NDV], [4.0, 4.0]])
    depth = Calculator._depth_from_water_level(
        dem=dem, fillvalue=fillvalue, waterlevel=waterlevel,
    )
    assert depth[0, 0] == fillvalue  # because dem is fillvalue
    assert depth[0, 1] == fillvalue  # because waterlevel is no data
    assert depth[1, 0] == 1.0        # really, a depth
    assert depth[1, 1] == fillvalue  # because depth <= 0


# some interpolated values as expected results for the linear mode
EL1 = 4.0 / 4 + NLV * 3 / 4
EL2 = 6.0 / 3 + NLV * 2 / 3
EL3 = 5.0 / 2 + NLV / 4 + 4.0 / 4
EL4 = 6.0 / 3 + NLV / 6 + 5.0 / 2
data = (
    (
        MODE_COPY,
        np.array([[6.0, 6.0, 6.0, 6.0],
                  [6.0, 6.0, 6.0, 6.0],
                  [6.0, 6.0, 6.0, NDV]]),
    ),
    (
        MODE_NODGRID,
        np.array([[2, 2, 3, 3],
                  [0, 5, 6, 7],
                  [5, 5, 8, 9]]),
    ),
    (
        MODE_CONSTANT_S1,
        np.array([[NLV, NLV, NLV, NLV],
                  [NLV, 5.0, 6.0, 7.0],
                  [5.0, 5.0, 8.0, 9.0]]),
    ),
    (
        MODE_LINEAR_S1,
        np.array([[EL1, EL2, EL2, NLV],
                  [EL3, EL4, 6.0, 7.0],
                  [NLV, NLV, 8.0, 9.0]]),
    ),
    (
        MODE_LIZARD_S1,
        np.array([[NLV, NLV, NLV, NLV],
                  [NLV, 5.4, 6.0, 7.0],
                  [5.0, 5.0, 8.0, 9.0]]),
    ),
)


@mark.parametrize("mode, expected", data)
def test_calculators(mode, expected, admin):
    nodgrid = np.array([
        [1, 1, 2, 2, 3, 3],
        [1, 1, 2, 2, 3, 3],
        [4, 4, 0, 5, 6, 7],
        [4, 4, 5, 5, 8, 9],
    ])  # note that 0 means no grid cell at all in threedigrid
    geo_transform = (20, 2, 0, 48, 0, -2)  # bbox: (20, 40, 32, 48)
    s1 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
    s1[:, :3] = -9999.0
    ids = np.arange(1, 10)
    coordinates = np.array([
        [22, 26, 30, 22, 26, 29, 31, 29, 31],
        [46, 46, 46, 42, 42, 43, 43, 41, 41],
    ])

    # request indices and selection of nodgrid
    indices = ((1, 2), (4, 6))  # request bbox (24, 42), (32, 48)

    # dem values
    values = np.array([
        [6.0, 6.0, 6.0, 6.0],
        [6.0, 6.0, 6.0, 6.0],
        [6.0, 6.0, 6.0, NDV],
    ])

    # prepare gridadmin uses
    admin.cells.get_nodgrid.return_value = nodgrid[tuple(map(slice, *indices))]
    if mode in (MODE_CONSTANT_S1):
        data = {"id": ids, "s1": s1}
        admin.nodes.subset().timeseries().only().data = data
    if mode in (MODE_LINEAR_S1):
        data = {"coordinates": coordinates, "s1": s1}
        admin.nodes.subset().timeseries().only().data = data
    if mode in (MODE_LIZARD_S1):
        admin.nodes.subset().timeseries().only.side_effect = [
            mock.Mock(data={"id": ids, "s1": s1}),
            mock.Mock(data={"coordinates": coordinates, "s1": s1}),
        ]

    calculator_kwargs = {
        "gridadmin_path": "dummy",
        "results_3di_path": "dummy",
        "calculation_step": -1,
        "dem_shape": nodgrid.shape,
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
    assert_allclose(result, expected)

    # check gridadmin uses
    if mode in (MODE_NODGRID, MODE_CONSTANT_S1, MODE_LIZARD_S1):
        # get_nodgrid() wants (1, 2), (4, 6) flipped & transposed
        admin.cells.get_nodgrid.assert_called_once_with(
            [2, 0, 6, 3],
            subset_name=SUBSET_2D_OPEN_WATER,
        )
    if mode in (MODE_LINEAR_S1):
        admin.nodes.subset.assert_called_with(SUBSET_2D_OPEN_WATER)
        admin.nodes.subset().timeseries.assert_called_with(indexes=[-1])
        admin.nodes.subset().timeseries().only.assert_called_with(
            "s1", "coordinates",
        )
    if mode in (MODE_CONSTANT_S1):
        admin.nodes.subset.assert_called_with(SUBSET_2D_OPEN_WATER)
        admin.nodes.subset().timeseries.assert_called_with(indexes=[-1])
        admin.nodes.subset().timeseries().only.assert_called_with(
            "s1", "id",
        )
    if mode in (MODE_LIZARD_S1):
        admin.nodes.subset.assert_called_with(SUBSET_2D_OPEN_WATER)
        admin.nodes.subset().timeseries.assert_called_with(indexes=[-1])
        admin.nodes.subset().timeseries().only.assert_has_calls(
            [mock.call("s1", "id"), mock.call("s1", "coordinates")]
        )


@fixture
def depthmock():
    import_path = "threedidepth.calculate.Calculator._depth_from_water_level"
    with mock.patch(import_path) as depthmock:
        yield depthmock


mode = (
    MODE_CONSTANT,
    MODE_LINEAR,
    MODE_LIZARD,
)


@mark.parametrize("mode", mode)
def test_depth_calculators(depthmock, mode):
    CalculatorClass = calculator_classes[mode]
    BaseClass = CalculatorClass.__bases__[0]
    depthmock.return_value = 5
    import_path = "threedidepth.calculate." + BaseClass.__name__ + ".__call__"
    with mock.patch(import_path) as callmock:
        callmock.return_value = 4
        calculator = CalculatorClass("", "", "", "", "")
        depth = calculator(1, 2, 3)

    callmock.assert_called_once_with(1, 2, 3)
    depthmock.assert_called_once_with(dem=2, fillvalue=3, waterlevel=4)
    assert depth == 5


def test_calculator_not_implemented():
    with raises(NotImplementedError):
        Calculator("", "", "", "", "")("", "", "")
