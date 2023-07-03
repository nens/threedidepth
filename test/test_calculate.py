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

from threedidepth.calculate import BaseCalculator
from threedidepth.calculate import GeoTIFFConverter
from threedidepth.calculate import MODE_CONSTANT
from threedidepth.calculate import MODE_CONSTANT_S1
from threedidepth.calculate import MODE_COPY
from threedidepth.calculate import MODE_LINEAR
from threedidepth.calculate import MODE_LINEAR_S1
from threedidepth.calculate import MODE_LIZARD
from threedidepth.calculate import MODE_LIZARD_S1
from threedidepth.calculate import MODE_NODGRID
from threedidepth.calculate import NetcdfConverter
from threedidepth.calculate import ProgressClass
from threedidepth.calculate import SUBSET_2D_OPEN_WATER
from threedidepth.calculate import calculate_waterdepth
from threedidepth.calculate import calculator_classes

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
    result_admin = mock.Mock()
    import_path = "threedidepth.calculate.ResultAdmin"
    with mock.patch(import_path) as ResultAdmin:
        ResultAdmin.return_value = result_admin
        yield result_admin


def test_progress_class():
    progress_func = mock.Mock()
    calculation_steps = [10, 20]
    progress_class = ProgressClass(
        calculation_steps=[10, 20],
        progress_func=progress_func,
    )
    for i, calculation_step in progress_class:
        assert calculation_step == calculation_steps[i]
        progress_class(0.5)
        assert progress_func.called_with(i + 0.25)
        progress_class(1.0)
        assert progress_func.called_with(i + 0.50)


@mark.parametrize("source_path", [False, True], indirect=True)
def test_geotiff_converter(source_path, target_path):
    progress_func = mock.Mock()
    band_count = 3
    converter_kwargs = {
        "source_path": source_path,
        "target_path": target_path,
        "band_count": band_count,
        "progress_func": progress_func,
    }

    def calculator(indices, values, no_data_value):
        """Return input values unmodified."""
        return values

    with GeoTIFFConverter(**converter_kwargs) as converter:
        converter.convert_using(calculator=calculator, band=0)

        # check progess func calls
        progress_func_values = [r[0][0] for r in progress_func.call_args_list]
        assert len(progress_func_values) == len(converter)
        assert progress_func_values[0] < 1
        assert progress_func_values[-1] == 1
        assert progress_func_values == sorted(progress_func_values)

        # convert another band
        converter.convert_using(calculator=calculator, band=2)

    # check target
    source = gdal.Open(source_path)
    source_array = source.ReadAsArray()
    source_band = source.GetRasterBand(1)
    source_no_data_value = source_band.GetNoDataValue()
    source_block_size = source_band.GetBlockSize()

    target = gdal.Open(target_path)
    target_array = target.ReadAsArray()
    target_bands = [target.GetRasterBand(i + 1) for i in range(band_count)]
    target_no_data_values = [b.GetNoDataValue() for b in target_bands]
    target_block_sizes = [b.GetBlockSize() for b in target_bands]

    assert source.GetGeoTransform() == target.GetGeoTransform()
    assert source.GetProjection() == target.GetProjection()
    assert all(source_block_size == v for v in target_block_sizes)
    assert all(source_no_data_value == v for v in target_no_data_values)
    assert len(target_array) == band_count
    assert np.equal(source_array, target_array[[0, 2]]).all()
    assert np.equal(0, target_array[[1]]).all()


def test_tiff_converter_existing_target(tmpdir):
    target_path = tmpdir.join("target.tif")
    target_path.ensure(file=True)  # "touch" the file
    with raises(OSError, match="exists"):
        GeoTIFFConverter(
            source_path=None, target_path=target_path, progress_func=None)


@mark.parametrize("source_path", [False, True], indirect=True)
def test_netcdf_converter(source_path, tmp_path, admin):
    admin.result_type = "raw"
    admin.get_timestamps.return_value = np.array([1, -1])
    admin.get_time_units.return_value = b'seconds since 2021-03-02 10:00:00'
    admin.model_slug = "my_model_slug"
    admin.epsg_code = "28992"
    admin.variable = "s1"
    target_path = str(tmp_path / "waterdepth.nc")

    progress_func = mock.Mock()
    converter_kwargs = {
        "source_path": source_path,
        "target_path": target_path,
        "result_admin": admin,
        "calculation_steps": [1, -1],
        "progress_func": progress_func,
    }

    def calculator(indices, values, no_data_value):
        """Return input values unmodified."""
        return values

    with NetcdfConverter(**converter_kwargs) as converter:
        converter.convert_using(calculator, band=0)
        converter.convert_using(calculator, band=1)

        assert 2 * len(converter) == len(progress_func.call_args_list)
        assert progress_func.call_args_list[0][0][0] < 1
        assert progress_func.call_args_list[-1][0][0] == 1

    source = gdal.Open(source_path)
    target = gdal.Open(target_path)

    assert np.equal(source.ReadAsArray(), target.ReadAsArray()).all()
    assert source.GetGeoTransform() == target.GetGeoTransform()


def test_calculate_waterdepth_wrong_mode():
    with raises(ValueError, match="Unknown mode:"):
        calculate_waterdepth(
            gridadmin_path="dummy",
            results_3di_path='dummy',
            dem_path="dummy",
            waterdepth_path="dummy",
            mode="wrong",
        )


def test_calculate_waterdepth_wrong_step(admin):
    admin.calculation_steps = 2
    with raises(AssertionError, match="Maximum calculation step is"):
        calculate_waterdepth(
            gridadmin_path="dummy",
            results_3di_path='dummy',
            dem_path="dummy",
            waterdepth_path="dummy",
            calculation_steps=[3],
            mode=MODE_COPY,
        )


def test_calculate_waterdepth(source_path, target_path, admin):
    admin.calculation_steps = 2
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
    depth = BaseCalculator._depth_from_water_level(
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
    admin.variable = "s1"
    if mode in (MODE_CONSTANT_S1):
        data = {"id": ids, "s1": s1}
        admin.nodes.subset().timeseries().only().data = data
        admin.nodes.subset().only().data = data
    if mode in (MODE_LINEAR_S1):
        data = {"id": ids, "coordinates": coordinates, "s1": s1}
        admin.nodes.subset().timeseries().only().data = data
        admin.nodes.subset().only().data = data
    if mode in (MODE_LIZARD_S1):
        admin.nodes.subset().timeseries().only.side_effect = [
            mock.Mock(data={"id": ids, "coordinates": coordinates, "s1": s1})
        ]
        admin.nodes.subset().only.side_effect = [
            mock.Mock(data={"id": ids, "coordinates": coordinates, "s1": s1})
        ]

    indexes = slice(7, 8)
    calculator_kwargs = {
        "result_admin": admin,
        "calculation_step": indexes.start,
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
        admin.nodes.subset().timeseries.assert_called_with(indexes=indexes)
        admin.nodes.subset().timeseries().only.assert_called_with(
            "s1", "id",
        )
    if mode in (MODE_CONSTANT_S1):
        admin.nodes.subset.assert_called_with(SUBSET_2D_OPEN_WATER)
        admin.nodes.subset().timeseries.assert_called_with(indexes=indexes)
        admin.nodes.subset().timeseries().only.assert_called_with(
            "s1", "id",
        )
    if mode in (MODE_LIZARD_S1):
        admin.nodes.subset.assert_called_with(SUBSET_2D_OPEN_WATER)
        admin.nodes.subset().timeseries.assert_called_with(indexes=indexes)
        admin.nodes.subset().timeseries().only.assert_has_calls(
            [mock.call("s1", "id")]
        )


@fixture
def depthmock():
    import_path = (
        "threedidepth.calculate.BaseCalculator._depth_from_water_level"
    )
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
        calculator = CalculatorClass("", "", "", "")
        depth = calculator(1, 2, 3)

    callmock.assert_called_once_with(1, 2, 3)
    depthmock.assert_called_once_with(dem=2, fillvalue=3, waterlevel=4)
    assert depth == 5


def test_calculator_not_implemented():
    with raises(NotImplementedError):
        BaseCalculator("", "", "", "")("", "", "")
