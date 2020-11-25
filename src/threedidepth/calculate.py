from itertools import product
from os import path
import numpy as np
from osgeo import gdal
from threedigrid.admin.gridresultadmin import GridH5ResultAdmin
from threedigrid.admin.gridresultadmin import GridH5Admin
from threedigrid.admin.constants import SUBSET_2D_OPEN_WATER
from threedigrid.admin.constants import NO_DATA_VALUE
from threedidepth.fixes import fix_gridadmin
import time

from threedidepth.base_calculator import Calculator
from threedidepth.au_interpolator import calculator_classes as au_calculator_classes

MODE_COPY = "copy"
MODE_NODGRID = "nodgrid"
MODE_CONSTANT_S1 = "constant-s1"
MODE_INTERPOLATED_S1 = "interpolated-s1"
MODE_CONSTANT = "constant"
MODE_INTERPOLATED = "interpolated"

STEP = "0"


class CopyCalculator(Calculator):
    def __call__(self, indices, values, no_data_value):
        """Return input values unmodified."""
        return values


class NodGridCalculator(Calculator):
    def __call__(self, indices, values, no_data_value):
        """Return node grid."""
        return self._get_nodgrid(indices)


class ConstantLevelCalculator(Calculator):
    def __call__(self, indices, values, no_data_value):
        """Return waterlevel array."""
        return self.lookup_s1[self._get_nodgrid(indices)]


class InterpolatedLevelCalculator(Calculator):
    def __call__(self, indices, values, no_data_value):
        """Return waterlevel array."""
        points = self._get_points(indices)
        return self.interpolator(points).reshape(values.shape)


class ConstantLevelDepthCalculator(ConstantLevelCalculator):
    def __call__(self, indices, values, no_data_value):
        """Return waterdepth array."""
        waterlevel = super().__call__(indices, values, no_data_value)
        return self._depth_from_water_level(
            dem=values, fillvalue=no_data_value, waterlevel=waterlevel
        )


class InterpolatedLevelDepthCalculator(InterpolatedLevelCalculator):
    def __call__(self, indices, values, no_data_value):
        """Return waterdepth array."""
        waterlevel = super().__call__(indices, values, no_data_value)
        return self._depth_from_water_level(
            dem=values, fillvalue=no_data_value, waterlevel=waterlevel
        )


class GeoTIFFConverter:
    """Convert tiff, applying a calculating function to the data.
    Args:
        source_path (str): Path to source GeoTIFF file.
        target_path (str): Path to target GeoTIFF file.
        progress_func: a callable.
        The progress_func will be called multiple times with values between 0.0
        amd 1.0.
    """

    def __init__(
        self,
        gridadmin_path,
        results_3di_path,
        calculation_step,
        source_path,
        target_path,
        progress_func=None,
    ):
        self.source_path = source_path
        self.target_path = target_path

        self.progress_func = progress_func

        self.gridadmin_path = gridadmin_path
        self.results_3di_path = results_3di_path
        self.calculation_step = calculation_step

        if path.exists(self.target_path):
            raise OSError("%s already exists." % self.target_path)

    def __enter__(self):
        """Open datasets."""
        self.source = gdal.Open(self.source_path, gdal.GA_ReadOnly)
        block_x_size, block_y_size = self.block_size
        options = ["compress=deflate", "blockysize=%s" % block_y_size]
        if block_x_size != self.raster_x_size:
            options += ["tiled=yes", "blockxsize=%s" % block_x_size]

        self.target = gdal.GetDriverByName("gtiff").Create(
            self.target_path,
            self.raster_x_size,
            self.raster_y_size,
            1,  # band count
            self.source.GetRasterBand(1).DataType,
            options=options,
        )
        self.target.SetProjection(self.projection)
        self.target.SetGeoTransform(self.geo_transform)
        self.target.GetRasterBand(1).SetNoDataValue(self.no_data_value)

        return self

    def __exit__(self, *args):
        """Close datasets."""
        self.source = None
        self.target = None

    @property
    def projection(self):
        return self.source.GetProjection()

    @property
    def geo_transform(self):
        return self.source.GetGeoTransform()

    @property
    def no_data_value(self):
        return self.source.GetRasterBand(1).GetNoDataValue()

    @property
    def raster_x_size(self):
        return self.source.RasterXSize

    @property
    def raster_y_size(self):
        return self.source.RasterYSize

    @property
    def block_size(self):
        return self.source.GetRasterBand(1).GetBlockSize()

    def __len__(self):
        block_size = self.block_size
        blocks_x = -(-self.raster_x_size // block_size[0])
        blocks_y = -(-self.raster_y_size // block_size[1])
        return blocks_x * blocks_y

    def partition(self):
        """Return generator of (xoff, xsize), (yoff, ysize) values."""
        self.gr = GridH5ResultAdmin(self.gridadmin_path, self.results_3di_path)
        nodes2 = self.gr.nodes.subset(SUBSET_2D_OPEN_WATER)
        timeseries2 = nodes2.timeseries(indexes=[self.calculation_step])
        data2 = timeseries2.only("cell_coords").data
        cell_coords2 = data2["cell_coords"]
        STEP = int(np.max(cell_coords2[2, :] - cell_coords2[0, :]))

        def offset_size_range(stop, step):
            for start in range(0, stop, step):
                yield start, min(step, stop - start)

        # tiled tiff writing is much faster row-wise
        raster_size = self.raster_y_size, self.raster_x_size
        block_size = self.block_size[::-1]

        total_size = block_size[0] * block_size[1]
        if block_size[0] <= STEP:
            block_size[0] = STEP
        else:
            block_size[0] = block_size[0] - (block_size[0] % STEP)
        if block_size[1] < STEP:
            block_size[1] = STEP

        offset_size_range
        generator = product(*map(offset_size_range, raster_size, block_size))

        total = len(self)
        for count, result in enumerate(generator, start=1):
            yield result[::-1]
            if self.progress_func is not None:
                self.progress_func(count / total)

    def convert_using(self, calculator):
        """Convert data writing it to tiff. """
        no_data_value = self.no_data_value
        for (xoff, xsize), (yoff, ysize) in self.partition():
            values = self.source.ReadAsArray(
                xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize
            )
            indices = (yoff, xoff), (yoff + ysize, xoff + xsize)
            result = calculator(
                indices=indices, values=values, no_data_value=no_data_value
            )

            self.target.GetRasterBand(1).WriteArray(array=result, xoff=xoff, yoff=yoff)


calculator_classes = {
    MODE_COPY: CopyCalculator,
    MODE_NODGRID: NodGridCalculator,
    MODE_CONSTANT_S1: ConstantLevelCalculator,
    MODE_INTERPOLATED_S1: InterpolatedLevelCalculator,
    MODE_CONSTANT: ConstantLevelDepthCalculator,
    MODE_INTERPOLATED: InterpolatedLevelDepthCalculator,
    **au_calculator_classes,
}


def calculate_waterdepth(
    gridadmin_path,
    results_3di_path,
    dem_path,
    waterdepth_path,
    calculation_step=-1,
    mode=MODE_INTERPOLATED,
    progress_func=None,
):
    """Calculate waterdepth and save it as GeoTIFF.
    Args:
        gridadmin_path (str): Path to gridadmin.h5 file.
        results_3di_path (str): Path to results_3di.nc file.
        dem_path (str): Path to dem.tif file.
        waterdepth_path (str): Path to waterdepth.tif file.
        calculation_step (int): Calculation step (default: -1 (last))
        interpolate (bool): Interpolate linearly between nodes.
    """
    try:
        CalculatorClass = calculator_classes[mode]
    except KeyError:
        raise ValueError("Unknown mode: '%s'" % mode)

    # TODO remove at some point, newly produced gridadmins don't need it
    fix_gridadmin(gridadmin_path)

    converter_kwargs = {
        "gridadmin_path": gridadmin_path,
        "results_3di_path": results_3di_path,
        "calculation_step": calculation_step,
        "source_path": dem_path,
        "target_path": waterdepth_path,
        "progress_func": progress_func,
    }

    with GeoTIFFConverter(**converter_kwargs) as converter:

        calculator_kwargs = {
            "gridadmin_path": gridadmin_path,
            "results_3di_path": results_3di_path,
            "calculation_step": calculation_step,
            "dem_geo_transform": converter.geo_transform,
            "dem_pixelsize": converter.geo_transform[1],
            "dem_shape": (converter.raster_y_size, converter.raster_x_size),
        }
        with CalculatorClass(**calculator_kwargs) as calculator:
            converter.convert_using(calculator)


### Bergermeer Data Set ###
#''' # coarse grid
calculate_waterdepth(
    gridadmin_path="Bergermeer_Eva_Coarsegrid_gridadmin.h5",
    results_3di_path="Bergermeer_Eva_Coarsegrid_results_3di.nc",
    dem_path="dem_test_5m.tif",
    waterdepth_path="test2.tif",  # "Bergermeer_Eva_Coarsegrid_1h_Bilin_Depth.tif",#Combi_Depth
    calculation_step=60,  # 60,#(1h), -1
    mode="au-squares",  # MODE_MY_INTERP_2_DEPTH, #MODE_INTERPOLATED_S1,#MODE_CONSTANT_S1,
    progress_func=gdal.TermProgress_nocb,
)
#'''
